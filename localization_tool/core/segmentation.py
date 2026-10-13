"""Automatic masks for cell, nucleus and actin-rich region (METRICS.md §3, D6, D16, D18).

Classical pipeline: Gaussian smoothing in µm (anisotropic), one global threshold, morphology.
Every function returns the mask together with the threshold it used, so the value can be shown
and saved. Masks always live on the voxel grid of the crop: no resampling, no shift.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon2mask
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle

from .params import MaskParams

Voxel = tuple[float, float, float]
MAX_THRESHOLD_SAMPLE = 400_000


def smooth(volume: np.ndarray, voxel_um: Voxel, sigma_um: float) -> np.ndarray:
    """Gaussian smoothing with sigma given in µm, converted per axis. Returns float32."""
    vol = np.asarray(volume, dtype=np.float32)
    if sigma_um <= 0:
        return vol.copy()
    sigma = [sigma_um / s for s in voxel_um]
    return ndi.gaussian_filter(vol, sigma=sigma, mode="nearest")


def auto_threshold(values: np.ndarray, method: str) -> float:
    """Global threshold of a 1D sample. Constant or tiny samples give +inf (empty mask)."""
    values = np.asarray(values).ravel()
    if values.size > MAX_THRESHOLD_SAMPLE:
        # a fixed random sample gives the same threshold within a fraction of a grey level and
        # keeps the iterative Li method fast on large stacks
        idx = np.random.default_rng(0).choice(values.size, MAX_THRESHOLD_SAMPLE, replace=False)
        values = values[idx]
    values = values.astype(np.float64)
    if values.size < 16 or values.min() == values.max():
        return float("inf")
    if method == "otsu":
        return float(threshold_otsu(values))
    if method == "triangle":
        return float(threshold_triangle(values))
    if method == "li":
        return float(threshold_li(values))
    raise ValueError(f"no automatic threshold for method {method!r}")


def mask_threshold(smoothed: np.ndarray, mp: MaskParams, region: np.ndarray | None = None) -> float:
    """Threshold for one mask: automatic on the voxels of `region` (or all voxels), times factor."""
    if mp.method == "manual":
        return float(mp.manual_threshold)
    sample = smoothed[region] if region is not None else smoothed
    return auto_threshold(sample, mp.method) * mp.factor


def remove_small(mask: np.ndarray, voxel_um: Voxel, min_volume_um3: float) -> np.ndarray:
    if min_volume_um3 <= 0 or not mask.any():
        return mask
    labels, n = ndi.label(mask)
    sizes = np.bincount(labels.ravel())[1:] * float(np.prod(voxel_um))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_volume_um3
    return keep[labels]


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    labels, n = ndi.label(mask)
    if n == 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Same result as scipy.ndimage.binary_fill_holes (background connected along the axes to the
    array border is outside, all other background is a hole), but by labelling the background once
    instead of growing it step by step, which is much faster on large stacks."""
    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        return mask.copy()
    lab, n = ndi.label(~mask)
    border = np.zeros(n + 1, bool)
    for ax in range(mask.ndim):
        border[np.take(lab, 0, axis=ax)] = True
        border[np.take(lab, -1, axis=ax)] = True
    border[0] = False
    return ~border[lab]


def fill_holes_per_plane(mask: np.ndarray, max_area_px: float | None = None) -> np.ndarray:
    """Fill holes plane by plane. With max_area_px only holes up to that area are filled."""
    out = mask.copy()
    for i in range(out.shape[0]):
        filled = fill_holes(out[i])
        if max_area_px is None:
            out[i] = filled
            continue
        holes = filled & ~out[i]
        if not holes.any():
            continue
        lab, n = ndi.label(holes)
        sizes = np.bincount(lab.ravel())
        small = sizes <= max_area_px
        small[0] = False
        out[i] |= small[lab]
    return out


def exclusion_mask(polygons_yx: list, origin_zyx: tuple[int, int, int], shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Voxels excluded by user polygons (D18). Polygons are (N, 2) arrays of (y, x) in source-image
    voxel coordinates and apply to all planes. A voxel is excluded when its centre lies inside."""
    nz, ny, nx = shape_zyx
    out2d = np.zeros((ny, nx), bool)
    for poly in polygons_yx or []:
        p = np.asarray(poly, dtype=float)
        if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2:
            raise ValueError("an exclusion polygon needs at least three (y, x) points")
        local = p - np.array([origin_zyx[1], origin_zyx[2]], dtype=float)
        out2d |= polygon2mask((ny, nx), local)
    return np.broadcast_to(out2d, (nz, ny, nx)).copy()


CELL_THRESHOLD_MARGIN_UM = 1.0
CHANNEL_SPECK_UM3 = 0.5


def _bbox_with_margin(mask: np.ndarray, voxel_um: Voxel, margin_um: float) -> tuple[slice, slice, slice]:
    out = []
    for axis, size in enumerate(mask.shape):
        other = tuple(i for i in range(3) if i != axis)
        idx = np.flatnonzero(mask.any(axis=other))
        m = int(round(margin_um / voxel_um[axis]))
        out.append(slice(max(0, idx[0] - m), min(size, idx[-1] + m + 1)))
    return tuple(out)


SIGNAL_TO_NOISE = 10.0
CLIPPED_FRACTION = 0.05            # more voxels exactly 0: data clipped at 0 (Airyscan), noise not readable per voxel


def smoothed_noise(raw: np.ndarray, voxel_um: Voxel, sigma_um: float) -> float | None:
    """Noise width of a channel after smoothing, from the differences of neighbouring voxels (D77).

    Neighbouring voxels share the stained structure, so their difference holds almost only noise: its MAD
    over the square root of 2 is the noise of one voxel, whatever part of the image is stained. The
    Gaussian smoothing scales white noise by the square root of the sum of its squared weights. Returns
    None for data clipped at 0 (more than 5 % of the voxels exactly 0, as after Airyscan processing):
    there the differences are mostly 0 and say nothing about the noise."""
    r = np.asarray(raw)
    sample = r[:, ::2, :] if r.ndim == 3 else r
    if float(np.mean(sample == 0)) > CLIPPED_FRACTION:
        return None
    d = np.diff(sample.astype(np.float32), axis=-1).ravel()
    per_voxel = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    if sigma_um <= 0:
        return per_voxel
    sig = [sigma_um / s for s in voxel_um][-r.ndim:]
    size = [2 * int(4 * s) + 1 for s in sig]
    k = np.zeros(size, np.float32)
    k[tuple(n // 2 for n in size)] = 1.0
    k = ndi.gaussian_filter(k, sig)
    return per_voxel * float(np.sqrt((k.astype(np.float64) ** 2).sum()))


def has_signal(smoothed: np.ndarray, raw: np.ndarray | None = None, voxel_um: Voxel | None = None,
               sigma_um: float = 0.0) -> bool:
    """True if a channel shows real staining, not only noise: the brightest 0.1 % must lie at least
    10 noise widths above the median. Channels with only noise (e.g. a secondary antibody control)
    would otherwise add specks to the cell mask.

    Noise width: with the unsmoothed channel (raw, voxel_um, sigma_um) from neighbouring voxels (D77),
    otherwise and for data clipped at 0 from the spread of all voxels (MAD). The spread of all voxels
    also contains the staining itself: when stained structures fill much of the image (many nuclei in a
    colony), it rejected clearly visible stains."""
    v = smoothed[:, ::2, ::2].ravel() if smoothed.ndim == 3 else smoothed.ravel()
    med = float(np.median(v))
    noise = smoothed_noise(raw, voxel_um, sigma_um) if raw is not None and voxel_um is not None else None
    if noise is None:
        noise = 1.4826 * float(np.median(np.abs(v - med)))
    return float(np.percentile(v, 99.9)) - med >= SIGNAL_TO_NOISE * max(noise, 1.0)


def segment_cell(channels: list[np.ndarray], voxel_um: Voxel, mp: MaskParams,
                 excluded: np.ndarray | None = None, variants: tuple[float, ...] = ()):
    """Cell mask (D28): everywhere at least one channel shows signal above its own threshold.

    Each channel is smoothed and thresholded on its own (Li by default), small specks below
    0.5 µm³ are dropped, and the channel masks are combined. So the cell edge lies where every
    stain has faded into the dark background, also at a bright actin front or a thin protrusion.
    Then holes are filled, excluded regions removed and the largest connected part is kept.
    Second pass: the thresholds are recomputed on the box around the first mask plus 1 µm, so
    the result hardly depends on how much background the image contains.
    With method "halfmax" (D54) the Li mask is only the start: each channel then gets the threshold
    halfway between its background and its level just inside the Li edge (edge_halfmax_threshold).
    These thresholds may only trim the outer shell of the Li cell (EDGE_TRIM_UM below its envelope),
    where the blur lies. Deeper voxels stay cell even where every stain is dim, e.g. cytoplasm between
    bright granules, which a global half maximum would otherwise cut out of the cell interior.
    Returns the mask and one threshold per input channel (NaN for a channel without staining).
    variants: extra factors (e.g. 0.8, 1.2) for the sensitivity span (D56); then a third value is returned,
    {factor: mask}, built from the same smoothed channels with every threshold times that factor.
    """
    smoothed = [smooth(ch, voxel_um, mp.sigma_um) for ch in channels]
    used = [i for i, sm in enumerate(smoothed) if has_signal(sm, channels[i], voxel_um, mp.sigma_um)] or [0]
    region = None if excluded is None else ~excluded
    li_mp = MaskParams(sigma_um=mp.sigma_um, method="li", factor=1.0,
                       min_volume_um3=mp.min_volume_um3) if mp.method == "halfmax" else mp

    def combine(region_mask, thresholds=None):
        union = np.zeros(smoothed[0].shape, bool)
        thrs = [float("nan")] * len(smoothed)     # NaN: channel without staining, not used
        for i in used:
            sm = smoothed[i]
            thr = thresholds[i] if thresholds is not None else mask_threshold(sm, li_mp, region_mask)
            thrs[i] = thr
            m = sm > thr
            if excluded is not None:
                m &= ~excluded
            union |= remove_small(m, voxel_um, CHANNEL_SPECK_UM3)
        return union, thrs

    def finish(union):
        mask = fill_holes(fill_holes_per_plane(union))
        if excluded is not None:
            mask &= ~excluded
        mask = largest_component(mask)
        return remove_small(mask, voxel_um, mp.min_volume_um3)

    def finish_closed(union):            # like finish, cavities at a stack end count as enclosed
        mask = fill_closed_ends(fill_holes_per_plane(union))
        if excluded is not None:
            mask &= ~excluded
        mask = largest_component(mask)
        return remove_small(mask, voxel_um, mp.min_volume_um3)

    union, thrs = combine(region)
    first = largest_component(union)
    if first.any() and mp.method != "manual":
        box = _bbox_with_margin(first, voxel_um, CELL_THRESHOLD_MARGIN_UM)
        local = np.zeros(first.shape, bool)
        local[box] = True
        if excluded is not None:
            local &= ~excluded
        union, thrs = combine(local)
    mask = finish(union)
    core = None
    if mp.method == "halfmax" and mask.any():
        hm = [float("nan")] * len(smoothed)
        geometry = edge_geometry(mask, voxel_um, excluded)          # the same for every channel
        for i in used:
            hm[i] = edge_halfmax_threshold(smoothed[i], mask, voxel_um, thrs[i], excluded, geometry) * mp.factor
        env = cell_envelope(mask, voxel_um)
        box = _bbox_with_margin(env, voxel_um, 1.0)                 # distances only near the cell (faster)
        core = np.zeros(env.shape, bool)
        core[box] = ndi.distance_transform_edt(env[box], sampling=voxel_um) > EDGE_TRIM_UM   # stack ends are no surface
        if excluded is not None:
            core &= ~excluded
        union, thrs = combine(None, hm)
        mask = finish_closed(union | core)
    if variants:
        if core is not None:
            extra = {f: finish_closed(combine(None, [t * f for t in thrs])[0] | core) for f in variants}
        else:
            extra = {f: finish(combine(None, [t * f for t in thrs])[0]) for f in variants}
        return mask, thrs, extra
    return mask, thrs


EDGE_TRIM_UM = 0.5                 # the half maximum trims only this outer shell of the Li cell (> Li overshoot 0.33 µm)
EDGE_CRACK_UM = 0.3                # cracks narrower than this are sealed when the outer envelope is found


def fill_closed_ends(mask: np.ndarray) -> np.ndarray:
    """3D hole filling with both stack ends closed: a cell rests on the coverslip and continues past a
    cut stack end, so a dark pocket that reaches the first or last plane is still inside the cell."""
    if mask.shape[0] < 2:
        return fill_holes(mask)
    pad = np.concatenate([fill_holes(mask[0])[None], mask, fill_holes(mask[-1])[None]], 0)
    return fill_holes(pad)[1:-1] | mask


def cell_envelope(mask: np.ndarray, voxel_um: Voxel) -> np.ndarray:
    """Outer envelope of a cell mask: narrow cracks sealed in each plane, holes filled per plane and in
    3D with closed stack ends. Depth below this envelope tells blur shell from cell interior (D54)."""
    r = max(1, int(round(EDGE_CRACK_UM / min(voxel_um[1], voxel_um[2]))))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = ((yy ** 2 + xx ** 2) <= r ** 2)[None]
    out = np.zeros(mask.shape, bool)
    if not mask.any():
        return out
    box = _xy_crop(mask, 2)                        # only around the cell: same result, much faster
    m = mask[box]
    closed = ndi.binary_closing(np.pad(m, ((0, 0), (r, r), (r, r))), structure=disk)[:, r:-r, r:-r]
    out[box] = fill_closed_ends(fill_holes_per_plane(closed | m))
    return out


def _xy_crop(mask: np.ndarray, margin_px: int) -> tuple[slice, slice, slice]:
    """All planes, the rows and columns of the mask plus a margin. Background outside the crop is
    connected to the border anyway, so hole filling inside the crop gives the same result."""
    ys = np.flatnonzero(mask.any(axis=(0, 2)))
    xs = np.flatnonzero(mask.any(axis=(0, 1)))
    ny, nx = mask.shape[1:]
    return (slice(None), slice(max(0, ys[0] - margin_px), min(ny, ys[-1] + margin_px + 1)),
            slice(max(0, xs[0] - margin_px), min(nx, xs[-1] + margin_px + 1)))


EDGE_BACKGROUND_UM = 1.0           # background: at least this far outside the Li cell
EDGE_RIM_UM = (0.3, 0.8)           # level: signal voxels this deep inside the Li cell
EDGE_RIM_MIN_VOXELS = 100


def edge_geometry(cell: np.ndarray, voxel_um: Voxel, excluded: np.ndarray | None = None):
    """Box, background voxels and rim voxels of a cell for edge_halfmax_threshold. Computed once per
    cell, because the two distance maps are the slow part and do not depend on the channel."""
    box = _bbox_with_margin(cell, voxel_um, 3.0)
    c = cell[box]
    bg_voxels = ndi.distance_transform_edt(~c, sampling=voxel_um) >= EDGE_BACKGROUND_UM
    if excluded is not None:
        bg_voxels &= ~excluded[box]
    depth = ndi.distance_transform_edt(c, sampling=voxel_um)
    rim = c & (depth >= EDGE_RIM_UM[0]) & (depth <= EDGE_RIM_UM[1])
    return box, bg_voxels, rim


def edge_halfmax_threshold(sm: np.ndarray, cell: np.ndarray, voxel_um: Voxel, li_threshold: float,
                           excluded: np.ndarray | None = None, geometry=None) -> float:
    """Cell-edge threshold of one channel halfway between background and the level just inside the
    edge (D54). The edge of a blurred object lies at half of the step between outside and inside, Li
    puts it into the blur halo, so cells came out 13 to 17 % too large (METRICS §4.10).

    Background: median of the voxels at least 1 µm outside the Li cell (within the box around it
    extended by 3 µm). Level: median of the voxels 0.3 to 0.8 µm inside the Li edge where the channel
    is above its Li threshold, i.e. the stain as it is near the edge, not in the bright cell centre,
    so dim peripheral cytoplasm is kept. A channel that hardly reaches the edge (DAPI) falls back to
    its median above Li inside the cell. geometry: edge_geometry(cell, ...) if already computed.
    """
    box, bg_voxels, rim_geo = geometry if geometry is not None else edge_geometry(cell, voxel_um, excluded)
    c, s = cell[box], sm[box]
    if bg_voxels.sum() < EDGE_RIM_MIN_VOXELS:
        return li_threshold
    bg = float(np.median(s[bg_voxels]))
    rim = rim_geo & (s > li_threshold)
    if rim.sum() >= EDGE_RIM_MIN_VOXELS:
        level = float(np.median(s[rim]))
    else:
        inner = c & (s > li_threshold)
        if inner.sum() < EDGE_RIM_MIN_VOXELS:
            return li_threshold
        level = float(np.median(s[inner]))
    if level <= bg:
        return li_threshold
    return bg + 0.5 * (level - bg)


HALFMAX_ITERATIONS = 6
HALFMAX_BACKGROUND_UM = 0.5        # cytoplasm level from cell voxels at least this far from the nucleus


def halfmax_threshold(sm: np.ndarray, cell: np.ndarray, voxel_um: Voxel) -> float:
    """Nucleus threshold halfway between the cytoplasm level and the nuclear DAPI level (D51).

    The edge of a uniformly stained object, blurred by the microscope, lies where the signal has
    fallen to half of the step between object and surroundings. Levels: the median smoothed DAPI
    inside the current nucleus mask and the median of the cell voxels at least 0.5 µm away from it.
    Starts at Otsu and is iterated until the threshold settles. It depends much less than Otsu on how
    large the nucleus is relative to the cell, so in compact cells it does not cut into the chromatin.
    With less than about 0.5 µm of cytoplasm around the nucleus the cytoplasm level falls back to the
    10th percentile outside the nucleus, and the edge can lie slightly inside. The levels are statistics, so they are taken on every second voxel in x and y (faster).
    """
    thr = auto_threshold(sm[cell], "otsu")
    if not np.isfinite(thr):
        return thr
    s2, c2 = sm[:, ::2, ::2], cell[:, ::2, ::2]
    v2 = (voxel_um[0], voxel_um[1] * 2, voxel_um[2] * 2)
    for _ in range(HALFMAX_ITERATIONS):
        nuc = (s2 > thr) & c2
        if not nuc.any() or nuc.sum() == c2.sum():
            return thr
        far = (ndi.distance_transform_edt(~nuc, sampling=v2) >= HALFMAX_BACKGROUND_UM) & c2
        if far.sum() >= 0.02 * c2.sum():
            low = float(np.median(s2[far]))
        else:                               # the nucleus fills nearly the whole cell
            low = float(np.percentile(s2[c2 & ~nuc], 10))
        high = float(np.median(s2[nuc]))
        new = low + 0.5 * (high - low)
        if abs(new - thr) <= 0.005 * max(abs(thr), 1e-9):
            return new
        thr = new
    return thr


def fill_dim_holes(mask: np.ndarray, sm: np.ndarray, level: float) -> np.ndarray:
    """Fill holes per plane, of any size, whose mean smoothed DAPI reaches `level` (D51).
    Euchromatin is dimmer than heterochromatin but still DAPI positive. A ring-shaped nucleus keeps
    its centre: there the DAPI drops to the cytoplasm level."""
    out = mask.copy()
    for i in range(out.shape[0]):
        holes = fill_holes(out[i]) & ~out[i]
        if not holes.any():
            continue
        lab, n = ndi.label(holes)
        means = ndi.mean(sm[i], lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = np.asarray(means) >= level
        out[i] |= keep[lab]
    return out


def segment_nucleus(dapi: np.ndarray, voxel_um: Voxel, mp: MaskParams, cell: np.ndarray,
                    max_hole_um2: float = 0.25, close_um: float = 0.2, dim_hole_fraction: float = 0.0,
                    coverslip_first: bool | None = True, variants: tuple[float, ...] = ()):
    """Nucleus mask inside the cell. Threshold from the voxels of the cell.

    Default threshold: half maximum between cytoplasm and nucleus (D51), Otsu, Li and Triangle stay
    selectable. Dim chromatin inside the nucleus is filled: holes up to max_hole_um2 per plane, every
    cavity that is closed in 3D and, optionally (off by default), holes of any size whose DAPI reaches
    dim_hole_fraction times the threshold. That option can fill the centre of a ring-shaped nucleus
    and gained at most 0.33 µm³ on the test images, so it stays off. The plane at the end of the stack away from the coverslip counts
    as closed, because a stack that ends inside the nucleus would otherwise leave dim regions open
    towards the stack edge. The coverslip end stays open (D52): a gap between two lobes that opens
    onto the coverslip is cytoplasm. Cracks narrower than 2 * close_um that connect such a region to
    the cytoplasm are closed for this test. A tunnel through the nucleus that connects to the
    cytoplasm stays open: ring-shaped mouse neutrophil nuclei keep the cytoplasm in their centre. All
    lobes above the minimum volume are kept, not only the largest.
    variants: extra factors on the threshold (sensitivity span, D56); then {factor: mask} is returned third.
    """
    sm = smooth(dapi, voxel_um, mp.sigma_um)
    if not cell.any():
        empty = np.zeros(cell.shape, bool)
        return (empty, float("inf"), {f: empty for f in variants}) if variants else (empty, float("inf"))
    if mp.method == "halfmax":
        thr = halfmax_threshold(sm, cell, voxel_um) * mp.factor
    else:
        thr = mask_threshold(sm, mp, cell)

    def build(t: float) -> np.ndarray:
        mask = (sm > t) & cell
        mask = fill_holes_per_plane(mask, max_hole_um2 / (voxel_um[1] * voxel_um[2]))
        if dim_hole_fraction > 0 and np.isfinite(t):
            mask = fill_dim_holes(mask, sm, dim_hole_fraction * t) & cell
        mask = _fill_enclosed(mask, voxel_um, close_um, coverslip_first) & cell
        return remove_small(mask, voxel_um, mp.min_volume_um3)

    mask = build(thr)
    if variants:
        return mask, thr, {f: build(thr * f) for f in variants}
    return mask, thr


def _fill_enclosed(mask: np.ndarray, voxel_um: Voxel, close_um: float,
                   coverslip_first: bool | None = True) -> np.ndarray:
    """Fill cavities closed in 3D after sealing narrow cracks. The stack end away from the coverslip
    counts as closed, the coverslip end stays open (D52). Unknown orientation (None): both ends closed."""
    if not mask.any():
        return mask
    full = mask
    box = _xy_crop(mask, 2)                       # only around the nucleus: same result, faster
    mask = mask[box]
    hull = fill_holes_per_plane(mask)
    closed = mask
    if close_um > 0:
        r = max(1, int(round(close_um / min(voxel_um[1], voxel_um[2]))))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (yy ** 2 + xx ** 2) <= r ** 2
        closed = ndi.binary_closing(np.pad(mask, ((0, 0), (r, r), (r, r))), structure=disk[None])[:, r:-r, r:-r]
        closed = mask | (closed & hull)          # seal cracks only inside the nucleus outline
    lid = np.pad(closed, ((1, 1), (0, 0), (0, 0)), mode="constant", constant_values=True)
    if coverslip_first is not None:
        if coverslip_first:
            lid[0] = False                       # plane 0 at the coverslip: open
        else:
            lid[-1] = False
    filled = fill_holes(lid)[1:-1]
    cavities = filled & ~closed
    if not cavities.any():
        return full
    out = full.copy()
    out[box] = mask | cavities | (closed & ~mask & ndi.binary_dilation(cavities))
    return out


PLANE_MIN_VOXELS = 200             # a plane needs this many cell voxels for its own actin threshold (D70)
PLANE_MIN_CONTRAST = 2.0           # and its actin at least twice as bright as its cytoplasm (else: no actin of its own)
PLANE_CYTOPLASM_DEPTH_UM = 0.3     # cytoplasm of a plane: voxels at least this far inside the cell edge of the plane


def plane_halfmax_thresholds(sm: np.ndarray, cell: np.ndarray, start: float, voxel_um: Voxel) -> np.ndarray:
    """Actin threshold of every plane halfway between its cytoplasm and its actin (D70), the rule of the
    nucleus (D51) applied per plane: start at Otsu on the cell voxels of the plane, then the median of the
    voxels above the threshold (actin) and below it (cytoplasm) give a new threshold, until it settles.
    A plane keeps `start` (the threshold of the whole cell) when it has too few cell voxels, when it is
    almost only actin or almost no actin, or when its actin is less than twice as bright as its cytoplasm,
    so that noise in a plane without cortex does not become actin. The cytoplasm level comes from voxels
    at least 0.3 µm inside the cell edge of the plane: the blurred rim at the top or bottom of a cell is
    dim, and against it plain cytoplasm would look like actin."""
    out = np.full(cell.shape[0], float(start))
    for z in range(cell.shape[0]):
        c = cell[z]
        if c.sum() < PLANE_MIN_VOXELS:
            continue
        s = sm[z]
        deep = c & (ndi.distance_transform_edt(c, sampling=voxel_um[1:]) >= PLANE_CYTOPLASM_DEPTH_UM)
        thr = auto_threshold(s[c], "otsu")
        if not np.isfinite(thr):
            continue
        ok = False
        for _ in range(HALFMAX_ITERATIONS):
            a = c & (s > thr)
            rest = deep & ~a
            if a.sum() < PLANE_MIN_VOXELS // 4 or rest.sum() < PLANE_MIN_VOXELS // 4:
                ok = False
                break
            low, high = float(np.median(s[rest])), float(np.median(s[a]))
            ok = high >= PLANE_MIN_CONTRAST * max(low, 1e-9)
            new = low + 0.5 * (high - low)
            done = abs(new - thr) <= 0.005 * max(abs(thr), 1e-9)
            thr = new
            if done:
                break
        if ok:
            out[z] = thr
    return out


def segment_actin_planes(actin: np.ndarray, voxel_um: Voxel, mp: MaskParams,
                         cell: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Actin compartment (D28, D70): voxels above the threshold of the whole cell (Li) OR above the
    half-maximum threshold of their own plane. The stain of a flat cell is often several times brighter
    in the planes at the coverslip than in the cortex above; one threshold for the stack then takes only
    fragments of the upper cortex. The per-plane threshold finds the cortex in every plane, the whole-cell
    threshold keeps a plane that consists almost only of actin (lamella at the coverslip).
    Returns the mask, the whole-cell threshold and the threshold used per plane (the lower of the two)."""
    sm = smooth(actin, voxel_um, mp.sigma_um)
    if not cell.any():
        return np.zeros(cell.shape, bool), float("inf"), np.full(cell.shape[0], float("inf"))
    thr = mask_threshold(sm, mp, cell)
    if mp.method == "manual":
        per_plane = np.full(cell.shape[0], thr)
    else:
        per_plane = np.minimum(thr, plane_halfmax_thresholds(sm, cell, thr / mp.factor, voxel_um) * mp.factor)
    mask = (sm > per_plane[:, None, None]) & cell
    mask = remove_small(mask, voxel_um, mp.min_volume_um3)
    return mask, thr, per_plane


def segment_actin(actin: np.ndarray, voxel_um: Voxel, mp: MaskParams, cell: np.ndarray) -> tuple[np.ndarray, float]:
    """Actin mask inside the cell: actin above a threshold computed from the voxels of the cell.
    Used twice (D28): with Li for all actin (compartment), with Otsu for the actin-rich part."""
    sm = smooth(actin, voxel_um, mp.sigma_um)
    if not cell.any():
        return np.zeros(cell.shape, bool), float("inf")
    thr = mask_threshold(sm, mp, cell)
    mask = (sm > thr) & cell
    mask = remove_small(mask, voxel_um, mp.min_volume_um3)
    return mask, thr
