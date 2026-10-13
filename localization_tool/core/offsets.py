"""Channel offset from a bead image (D55).

Multicolour beads (e.g. 0.1 µm TetraSpeck) are recorded with the same settings as the cells. Each
bead appears in every channel; the shift of its centre between channels is the chromatic offset of
the microscope. The offset of every channel to a reference channel is the median over all beads that
are found in both channels (Kozubek and Matula 2000, doi:10.1046/j.1365-2818.2000.00754.x; Matsuda
et al. 2018, doi:10.1038/s41598-018-25922-7). Positions are refined below the voxel size with an
intensity-weighted centre, the correction itself later moves whole voxels only (analysis._shift).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .segmentation import smooth

Voxel = tuple[float, float, float]
MATCH_UM = 0.5            # a bead in another channel further away than this is not the same bead
WINDOW_UM = (0.5, 0.25, 0.25)
MIN_BEADS = 3


def find_beads(volume: np.ndarray, voxel_um: Voxel, max_beads: int = 200) -> np.ndarray:
    """Bead centres in µm (z, y, x), refined by the intensity-weighted centre around each maximum.

    Maxima: local maxima of the lightly smoothed channel at least 10 robust noise widths above the
    median and at least 1 µm apart in x and y. Beads touching the stack border are left out.
    """
    sm = smooth(volume, voxel_um, 0.05)
    med = float(np.median(sm))
    noise = max(1.4826 * float(np.median(np.abs(sm - med))), 1e-6)
    thr = med + 10.0 * noise
    size = [max(3, 2 * int(round(1.0 / v)) + 1) for v in voxel_um]
    size[0] = max(3, 2 * int(round(0.6 / voxel_um[0])) + 1)
    peaks = (sm == ndi.maximum_filter(sm, size=size)) & (sm > thr)
    idx = np.argwhere(peaks)
    if len(idx) > max_beads:
        idx = idx[np.argsort(sm[tuple(idx.T)])[::-1][:max_beads]]
    half = [max(1, int(round(w / v))) for w, v in zip(WINDOW_UM, voxel_um)]
    raw = np.asarray(volume, np.float64)
    out = []
    for p in idx:
        lo = [c - h for c, h in zip(p, half)]
        hi = [c + h + 1 for c, h in zip(p, half)]
        if any(a < 0 for a in lo) or any(b > n for b, n in zip(hi, raw.shape)):
            continue
        win = raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        w = np.clip(win - np.median(win), 0, None)
        if w.sum() <= 0:
            continue
        com = ndi.center_of_mass(w)
        out.append([(lo[k] + com[k]) * voxel_um[k] for k in range(3)])
    return np.asarray(out, float).reshape(-1, 3)


def estimate_offsets(data: np.ndarray, voxel_um: Voxel, reference: int, labels: list[str]) -> dict:
    """Offset of every channel to the reference channel, in µm.

    data: (C, Z, Y, X). Returns {"reference": label, "channels": {label: {"dz_um", "dy_um", "dx_um",
    "beads", "spread_um"}}}. spread_um: median absolute deviation of the per-bead offsets (z, y, x),
    a measure of the precision. A channel with fewer than three matched beads gets no offset.
    """
    ref = find_beads(data[reference], voxel_um)
    result = {"reference": labels[reference], "reference_beads": int(len(ref)), "channels": {}}
    for c in range(data.shape[0]):
        if c == reference:
            continue
        pos = find_beads(data[c], voxel_um)
        diffs = []
        if len(ref) and len(pos):
            for r in ref:
                d = np.linalg.norm(pos - r, axis=1)
                j = int(np.argmin(d))
                if d[j] <= MATCH_UM:
                    diffs.append(pos[j] - r)
        if len(diffs) < MIN_BEADS:
            result["channels"][labels[c]] = {"beads": len(diffs), "error": "fewer than 3 beads found in both channels"}
            continue
        diffs = np.asarray(diffs)
        med = np.median(diffs, axis=0)
        mad = 1.4826 * np.median(np.abs(diffs - med), axis=0)
        result["channels"][labels[c]] = {"dz_um": round(float(med[0]), 4), "dy_um": round(float(med[1]), 4),
                                         "dx_um": round(float(med[2]), 4), "beads": int(len(diffs)),
                                         "spread_um": [round(float(v), 4) for v in mad]}
    return result



def match_channels(bead_fluors: list[str], channels: list[tuple[str, str, str]]) -> tuple[list[str] | None, str]:
    """Label for each channel of a bead image.

    channels: (label, fluorophore, role) of the open image in file order, Ignore channels included.
    By fluorophore when every bead channel finds exactly one partner, else by position when the
    channel count is the same. Without an open image the fluorophore itself is the label (offsets are
    then stored per fluorophore). Returns (labels or None, how they were matched or why not).
    """
    if not channels:
        return list(bead_fluors), "by fluorophore (no image open)"
    bead = [f.strip().lower() for f in bead_fluors]
    mine = [(lab, (fl or "").strip().lower()) for lab, fl, _ in channels]
    out = []
    for f in bead:
        hits = [lab for lab, fl in mine if fl and fl == f]
        out.append(hits[0] if len(hits) == 1 else None)
    if all(out) and len(set(out)) == len(out):
        return out, "by fluorophore"
    if len(bead) == len(mine):
        return [lab for lab, _ in mine], "by position (fluorophore names differ)"
    return None, (f"The bead image has the channels {', '.join(bead_fluors)}, the open image "
                  f"{', '.join(fl or lab for lab, fl, _ in channels)}. Record the beads with the same channels, "
                  "or measure on the start page (Settings there: offsets per fluorophore).")

def offsets_as_params(result: dict) -> list[dict]:
    """The measured offsets in the form of AnalysisParams.channel_offsets."""
    return [{"channel": lab, "dz_um": v["dz_um"], "dy_um": v["dy_um"], "dx_um": v["dx_um"]}
            for lab, v in result.get("channels", {}).items() if "dz_um" in v]


def synthetic_beads(shape=(40, 160, 160), voxel_um=(0.13, 0.05, 0.05), shifts_um=((0, 0, 0), (0.15, 0.03, -0.02)),
                    n_beads: int = 12, sigma_um=(0.25, 0.08, 0.08), seed: int = 0) -> np.ndarray:
    """Bead image for tests and the self-test: one channel per entry of shifts_um."""
    rng = np.random.default_rng(seed)
    z, y, x = [np.arange(n) * v for n, v in zip(shape, voxel_um)]
    margin = 1.2                         # beads on a jittered grid, at least 1.5 µm apart
    gy = np.linspace(margin, shape[1] * voxel_um[1] - margin, 4)
    gx = np.linspace(margin, shape[2] * voxel_um[2] - margin, 4)
    grid = [(yy, xx) for yy in gy for xx in gx][:n_beads]
    centres = np.array([(rng.uniform(margin, shape[0] * voxel_um[0] - margin), yy + rng.uniform(-0.2, 0.2),
                         xx + rng.uniform(-0.2, 0.2)) for yy, xx in grid])
    chans = []
    for sh in shifts_um:
        img = np.zeros(shape)
        for c in centres:
            cz, cy, cx = c + np.asarray(sh)
            g = (np.exp(-((z - cz) ** 2) / (2 * sigma_um[0] ** 2))[:, None, None]
                 * np.exp(-((y - cy) ** 2) / (2 * sigma_um[1] ** 2))[None, :, None]
                 * np.exp(-((x - cx) ** 2) / (2 * sigma_um[2] ** 2))[None, None, :])
            img += 1000 * g
        img = rng.poisson(img + 5).astype(np.uint16)
        chans.append(img)
    return np.stack(chans)
