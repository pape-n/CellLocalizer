"""Analysis of one cell crop: masks, compartments, measurements, notes.

This is the single computational path used by the command line, the tests and later the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import coloc, crosstalk, distances, metrics, polarity, qc, segmentation
from .channels import ChannelAssignment
from .image import ImageStack
from .params import AnalysisParams


@dataclass
class CellResult:
    masks: dict[str, np.ndarray]            # cell, nucleus, actin, actin_rich, labels (compartments)
    thresholds: dict[str, float]
    cell_values: dict
    target_values: dict[str, dict]          # target label -> measurements
    profiles: dict[str, dict[str, list]]    # target label -> profile name -> rows
    notes: list[str]
    voxel_um: tuple[float, float, float]    # effective voxel size used (axial scale applied)
    coverslip_first: bool
    origin: tuple[int, int, int]
    info: dict = field(default_factory=dict)
    overlays: dict = field(default_factory=dict)                 # for drawing: polarity axis


def effective_voxel(stack: ImageStack, params: AnalysisParams) -> tuple[float, float, float]:
    dz, dy, dx = stack.voxel_um
    return (dz * params.axial_scale, dy, dx)


def coverslip_first(stack: ImageStack, params: AnalysisParams) -> tuple[bool, str]:
    if params.z_reference == "first_plane":
        return True, "setting: first plane at the coverslip"
    if params.z_reference == "last_plane":
        return False, "setting: last plane at the coverslip"
    if "coverslip_first" in stack.info:
        return bool(stack.info["coverslip_first"]), stack.info.get("z_orientation_source", "file metadata")
    return True, "default: first plane at the coverslip (no metadata)"


def _roles(assignments: list[ChannelAssignment]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, a in enumerate(assignments):
        out.setdefault(a.role, []).append(i)
    return out


def _dtype_max(dtype: np.dtype) -> float | None:
    return float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else None


def _xy_box(mask: np.ndarray, voxel_um, margin_um: float) -> tuple[int, int, int, int]:
    _, ny, nx = mask.shape
    if not mask.any():
        return 0, ny, 0, nx
    ys = np.flatnonzero(mask.any(axis=(0, 2)))
    xs = np.flatnonzero(mask.any(axis=(0, 1)))
    my, mx = int(np.ceil(margin_um / voxel_um[1])), int(np.ceil(margin_um / voxel_um[2]))
    return max(0, ys[0] - my), min(ny, ys[-1] + my + 1), max(0, xs[0] - mx), min(nx, xs[-1] + mx + 1)


def _shift(volume: np.ndarray, offset_um: tuple[float, float, float], voxel_um) -> np.ndarray:
    """Move a channel back by its measured offset (D55), in whole voxels without interpolation, so the
    voxel values stay unchanged (D18). At the border of the stack or crop the outermost plane, row or
    column is repeated (a 0 there would count as missing signal inside a cell that touches the stack end)."""
    steps = [int(round(-d / v)) if v > 0 else 0 for d, v in zip(offset_um, voxel_um)]
    if not any(steps):
        return volume
    idx = [np.clip(np.arange(n) - k, 0, n - 1) for k, n in zip(steps, volume.shape)]
    return volume[np.ix_(*idx)]


def channel_offset(params: AnalysisParams, label: str, fluor: str | None = None) -> tuple[float, float, float] | None:
    """Offset of a channel: by its label in this analysis, else by its fluorophore (offsets measured on
    the start page are stored per fluorophore, because the optics shift a colour, not a protein)."""
    found = None
    for o in params.channel_offsets:
        if not any(o[k] for k in ("dz_um", "dy_um", "dx_um")):
            continue
        if o["channel"] == label:
            return (o["dz_um"], o["dy_um"], o["dx_um"])
        if fluor and found is None and str(o.get("fluor") or "").strip().lower() == fluor.strip().lower():
            found = (o["dz_um"], o["dy_um"], o["dx_um"])
    return found


def applied_offsets(params: AnalysisParams, assignments: list[ChannelAssignment], channels) -> list[dict]:
    """The offsets actually applied to the channels of this image, by label."""
    out = []
    for i, a in enumerate(assignments):
        fl = channels[i].fluor if channels is not None and i < len(channels) else None
        off = channel_offset(params, a.label, fl)
        if off and a.role != "Ignore":
            out.append({"channel": a.label, "dz_um": off[0], "dy_um": off[1], "dx_um": off[2]})
    return out


def _intensity(data: np.ndarray, assignments: list[ChannelAssignment], params: AnalysisParams, i: int,
               voxel_um=None, channels=None) -> np.ndarray:
    """Background-subtracted intensity of channel i, aligned by the optional channel offset (D55) and
    with the optional crosstalk correction (D27, D38). Works on a copy, the raw data stay unchanged.
    voxel_um: nominal voxel size of the stack (needed only for channel offsets). channels: the
    ChannelInfo of the stack, for offsets stored per fluorophore."""
    labels = [a.label for a in assignments]

    def aligned(j: int) -> np.ndarray:
        fl = channels[j].fluor if channels is not None and j < len(channels) else None
        off = channel_offset(params, labels[j], fl) if voxel_um is not None else None
        return _shift(data[j], off, voxel_um) if off else data[j]

    label = labels[i]
    out = metrics.background_subtract(aligned(i), params.background_for(label))
    for c in params.crosstalk:
        if c["into"] == label and c["from"] in labels and c["k"] > 0:
            j = labels.index(c["from"])
            src = metrics.background_subtract(aligned(j), params.background_for(c["from"]))
            out = crosstalk.apply(out, src, c["k"])
    return out


SENSITIVITY_FACTORS = (0.8, 1.2)


def cell_mask(stack: ImageStack, assignments: list[ChannelAssignment], params: AnalysisParams | None = None,
              exclusions_yx: list | None = None, variants: tuple[float, ...] = ()):
    """Only the cell mask (step 3 of the workflow, D28, D29). Returns the mask and the thresholds, and with
    variants also {factor: mask} for the sensitivity span (D56)."""
    params = params or AnalysisParams()
    roles = _roles(assignments)
    voxel = effective_voxel(stack, params)
    excluded = None
    if exclusions_yx:
        excluded = segmentation.exclusion_mask(exclusions_yx, stack.origin, stack.shape_zyx)
    if "Membrane" in roles:
        idx = [roles["Membrane"][0]]
    else:
        idx = [i for i, a in enumerate(assignments) if a.role != "Ignore"]
    sources = [_intensity(stack.data, assignments, params, i, stack.voxel_um, stack.channels) for i in idx]
    res = segmentation.segment_cell(sources, voxel, params.cell, excluded, variants=variants)
    names = {f"cell_{assignments[i].label}": t for i, t in zip(idx, res[1])}
    return (res[0], names, res[2]) if variants else (res[0], names)


def analyze_cell(stack: ImageStack, assignments: list[ChannelAssignment], params: AnalysisParams | None = None,
                 exclusions_yx: list | None = None, masks: dict[str, np.ndarray] | None = None,
                 cell_variants: dict[float, np.ndarray] | None = None) -> CellResult:
    """Analyse one cell crop.

    stack: the crop (or a full image that contains one cell). Values are read, never changed.
    exclusions_yx: polygons in source-image (y, x) voxel coordinates that do not belong to the cell (D18).
    masks: optional corrected masks ("cell", "nucleus", "actin", "actin_rich") that replace the
    automatic ones. Without a corrected actin-rich mask it is computed and kept inside the actin mask.
    cell_variants: the cell masks at the sensitivity factors (from cell_mask with variants) when the cell
    mask is given, so a cell from step 3 of the workflow gets the same span as in a batch run.
    """
    params = params or AnalysisParams()
    if len(assignments) != stack.n_channels:
        raise ValueError("one channel assignment per channel is required")
    roles = _roles(assignments)
    targets = roles.get("Target", [])
    if not targets:
        raise ValueError("at least one channel needs the role Target")
    if len(roles.get("Nucleus", [])) > 1 or len(roles.get("Actin", [])) > 1 or len(roles.get("Membrane", [])) > 1:
        raise ValueError("Nucleus, Actin and Membrane can each be assigned to one channel only")

    voxel = effective_voxel(stack, params)
    first, z_source = coverslip_first(stack, params)
    shape = stack.shape_zyx
    raw = stack.data  # never modified

    cache: dict[int, np.ndarray] = {}

    def corrected(i: int) -> np.ndarray:
        # computed once per channel; callers only read it
        if i not in cache:
            cache[i] = _intensity(raw, assignments, params, i, stack.voxel_um, stack.channels)
        return cache[i]

    excluded = None
    if exclusions_yx:
        excluded = segmentation.exclusion_mask(exclusions_yx, stack.origin, shape)

    thresholds: dict[str, float] = {}
    cell_variants = dict(cell_variants or {})
    nucleus_variants: dict[float, np.ndarray] = {}
    masks = dict(masks or {})
    given_masks = set(masks)                 # corrected masks from the user, kept as they are

    # cell
    if "cell" in masks:
        cell = np.asarray(masks["cell"], bool)
        if excluded is not None:
            cell = cell & ~excluded
        thresholds["cell"] = float("nan")
        if params.sensitivity and cell_variants:
            cell_variants = {f: np.asarray(m, bool) for f, m in cell_variants.items()}
    else:
        if params.sensitivity:
            cell, cell_thr, cell_variants = cell_mask(stack, assignments, params, exclusions_yx, SENSITIVITY_FACTORS)
        else:
            cell, cell_thr = cell_mask(stack, assignments, params, exclusions_yx)
        thresholds.update(cell_thr)

    # Everything after the cell mask runs on the box around the cell plus 1 µm in x and y (all
    # planes). The measured values are identical, it is only faster on large images.
    full_cell = cell
    y0, y1, x0, x1 = _xy_box(cell, voxel, 1.0)
    cell = cell[:, y0:y1, x0:x1]
    raw = stack.data[:, :, y0:y1, x0:x1]
    shape = cell.shape
    masks = {k: np.asarray(v, bool)[:, y0:y1, x0:x1] for k, v in masks.items() if k != "cell"}

    # nucleus
    has_nucleus = "Nucleus" in roles
    if "nucleus" in masks:
        nucleus = np.asarray(masks["nucleus"], bool) & cell
        thresholds["nucleus"] = float("nan")
    elif has_nucleus and not segmentation.has_signal(
            segmentation.smooth(corrected(roles["Nucleus"][0]), voxel, params.nucleus.sigma_um),
            corrected(roles["Nucleus"][0]), voxel, params.nucleus.sigma_um):
        # a DAPI channel with noise only: any threshold would mark most of the cell as nucleus
        nucleus = np.zeros(shape, bool)
        thresholds["nucleus"] = float("nan")
    elif has_nucleus:
        res = segmentation.segment_nucleus(
            corrected(roles["Nucleus"][0]), voxel, params.nucleus, cell, params.nucleus_max_hole_um2,
            dim_hole_fraction=params.nucleus_dim_hole_fraction, coverslip_first=first,
            variants=SENSITIVITY_FACTORS if params.sensitivity else ())
        nucleus, thresholds["nucleus"] = res[0], res[1]
        nucleus_variants = res[2] if params.sensitivity else {}
    else:
        nucleus = np.zeros(shape, bool)
        thresholds["nucleus"] = float("nan")

    # surface depth is needed for the cortex shell and for actin definition (b)
    depth = distances.depth_below_surface(cell, voxel)

    # actin (compartment) and actin-rich (zone inside it), D28; per plane as well (D70)
    actin_planes = None
    if "actin" in masks:
        actin = np.asarray(masks["actin"], bool) & cell
        thresholds["actin"] = float("nan")
    elif params.actin_definition == "shell":
        actin = cell & (depth <= params.cortex_um)
        thresholds["actin"] = float("nan")
    elif "Actin" in roles and params.actin_per_plane:
        actin, thresholds["actin"], actin_planes = segmentation.segment_actin_planes(
            corrected(roles["Actin"][0]), voxel, params.actin, cell)
    elif "Actin" in roles:
        actin, thresholds["actin"] = segmentation.segment_actin(
            corrected(roles["Actin"][0]), voxel, params.actin, cell)
    else:
        actin = np.zeros(shape, bool)
        thresholds["actin"] = float("nan")

    if "Actin" in roles and "actin_rich" not in masks:
        actin_rich, thresholds["actin_rich"] = segmentation.segment_actin(
            corrected(roles["Actin"][0]), voxel, params.actin_rich, cell)
    elif "actin_rich" in masks:
        actin_rich = np.asarray(masks["actin_rich"], bool) & cell
        thresholds["actin_rich"] = float("nan")
    else:
        actin_rich = np.zeros(shape, bool)
        thresholds["actin_rich"] = float("nan")

    actin_rich &= actin                     # actin-rich is always part of the actin
    labels = metrics.compartment_labels(cell, nucleus, actin)
    overlap = nucleus & actin
    d_nucleus = distances.signed_distance(nucleus, voxel) if nucleus.any() else np.full(shape, np.nan, np.float32)
    height = distances.height_above_substrate(cell, voxel[0], first)
    v0 = float(np.prod(voxel))

    cell_values = metrics.cell_descriptors(cell, nucleus, actin, labels, voxel)
    cell_values["actin_rich_volume_um3"] = float(actin_rich.sum()) * float(np.prod(voxel))
    cell_values["base_plane"] = distances.base_plane(cell, first)
    # polarity from the actin channel (D36)
    actin_int = corrected(roles["Actin"][0]) if "Actin" in roles else None
    axis = None
    if params.polarity and actin_int is not None:
        pol = polarity.polarity(cell, actin_int, nucleus, voxel)
        axis = pol.pop("_axis")
        cell_values.update(pol)
    cell_volume_median = {
        "cell_median_distance_nucleus_um": _median(d_nucleus[cell]),
        "cell_median_depth_um": _median(depth[cell]),
        "cell_median_height_um": _median(height[cell]),
    }
    cell_values.update(cell_volume_median)

    target_values: dict[str, dict] = {}
    profiles: dict[str, dict[str, list]] = {}
    target_masks: dict[str, np.ndarray] = {}
    target_int: dict[str, np.ndarray] = {}
    for i in targets:
        label = assignments[i].label
        inten = corrected(i)
        target_int[label] = inten
        target_values[label] = metrics.measure_target(
            inten, labels, overlap, d_nucleus, depth, height, v0,
            params.envelope_um, params.cortex_um, params.basal_um, actin_rich=actin_rich)
        # target mask (D30): only a view on the brightest target signal, the shares above do not use it
        key = f"target_{label}"
        if key in masks:
            tmask = np.asarray(masks[key], bool) & cell
            thresholds[key] = float("nan")
        elif segmentation.has_signal(segmentation.smooth(inten, voxel, params.target.sigma_um), inten, voxel,
                                     params.target.sigma_um):
            tmask, thresholds[key] = segmentation.segment_actin(inten, voxel, params.target, cell)
        else:
            # no staining above noise (D53): a threshold would still mark a third of the cell
            tmask, thresholds[key] = np.zeros(shape, bool), float("nan")
        target_masks[label] = tmask
        measures = metrics.target_mask_measures(inten, labels, tmask, voxel)
        if not tmask.any() and np.isnan(thresholds[key]):
            measures = {k: float("nan") for k in measures}
        target_values[label].update(measures)
        profiles[label] = {
            "distance_to_nucleus_surface": metrics.profile(inten, d_nucleus, cell, params.profile_bin_um),
            "depth_below_cell_surface": metrics.profile(inten, depth, cell, params.profile_bin_um),
            "height_above_substrate": metrics.profile(inten, height, cell, params.profile_bin_um),
        }
        if params.polarity and actin_int is not None:
            target_values[label].update(polarity.target_front(inten, cell, axis, voxel))
        if params.colocalization and actin_int is not None:
            c = coloc.colocalization(inten, actin_int, cell)
            target_values[label].update({"pcc_actin": c["pcc"], "m1_actin": c["m1"], "m2_actin": c["m2"],
                                         "costes_threshold_target_vs_actin": c["threshold_a"],
                                         "costes_threshold_actin": c["threshold_b"],
                                         "zero_voxels_target_pct": c["zero_a_pct"],
                                         "zero_voxels_actin_pct": c["zero_b_pct"]})

    # two targets against each other (D35)
    if params.colocalization and len(target_int) > 1:
        labs = list(target_int)
        for a in labs:
            for b in labs:
                if a == b:
                    continue
                c = coloc.colocalization(target_int[a], target_int[b], cell)
                target_values[a].update({f"pcc_{b}": c["pcc"], f"m1_{b}": c["m1"], f"m2_{b}": c["m2"]})

    # sensitivity span (D56): main shares again with the cell and the nucleus edge at factor 0.8 and 1.2.
    # Only the edge in question moves: the variant cells keep the nucleus and the actin threshold of the
    # main result, the variant nuclei keep the cell and the actin. Smoothing is done once.
    span_cut = 0.0
    if params.sensitivity and cell.any():
        variants = []
        if cell_variants:
            actin_sm = None
            if "Actin" in roles and "actin" not in given_masks and params.actin_definition != "shell":
                actin_sm = segmentation.smooth(corrected(roles["Actin"][0]), voxel, params.actin.sigma_um)
            for f, c_full in cell_variants.items():
                c_box = c_full[:, y0:y1, x0:x1]
                if not c_box.any():
                    continue
                total = int(c_full.sum())
                if total:                     # the wider variant can reach past the analysis box
                    span_cut = max(span_cut, 1.0 - int(c_box.sum()) / total)
                if actin_sm is not None and np.isfinite(thresholds.get("actin", float("nan"))):
                    thr = actin_planes[:, None, None] if actin_planes is not None else thresholds["actin"]
                    a_box = segmentation.remove_small((actin_sm > thr) & c_box, voxel, params.actin.min_volume_um3)
                elif params.actin_definition == "shell" and "actin" not in given_masks:
                    a_box = c_box & (distances.depth_below_surface(c_box, voxel) <= params.cortex_um)
                else:
                    a_box = actin & c_box
                variants.append(metrics.compartment_labels(c_box, nucleus & c_box, a_box))
        for f, n_box in nucleus_variants.items():
            variants.append(metrics.compartment_labels(cell, n_box, actin))
        for label, inten in target_int.items():
            base = target_values[label]
            spans = [metrics.compartment_shares(inten, lab_v) for lab_v in variants]
            for key in metrics.SENSITIVITY_KEYS:
                vals = [v[key] for v in spans if np.isfinite(v[key])]
                if vals and np.isfinite(base.get(key, float("nan"))):
                    vals.append(base[key])     # no variant (e.g. corrected masks): empty, not a zero-width span
                base[f"{key}_min"] = min(vals) if vals else float("nan")
                base[f"{key}_max"] = max(vals) if vals else float("nan")
        cell_values["sensitivity_variants"] = len(variants)

    def embed(m: np.ndarray) -> np.ndarray:
        out = np.zeros(full_cell.shape, m.dtype)
        out[:, y0:y1, x0:x1] = m
        return out

    nucleus_full = embed(nucleus)
    raw_by_label = {assignments[i].label: stack.data[i] for i, a in enumerate(assignments) if a.role != "Ignore"}
    notes = qc.flags(full_cell, nucleus_full, raw_by_label, _dtype_max(stack.data.dtype), first,
                     params.truncation_fraction, params.saturation_fraction, has_nucleus)
    if cell.any():
        lows = [corrected(i) for i in roles.get("Actin", [])] + list(target_int.values())
        ratio = qc.bottom_ratio(cell, lows, first)
        cell_values["bottom_plane_ratio"] = ratio        # column only, no note (D67)
    if span_cut > 0.02:
        notes.append(f"sensitivity span: the wider cell edge reaches {100 * span_cut:.0f} % past the analysis box "
                     "(neighbour or haze?), the span may be too narrow")

    # Missing compartments are "no value", never 0 % (they would enter the means as real zeros)
    has_actin = bool(actin.any()) and ("Actin" in roles or "actin" in masks or params.actin_definition == "shell")
    if cell.any() and not nucleus.any():
        if not has_nucleus:
            notes.append("no nucleus channel: nucleus and cytoplasm values left empty")
        _blank(cell_values, target_values, NUCLEUS_KEYS)
    if cell.any() and not has_actin:
        notes.append("no actin channel: actin values left empty" if "Actin" not in roles and "actin" not in masks
                     else "no actin found: actin values left empty")
        _blank(cell_values, target_values, ACTIN_KEYS)
    for i, a in enumerate(assignments):          # D80: a camera offset that is still in the data
        if a.role != "Ignore" and i < raw.shape[0]:
            note = qc.dark_level_note(a.label, raw[i], params.background_for(a.label))
            if note:
                notes.append(note)
    if cell.any():
        notes += qc.cell_notes(stack.calibrated, nucleus, voxel,
                               corrected(roles["Nucleus"][0]) if has_nucleus else None, cell,
                               {lab: target_int[lab] for lab in target_int}, target_values,
                               params.target.sigma_um)   # same smoothing as the mask decision (D53)

    overlays: dict = {}
    if axis is not None:
        cy, cx = axis["centre_um"]
        uy, ux = axis["unit"]
        rad = axis["radius_um"]
        overlays["polarity"] = {"start": (cy / voxel[1] + y0, cx / voxel[2] + x0),
                                "end": ((cy + uy * rad) / voxel[1] + y0, (cx + ux * rad) / voxel[2] + x0)}
    labels_present = {a.label for a in assignments}
    applied = [c for c in params.crosstalk if c["k"] > 0 and {c["into"], c["from"]} <= labels_present]
    if applied:
        notes.append("crosstalk corrected: " + ", ".join(f"{c['into']} - {c['k']:g} x {c['from']}" for c in applied))

    return CellResult(
        masks={"cell": full_cell, "nucleus": nucleus_full, "actin": embed(actin), "actin_rich": embed(actin_rich),
               "labels": embed(labels), **{f"target_{k}": embed(v) for k, v in target_masks.items()}},
        thresholds=thresholds, cell_values=cell_values, target_values=target_values,
        profiles=profiles, notes=notes, voxel_um=voxel, coverslip_first=first,
        origin=stack.origin, info={"z_orientation": z_source, "acquisition": _acquisition_by_label(stack, assignments),
                                   "objective": (stack.info.get("acquisition") or {}).get("objective",
                                                                                     stack.info.get("objective", "")),
                                   "immersion": (stack.info.get("acquisition") or {}).get("immersion", ""),
                                   "channel_offsets": applied_offsets(params, assignments, stack.channels)},
        overlays=overlays,
    )


def _acquisition_by_label(stack: ImageStack, assignments: list[ChannelAssignment]) -> dict[str, dict]:
    """Acquisition settings of the file (D63) keyed by the channel label of this analysis."""
    chans = (stack.info.get("acquisition") or {}).get("channels") or []
    return {a.label: dict(ch) for a, ch in zip(assignments, chans) if a.role != "Ignore"}


# Values that have no meaning without a nucleus or without actin
NUCLEUS_KEYS = (["nucleus_volume_um3", "cytoplasm_volume_um3", "cytoplasm_without_actin_volume_um3",
                 "overlap_nucleus_actin_um3", "signal_overlap_nucleus_actin_pct",
                 "target_mask_nucleus_pct", "target_mask_cytoplasm_pct", "target_mask_cytoplasm_without_actin_pct"]
                + [f"{q}_{c}{s}" for c in ("nucleus", "cytoplasm", "cytoplasm_without_actin",
                                           "envelope_zone", "nucleus_interior")
                   for q, s in (("signal", "_pct"), ("volume", "_pct"), ("enrichment", ""), ("mean", ""))]
                + [f"{k}_{m}" for k in ("signal_nucleus_pct", "enrichment_nucleus", "signal_cytoplasm_pct",
                                         "enrichment_cytoplasm") for m in ("min", "max")])
ACTIN_KEYS = (["actin_volume_um3", "actin_rich_volume_um3", "cytoplasm_without_actin_volume_um3",
               "overlap_nucleus_actin_um3", "signal_overlap_nucleus_actin_pct",
               "target_mask_actin_pct", "target_mask_cytoplasm_without_actin_pct"]
              + [f"{q}_{c}{s}" for c in ("actin", "cytoplasm_without_actin", "actin_rich")
                 for q, s in (("signal", "_pct"), ("volume", "_pct"), ("enrichment", ""), ("mean", ""))]
              + [f"{k}_{m}" for k in ("signal_actin_pct", "enrichment_actin") for m in ("min", "max")])


def _blank(cell_values: dict, target_values: dict[str, dict], keys) -> None:
    for d in (cell_values, *target_values.values()):
        for k in keys:
            if k in d:
                d[k] = float("nan")


def _median(values: np.ndarray) -> float:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")
