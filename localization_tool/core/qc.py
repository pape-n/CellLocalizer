"""Quality notes per cell (METRICS.md §7). They never exclude a cell (D9)."""

from __future__ import annotations

import numpy as np


def _plane_areas(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float)


def flags(cell: np.ndarray, nucleus: np.ndarray, raw_channels: dict[str, np.ndarray], dtype_max: float | None,
          coverslip_first: bool, truncation_fraction: float, saturation_fraction: float,
          has_nucleus_channel: bool) -> list[str]:
    """Short English notes, also used as column values in the export."""
    out: list[str] = []
    if not cell.any():
        return ["empty cell mask"]
    nz = cell.shape[0]
    top = nz - 1 if coverslip_first else 0
    areas = _plane_areas(cell)
    if nz > 1 and areas[top] > truncation_fraction * areas.max():
        out.append("stack cut off at the top")
    if has_nucleus_channel:
        if not nucleus.any():
            out.append("no nucleus found")
        else:
            n_areas = _plane_areas(nucleus)
            if nz > 1 and n_areas[top] > truncation_fraction * n_areas.max():
                out.append("nucleus cut off at the top")
    plane_mask = cell.any(axis=0)
    if plane_mask[0, :].any() or plane_mask[-1, :].any() or plane_mask[:, 0].any() or plane_mask[:, -1].any():
        out.append("cell touches the crop border")
    if dtype_max is not None:
        n = int(cell.sum())
        for label, raw in raw_channels.items():
            sat = int((raw[cell] >= dtype_max).sum())
            if n and sat / n > saturation_fraction:
                out.append(f"saturated voxels in {label} ({100.0 * sat / n:.2f} %)")
    return out



def bottom_ratio(cell: np.ndarray, channels: list[np.ndarray], coverslip_first: bool) -> float:
    """Brightness of the plane at the coverslip relative to the brightest plane, within the footprint of
    the cell, highest over the given channels (actin, targets). A stack that starts below the cell
    rises from dark glass over about 0.3 to 0.5 µm; a bright first plane means the stack starts inside
    the cell, so the lowest cell layer was not recorded (review of 1.2, D67)."""
    if cell.shape[0] < 3 or not channels or not cell.any():
        return float("nan")
    foot = cell.any(axis=0)
    bottom = 0 if coverslip_first else cell.shape[0] - 1
    best = 0.0
    for ch in channels:
        means = np.asarray(ch, np.float64)[:, foot].mean(axis=1)
        top = float(means.max())
        if top > 0:
            best = max(best, float(means[bottom]) / top)
    return best

TWO_CELLS_NUCLEUS_UM3 = 20.0       # a second nucleus part of this size is rarely a lobe (real data, phase 5)
DAPI_OUTSIDE_PCT = 15.0            # more DAPI outside the nucleus mask: the mask probably misses chromatin
CROSSTALK_PCC = 0.7                # target and actin this similar: check a single-stain control
DARK_ZERO_FRACTION = 0.01          # D80: fewer voxels exactly 0 than this ...
DARK_LEVEL_FRACTION = 0.05         # ... and the darkest voxels this high relative to the bright ones: offset left in


def dark_level(raw: np.ndarray) -> float:
    """Camera dark level of a channel: the 0.1st percentile of its voxels (D76, D80), from every second
    voxel in y and x (same value within a grey level, faster on large images)."""
    v = np.asarray(raw)
    v = v[..., ::2, ::2] if v.ndim >= 2 else v
    return float(np.percentile(v, 0.1))


def dark_level_note(label: str, raw: np.ndarray, background: float) -> str | None:
    """D80: a hint when a channel still carries a camera offset that is not subtracted. Images clipped at
    0 (Airyscan: 50 to 77 % of the voxels exactly 0) never get it. Describes, changes no value."""
    if background > 0:
        return None
    v = np.asarray(raw)
    v = v[..., ::2, ::2] if v.ndim >= 2 else v
    if v.size == 0 or float(np.mean(v == 0)) >= DARK_ZERO_FRACTION:
        return None
    lo, hi = float(np.percentile(v, 0.1)), float(np.percentile(v, 99.9))
    if hi <= 0 or lo <= DARK_LEVEL_FRACTION * hi:
        return None
    return (f"{label}: no voxel at 0, darkest voxels at {lo:.0f}: background not subtracted? "
            "(Settings > Channels > Estimate dark level)")


def cell_notes(calibrated: bool, nucleus: np.ndarray, voxel_um, dapi: np.ndarray | None, cell: np.ndarray,
               targets: dict[str, np.ndarray], target_values: dict[str, dict], sigma_um: float) -> list[str]:
    """Notes that need the measured intensities (phase 5 review). They describe, they never change values."""
    from scipy import ndimage as ndi

    from .segmentation import has_signal, smooth

    out: list[str] = []
    if not calibrated:
        out.append("no voxel calibration in the file: sizes are in voxels, not µm")
    if nucleus.any():
        lab, n = ndi.label(nucleus)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:] * float(np.prod(voxel_um))
            big = int((sizes >= TWO_CELLS_NUCLEUS_UM3).sum())
            if big >= 2:
                out.append(f"possibly two cells: {big} separate nucleus parts above {TWO_CELLS_NUCLEUS_UM3:g} µm³")
        if dapi is not None:
            total = float(dapi[cell].sum())
            outside = float(dapi[cell & ~nucleus].sum())
            if total > 0 and 100.0 * outside / total > DAPI_OUTSIDE_PCT:
                out.append(f"{100.0 * outside / total:.0f} % of the DAPI signal lies outside the nucleus mask: "
                           "check the nucleus edge")
    for label, inten in targets.items():
        if not has_signal(smooth(inten, voxel_um, sigma_um), inten, voxel_um, sigma_um):
            out.append(f"{label}: no staining above noise (control or empty channel?)")
        pcc = target_values.get(label, {}).get("pcc_actin")
        if pcc is not None and np.isfinite(pcc) and pcc > CROSSTALK_PCC:
            out.append(f"{label} and actin correlate strongly (PCC {pcc:.2f}): check for crosstalk")
    return out
