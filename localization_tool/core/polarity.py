"""Front-rear polarity from the actin channel (D36, METRICS §9.3).

Neutrophils build one F-actin rich front (Weiner et al. 2002, Xu et al. 2003, Houk et al. 2012).
The axis runs in the plane of the coverslip from the cell centroid to the actin-weighted centroid
(as marker polarity in Polarity-JaM, Giese et al. 2025). Its length divided by the radius of the
cell footprint is the polarity index: about 0 for an even cortex, larger for a cell with a front.

The plane through the cell centroid perpendicular to the axis splits the cell into a front and a
rear half. For each target: share of the signal in the front half, share of the volume, and the
enrichment (signal share / volume share). For cells with a low index the axis is mostly noise and
the front values mean little.
"""

from __future__ import annotations

import numpy as np


def _centroid_xy(mask: np.ndarray, weights: np.ndarray | None, voxel_um) -> np.ndarray | None:
    zz, yy, xx = np.nonzero(mask)
    if zz.size == 0:
        return None
    w = np.ones(zz.size) if weights is None else np.asarray(weights, np.float64)[mask]
    if w.sum() <= 0:
        return None
    return np.array([(yy * w).sum() / w.sum() * voxel_um[1], (xx * w).sum() / w.sum() * voxel_um[2]])


def polarity(cell: np.ndarray, actin: np.ndarray, nucleus: np.ndarray, voxel_um) -> dict:
    """Axis and index. Returns the values in µm and, for drawing, the axis in voxel coordinates (y, x)."""
    cell = np.asarray(cell, bool)
    nan = float("nan")
    out = {"polarity_index": nan, "polarity_axis_deg": nan, "actin_front_pct": nan, "volume_front_pct": nan,
           "nucleus_position_on_axis": nan, "footprint_radius_um": nan, "_axis": None}
    c_cell = _centroid_xy(cell, None, voxel_um)
    if c_cell is None:
        return out
    area = float(cell.any(axis=0).sum()) * voxel_um[1] * voxel_um[2]
    radius = float(np.sqrt(area / np.pi))
    out["footprint_radius_um"] = radius
    c_act = _centroid_xy(cell, actin, voxel_um)
    if c_act is None or radius <= 0:
        return out
    v = c_act - c_cell
    length = float(np.hypot(*v))
    out["polarity_index"] = length / radius
    if length <= 0:
        return out
    u = v / length
    out["polarity_axis_deg"] = float(np.degrees(np.arctan2(-u[0], u[1])))   # 0 = right, 90 = up in the image
    front = front_half(cell, c_cell, u, voxel_um)
    w = np.asarray(actin, np.float64)
    total = float(w[cell].sum())
    out["actin_front_pct"] = 100.0 * float(w[front].sum()) / total if total > 0 else nan
    out["volume_front_pct"] = 100.0 * float(front.sum()) / float(cell.sum())
    c_nuc = _centroid_xy(np.asarray(nucleus, bool) & cell, None, voxel_um) if nucleus is not None else None
    if c_nuc is not None:
        out["nucleus_position_on_axis"] = float(np.dot(c_nuc - c_cell, u)) / radius
    out["_axis"] = {"centre_um": c_cell, "unit": u, "radius_um": radius}
    return out


def front_half(cell: np.ndarray, centre_um: np.ndarray, unit: np.ndarray, voxel_um) -> np.ndarray:
    _, ny, nx = cell.shape
    yy = (np.arange(ny) * voxel_um[1] - centre_um[0])[:, None]
    xx = (np.arange(nx) * voxel_um[2] - centre_um[1])[None, :]
    side = (yy * unit[0] + xx * unit[1]) > 0
    return cell & side[None, :, :]


def target_front(intensity: np.ndarray, cell: np.ndarray, axis: dict | None, voxel_um) -> dict:
    nan = float("nan")
    if axis is None:
        return {"signal_front_pct": nan, "enrichment_front": nan}
    cell = np.asarray(cell, bool)
    front = front_half(cell, axis["centre_um"], axis["unit"], voxel_um)
    i = np.asarray(intensity, np.float64)
    total = float(i[cell].sum())
    f = 100.0 * float(i[front].sum()) / total if total > 0 else nan
    phi = 100.0 * float(front.sum()) / float(cell.sum())
    return {"signal_front_pct": f, "enrichment_front": f / phi if phi > 0 and np.isfinite(f) else nan}
