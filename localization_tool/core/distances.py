"""Distances in µm on the anisotropic voxel grid (METRICS.md §5).

Exact Euclidean distance transform with the true voxel size (scipy). A voxel centre is taken
to lie half the lateral voxel size away from the boundary it touches, so that a voxel on either
side of a boundary gets a distance of about ±half a voxel, not 0 and one full voxel.
The remaining error is below half a voxel in every direction.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

Voxel = tuple[float, float, float]


def _half_step(voxel_um: Voxel) -> float:
    return min(voxel_um) / 2.0


def signed_distance(mask: np.ndarray, voxel_um: Voxel) -> np.ndarray:
    """Signed distance to the mask boundary: negative inside, positive outside. NaN if the mask is empty or full."""
    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    h = _half_step(voxel_um)
    outside = ndi.distance_transform_edt(~mask, sampling=voxel_um)
    inside = ndi.distance_transform_edt(mask, sampling=voxel_um)
    d = np.where(mask, -(inside - h), outside - h)
    return d.astype(np.float32)


def depth_below_surface(cell: np.ndarray, voxel_um: Voxel) -> np.ndarray:
    """Distance from the cell surface measured inwards, >= 0 inside the cell, NaN outside.

    The stack border counts as outside on all sides. The basal face on the coverslip is a real
    cell surface with a ventral cortex. A stack cut off at the top also makes the cut face look
    like a surface; that case is reported by the truncation flag (METRICS §7).
    """
    cell = np.asarray(cell, bool)
    if not cell.any():
        return np.full(cell.shape, np.nan, dtype=np.float32)
    padded = np.pad(cell, 1, mode="constant", constant_values=False)
    inside = ndi.distance_transform_edt(padded, sampling=voxel_um)[1:-1, 1:-1, 1:-1]
    d = inside - _half_step(voxel_um)
    return np.where(cell, np.maximum(d, 0.0), np.nan).astype(np.float32)


def base_plane(cell: np.ndarray, coverslip_first: bool) -> int | None:
    """Index of the plane where the cell touches the substrate: the lowest plane that contains the cell."""
    planes = np.flatnonzero(np.asarray(cell, bool).reshape(cell.shape[0], -1).any(axis=1))
    if planes.size == 0:
        return None
    return int(planes[0] if coverslip_first else planes[-1])


def height_above_substrate(cell: np.ndarray, dz_um: float, coverslip_first: bool) -> np.ndarray:
    """Height of each plane centre above the centre of the base plane, in µm, NaN outside the cell."""
    nz = cell.shape[0]
    base = base_plane(cell, coverslip_first)
    if base is None:
        return np.full(cell.shape, np.nan, dtype=np.float32)
    idx = np.arange(nz, dtype=np.float32)
    h = (idx - base) if coverslip_first else (base - idx)
    h3 = np.broadcast_to((h * dz_um)[:, None, None], cell.shape)
    return np.where(cell, h3, np.nan).astype(np.float32)
