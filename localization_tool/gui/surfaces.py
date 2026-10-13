"""Smoothed 3D surfaces of the masks, only for the 3D view (0.4.0a1 feedback).

The z step (0.13 µm) is about four times the pixel size, so masks drawn voxel by voxel show steps
between the planes. For the 3D view each mask is smoothed (Gaussian, σ about one z step), resampled
to nearly cubic voxels with linear interpolation, and its 0.5 iso-surface is taken with marching
cubes (Lorensen and Cline 1987). The measured values never use these surfaces.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

XY_STEP_UM = 0.08          # lateral sampling of the surface: fine enough for the eye, fast enough


def mask_surface(mask: np.ndarray, voxel_um) -> tuple[np.ndarray, np.ndarray] | None:
    """(vertices in µm relative to the mask origin, faces), or None for an empty mask."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return None
    dz, dy, dx = voxel_um
    # crop to the mask with one voxel margin, keeps the arrays small
    zz, yy, xx = (np.flatnonzero(mask.any(axis=a)) for a in ((1, 2), (0, 2), (0, 1)))
    z0, z1 = max(0, zz[0] - 2), min(mask.shape[0], zz[-1] + 3)
    y0, y1 = max(0, yy[0] - 3), min(mask.shape[1], yy[-1] + 4)
    x0, x1 = max(0, xx[0] - 3), min(mask.shape[2], xx[-1] + 4)
    m = mask[z0:z1, y0:y1, x0:x1].astype(np.float32)
    step = max(1, int(round(XY_STEP_UM / dy)))
    m = ndi.gaussian_filter(m, sigma=(0.8, 0.5 * step, 0.5 * step))[:, ::step, ::step]
    zf = max(1.0, dz / (dy * step))
    if zf > 1.2:
        m = ndi.zoom(m, (zf, 1, 1), order=1, prefilter=False)
    m = np.pad(m, 1)                      # closed surfaces at the crop border
    spacing = (dz / zf if zf > 1.2 else dz, dy * step, dx * step)
    try:
        verts, faces, _, _ = marching_cubes(m, level=0.5, spacing=spacing)
    except (ValueError, RuntimeError):
        return None
    verts -= np.array(spacing)            # the padding
    verts += np.array([z0 * dz, y0 * dy, x0 * dx])
    return verts.astype(np.float32), faces.astype(np.int32)


def mask_surfaces(masks: dict[str, np.ndarray], voxel_um) -> dict[str, tuple | None]:
    return {name: mask_surface(m, voxel_um) for name, m in masks.items()}
