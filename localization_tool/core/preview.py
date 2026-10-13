"""Technical mask preview for phase 0.

Not the final segmentation (that is defined in METRICS.md and follows in phase 1).
It only produces a plausible editable mask so that display and manual editing can be tested.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def smooth(volume: np.ndarray, voxel_um: tuple[float, float, float], sigma_um: float) -> np.ndarray:
    sigma = [sigma_um / s for s in voxel_um]
    return ndi.gaussian_filter(volume.astype(np.float32), sigma=sigma)


def otsu_mask(volume: np.ndarray, voxel_um: tuple[float, float, float], sigma_um: float = 0.15,
              threshold: float | None = None) -> tuple[np.ndarray, float]:
    """Smoothed Otsu mask, holes filled per plane and in 3D, largest component kept."""
    sm = smooth(volume, voxel_um, sigma_um)
    if threshold is None:
        values = sm[sm > 0]
        if values.size < 10 or values.min() == values.max():
            return np.zeros(volume.shape, bool), 0.0
        threshold = float(threshold_otsu(values))
    mask = sm > threshold
    for i in range(mask.shape[0]):
        mask[i] = ndi.binary_fill_holes(mask[i])
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, labels, index=range(1, n + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)
    return mask, float(threshold)
