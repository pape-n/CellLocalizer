"""Optional correction of channel crosstalk (D27, D38).

Measured channel = own dye + k · dye of the source channel (linear model, Zimmermann 2005). With k from
a single-stained control the correction is I'_into = max(I_into − k · I_from, 0). It is applied to a
copy for the analysis, the raw data stay unchanged. Off by default.

Estimating k from a control stained only for the source dye: the slope through the origin of the
into channel against the source channel, over the cell voxels where the source is clearly above
background (Otsu threshold), because Airyscan processing clips values near 0.
"""

from __future__ import annotations

import numpy as np

from .segmentation import auto_threshold


def apply(intensity_into: np.ndarray, intensity_from: np.ndarray, k: float) -> np.ndarray:
    return np.maximum(np.asarray(intensity_into, np.float64) - float(k) * np.asarray(intensity_from, np.float64), 0.0)


def estimate_k(source: np.ndarray, into: np.ndarray, cell: np.ndarray) -> dict:
    """k = Σ(s·i) / Σ(s²) over cell voxels with the source above its Otsu threshold."""
    cell = np.asarray(cell, bool)
    s = np.asarray(source, np.float64)[cell]
    i = np.asarray(into, np.float64)[cell]
    if s.size < 10 or s.max() <= 0:
        return {"k": float("nan"), "voxels": 0, "r": float("nan")}
    t = auto_threshold(s, "otsu")
    sel = s > t
    s, i = s[sel], i[sel]
    den = float((s * s).sum())
    k = float((s * i).sum() / den) if den > 0 else float("nan")
    r = float(np.corrcoef(s, i)[0, 1]) if s.size > 2 and s.std() > 0 and i.std() > 0 else float("nan")
    return {"k": k, "voxels": int(s.size), "r": r}
