"""Colocalization inside the cell mask (D35, METRICS §9.2), only to compare with the literature.

Pearson correlation (PCC): do the two intensities vary together? Independent of offset and gain
(Adler and Parmryd 2010). Manders coefficients with the automatic threshold of Costes et al. (2004):
which share of one channel's signal lies where the other channel is above its threshold
(co-occurrence, Aaron et al. 2018). Definitions as tM1 and tM2 in Dunn et al. (2011):

    M1 = sum of A where B > T_B / sum of A,   M2 = sum of B where A > T_A / sum of B

All sums run over the voxels of the cell only (Dunn et al. 2011). No randomisation test and no
p value: inferential statistics belong in Prism (D10).

Costes threshold: orthogonal regression B = a·A + b over the cell voxels, then, going down from the
brightest value, the first threshold pair (T_A, a·T_A + b) for which the voxels below both thresholds
are no longer positively correlated.
"""

from __future__ import annotations

import numpy as np

MAX_SAMPLE = 500_000
MIN_VOXELS = 50


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 3:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    den = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / den) if den > 0 else float("nan")


def _orthogonal_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ma, mb = a.mean(), b.mean()
    sxx = ((a - ma) ** 2).mean()
    syy = ((b - mb) ** 2).mean()
    sxy = ((a - ma) * (b - mb)).mean()
    if sxy == 0:
        return float("nan"), float("nan")
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy * sxy)) / (2 * sxy)
    return float(slope), float(mb - slope * ma)


def costes_thresholds(a: np.ndarray, b: np.ndarray, iterations: int = 40) -> tuple[float, float]:
    """Thresholds (T_A, T_B) after Costes et al. (2004). NaN if the channels are not positively
    correlated, then a threshold cannot be found this way."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size > MAX_SAMPLE:
        idx = np.random.default_rng(0).choice(a.size, MAX_SAMPLE, replace=False)
        a, b = a[idx], b[idx]
    if a.size < MIN_VOXELS or not (pearson(a, b) > 0):
        return float("nan"), float("nan")
    slope, icpt = _orthogonal_fit(a, b)
    if not np.isfinite(slope) or slope <= 0:
        return float("nan"), float("nan")

    def pcc_below(t: float) -> float:
        sel = (a < t) & (b < slope * t + icpt)
        if sel.sum() < MIN_VOXELS:
            return -1.0                      # too few voxels left: treat as uncorrelated
        r = pearson(a[sel], b[sel])
        return r if np.isfinite(r) else -1.0

    # Costes et al. lower the threshold step by step from the top and stop at the first threshold
    # where the voxels below are no longer correlated. Noise lets the PCC of the background swing
    # around 0, so the first crossing from the top is searched (64 steps), then refined.
    top, bottom = float(a.max()), float(a.min())
    steps = np.linspace(top, bottom, 65)
    hi, lo = top, None
    for t in steps[1:]:
        if pcc_below(t) <= 0:
            lo = float(t)
            break
        hi = float(t)
    if lo is None:
        return float("nan"), float("nan")
    for _ in range(iterations // 2):
        mid = 0.5 * (lo + hi)
        if pcc_below(mid) > 0:
            hi = mid
        else:
            lo = mid
    t_a = lo
    return float(t_a), float(slope * t_a + icpt)


def manders(a: np.ndarray, b: np.ndarray, t_a: float, t_b: float) -> tuple[float, float]:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    sa, sb = a.sum(), b.sum()
    if not (np.isfinite(t_a) and np.isfinite(t_b)):
        return float("nan"), float("nan")
    m1 = float(a[b > t_b].sum() / sa) if sa > 0 else float("nan")
    m2 = float(b[a > t_a].sum() / sb) if sb > 0 else float("nan")
    return m1, m2


def colocalization(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> dict:
    """PCC, Manders M1 (share of A where B is above threshold), M2, both thresholds and the share
    of voxels that are exactly 0 (Airyscan processing clips at 0, many zeros pull the PCC)."""
    cell = np.asarray(cell, bool)
    va = np.asarray(a)[cell].astype(np.float64)
    vb = np.asarray(b)[cell].astype(np.float64)
    t_a, t_b = costes_thresholds(va, vb)
    m1, m2 = manders(va, vb, t_a, t_b)
    n = va.size
    return {"pcc": pearson(va, vb), "m1": m1, "m2": m2, "threshold_a": t_a, "threshold_b": t_b,
            "zero_a_pct": 100.0 * float((va == 0).sum()) / n if n else float("nan"),
            "zero_b_pct": 100.0 * float((vb == 0).sum()) / n if n else float("nan")}
