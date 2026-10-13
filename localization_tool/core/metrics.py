"""Compartments and measurements per target (METRICS.md §4 to §6).

All sums run over background-subtracted intensities I' = max(I - b, 0) of the raw voxel values.
Nothing here smooths or resamples the target channel: smoothing is used only to build masks.
"""

from __future__ import annotations

import numpy as np

NUCLEUS, ACTIN, CYTOPLASM = 1, 2, 3       # disjoint label codes; 3 = cytoplasm without actin
# Measured regions (D39): nucleus and cytoplasm split the cell (100 %). Actin is the actin-rich part
# of the cytoplasm, the rest of the cytoplasm is "cytoplasm without actin".
COMPARTMENTS = {"nucleus": (NUCLEUS,), "cytoplasm": (ACTIN, CYTOPLASM), "actin": (ACTIN,),
                "cytoplasm_without_actin": (CYTOPLASM,)}


def region(labels: np.ndarray, name: str) -> np.ndarray:
    return np.isin(labels, COMPARTMENTS[name])


def compartment_labels(cell: np.ndarray, nucleus: np.ndarray, actin: np.ndarray) -> np.ndarray:
    """Label codes in the fixed order nucleus before actin before the rest of the cytoplasm (D5, D28).
    Cytoplasm = codes 2 and 3 (D39)."""
    labels = np.zeros(cell.shape, np.uint8)
    labels[cell] = CYTOPLASM
    labels[cell & actin] = ACTIN
    labels[cell & nucleus] = NUCLEUS
    return labels


SENSITIVITY_KEYS = ("signal_nucleus_pct", "enrichment_nucleus", "signal_cytoplasm_pct", "enrichment_cytoplasm",
                    "signal_actin_pct", "enrichment_actin")


def compartment_shares(intensity: np.ndarray, labels: np.ndarray) -> dict:
    """Signal share and enrichment of nucleus, cytoplasm and actin only (for the sensitivity span, D56)."""
    cell = labels > 0
    total = float(intensity[cell].sum())
    n_cell = int(cell.sum())
    out = {}
    for name in ("nucleus", "cytoplasm", "actin"):
        k = region(labels, name)
        f = _pct(float(intensity[k].sum()), total)
        out[f"signal_{name}_pct"] = f
        out[f"enrichment_{name}"] = _ratio(f, _pct(int(k.sum()), n_cell))
    return out


def background_subtract(volume: np.ndarray, background: float) -> np.ndarray:
    return np.maximum(np.asarray(volume, dtype=np.float64) - float(background), 0.0)


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else float("nan")


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 and np.isfinite(a) else float("nan")


def measure_target(intensity: np.ndarray, labels: np.ndarray, overlap: np.ndarray,
                   d_nucleus: np.ndarray, depth: np.ndarray, height: np.ndarray,
                   voxel_volume_um3: float, envelope_um: float, cortex_um: float, basal_um: float,
                   actin_rich: np.ndarray | None = None) -> dict:
    """Signal shares, volume shares, enrichment and mean intensity per compartment and zone.

    intensity is already background-subtracted. Shares in percent of the total signal in the cell.
    Enrichment E = signal share / volume share (1 = as dense as the cell average).
    """
    cell = labels > 0
    i_cell = intensity[cell]
    total = float(i_cell.sum())
    n_cell = int(cell.sum())
    out: dict = {"total_intensity": total}
    for name in COMPARTMENTS:
        k = region(labels, name)
        s = float(intensity[k].sum())
        n = int(k.sum())
        f = _pct(s, total)
        phi = _pct(n, n_cell)
        out[f"signal_{name}_pct"] = f
        out[f"volume_{name}_pct"] = phi
        out[f"enrichment_{name}"] = _ratio(f, phi)
        out[f"mean_{name}"] = s / n if n else float("nan")
    out["signal_overlap_nucleus_actin_pct"] = _pct(float(intensity[cell & overlap].sum()), total)

    zones = {}
    if np.isfinite(d_nucleus[cell]).any():
        dn = d_nucleus
        zones["envelope_zone"] = cell & (np.abs(dn) <= envelope_um)
        zones["nucleus_interior"] = cell & (dn < -envelope_um)
    else:
        zones["envelope_zone"] = None
        zones["nucleus_interior"] = None
    zones["actin_rich"] = (cell & actin_rich) if actin_rich is not None else None
    zones["cortex_shell"] = cell & (depth <= cortex_um)
    zones["basal_layer"] = cell & (height <= basal_um)
    for name, zone in zones.items():
        if zone is None:
            out[f"signal_{name}_pct"] = float("nan")
            out[f"volume_{name}_pct"] = float("nan")
            out[f"enrichment_{name}"] = float("nan")
            continue
        f = _pct(float(intensity[zone].sum()), total)
        phi = _pct(int(zone.sum()), n_cell)
        out[f"signal_{name}_pct"] = f
        out[f"volume_{name}_pct"] = phi
        out[f"enrichment_{name}"] = _ratio(f, phi)
    out["mean_cell"] = total / n_cell if n_cell else float("nan")
    out["cell_voxels"] = n_cell
    out["voxel_volume_um3"] = voxel_volume_um3
    return out


def target_mask_measures(intensity: np.ndarray, labels: np.ndarray, tmask: np.ndarray,
                         voxel_um: tuple[float, float, float]) -> dict:
    """Measures of the target mask (D30): how much of the cell it covers, how much of the signal it
    holds, how its signal splits over the compartments, and how many separate objects it has."""
    from scipy import ndimage as ndi

    cell = labels > 0
    total = float(intensity[cell].sum())
    t = tmask & cell
    s_t = float(intensity[t].sum())
    out = {
        "target_mask_volume_pct": _pct(int(t.sum()), int(cell.sum())),
        "target_mask_signal_pct": _pct(s_t, total),
    }
    for name in COMPARTMENTS:
        out[f"target_mask_{name}_pct"] = _pct(float(intensity[t & region(labels, name)].sum()), s_t)
    lab, n = ndi.label(t)
    out["target_objects"] = int(n)
    out["target_object_volume_um3"] = float(t.sum()) * float(np.prod(voxel_um)) / n if n else float("nan")
    return out


def profile(intensity: np.ndarray, distance: np.ndarray, cell: np.ndarray, bin_um: float) -> list[dict]:
    """Intensity-weighted histogram of a distance inside the cell, with the volume histogram next to it.

    Each row: bin_start_um, bin_end_um, signal_fraction, volume_fraction, enrichment.
    Fractions sum to 1 over all bins. Bins start at a multiple of bin_um.
    """
    sel = cell & np.isfinite(distance)
    if not sel.any():
        return []
    d = distance[sel].astype(np.float64)
    w = intensity[sel].astype(np.float64)
    lo = np.floor(d.min() / bin_um) * bin_um
    hi = np.ceil(d.max() / bin_um) * bin_um
    if hi <= lo:
        hi = lo + bin_um
    edges = np.arange(lo, hi + bin_um * 0.5, bin_um)
    if edges.size < 2:
        edges = np.array([lo, lo + bin_um])
    sig, _ = np.histogram(d, bins=edges, weights=w)
    vol, _ = np.histogram(d, bins=edges)
    sig_f = sig / sig.sum() if sig.sum() > 0 else np.full(sig.shape, np.nan)
    vol_f = vol / vol.sum()
    rows = []
    for a, b, s, v in zip(edges[:-1], edges[1:], sig_f, vol_f):
        rows.append({"bin_start_um": round(float(a), 6), "bin_end_um": round(float(b), 6),
                     "signal_fraction": float(s), "volume_fraction": float(v),
                     "enrichment": float(s / v) if v > 0 and np.isfinite(s) else float("nan")})
    return rows


def cell_descriptors(cell: np.ndarray, nucleus: np.ndarray, actin: np.ndarray, labels: np.ndarray,
                     voxel_um: tuple[float, float, float]) -> dict:
    v0 = float(np.prod(voxel_um))
    n_cell = int(cell.sum())
    planes = np.flatnonzero(cell.reshape(cell.shape[0], -1).any(axis=1))
    return {
        "cell_volume_um3": n_cell * v0,
        "nucleus_volume_um3": int((labels == NUCLEUS).sum()) * v0,
        "actin_volume_um3": int((labels == ACTIN).sum()) * v0,
        "cytoplasm_volume_um3": int(np.isin(labels, (ACTIN, CYTOPLASM)).sum()) * v0,
        "cytoplasm_without_actin_volume_um3": int((labels == CYTOPLASM).sum()) * v0,
        "overlap_nucleus_actin_um3": int((cell & nucleus & actin).sum()) * v0,
        "cell_height_um": (int(planes[-1] - planes[0] + 1) * voxel_um[0]) if planes.size else 0.0,
        "cell_planes": int(planes.size),
        "stack_planes": int(cell.shape[0]),
    }
