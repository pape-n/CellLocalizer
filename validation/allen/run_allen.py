"""Positive controls with the Allen Cell hiPSC single-cell images (D72, criteria D75, protocol D76).

Usage (after download_allen.py):

    python run_allen.py "<folder with selection.csv>" [--protocol 2] [--jobs 2] [--limit N]

Writes into that folder (N = protocol):
  allen_cells_runN.csv     one row per cell and variant: enrichments, shares, Dice with the Allen masks
  allen_summary_runN.csv   per structure and variant: median, quartiles, share of cells meeting the criterion
  allen_report_runN.txt    the criteria of D75 with PASS or FAIL per structure

Protocol 1 (first run, D75 as proposed): program defaults, background 0. Variant A: Allen cell mask, own
nucleus. Variant B: own cell edge on the image set to 0 outside the Allen cell. This run is kept for the
record: the Allen images carry a camera offset of about 400 that background 0 leaves in (METRICS §4.11).

Protocol 2 (D76, default): background per channel = camera dark level of the crop (0.1st percentile of the
DNA channel). Variant A: Allen cell and nucleus masks (zones and measurement only). Variant B: Allen cell
mask, own nucleus. Variant C: own cell edge and nucleus, image set to the dark level outside the Allen cell.

Roles: DNA dye = Nucleus, membrane dye = Membrane, tagged structure = Target. All other settings are the
program defaults (the analysis is not tuned to these images).
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from localization_tool.core.analysis import analyze_cell  # noqa: E402
from localization_tool.core.channels import ChannelAssignment  # noqa: E402
from localization_tool.core.image import ChannelInfo, ImageStack  # noqa: E402
from localization_tool.core.params import AnalysisParams  # noqa: E402

GROUPS = {
    "nuclear": ("HIST1H2BJ", "FBL", "NPM1", "SON"),
    "envelope": ("LMNB1", "NUP153"),
    "cytoplasmic": ("TOMM20", "ST6GAL1", "LAMP1", "TUBA1B"),
    "er": ("SEC61B",),
    "surface": ("CTNNB1", "TJP1", "ACTB"),
    "diffuse": ("AAVS1",),
}
PASS_SHARE = 0.9


def criterion(group: str, v: dict) -> bool:
    """D75: the expected localization of one cell (enrichments E of the target)."""
    e = lambda k: v.get(f"enrichment_{k}", float("nan"))  # noqa: E731
    if group == "nuclear":
        return e("nucleus") > 1.5
    if group == "envelope":
        return e("envelope_zone") > 1.5 and e("envelope_zone") > e("nucleus_interior")
    if group == "cytoplasmic":
        return e("nucleus") < 0.5
    if group == "er":
        return e("nucleus") < 0.5 and e("envelope_zone") > e("nucleus_interior")
    if group == "surface":
        return e("cortex_shell") > 1.2
    if group == "diffuse":
        zones = [e(k) for k in ("envelope_zone", "nucleus_interior", "cortex_shell", "basal_layer")]
        return 0.5 < e("nucleus") < 2 and all(not z > 1.5 for z in zones)
    raise ValueError(group)


def group_of(structure: str) -> str:
    return next(g for g, s in GROUPS.items() if structure in s)


def read_ome(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None, list[str]]:
    """(C, Z, Y, X) data, voxel (dz, dy, dx) in µm if the header has it, and channel names of an OME-TIFF."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        data = tif.series[0].asarray()
        axes = tif.series[0].axes
        xml = tif.ome_metadata or ""
    order = [axes.index(a) for a in "CZYX" if a in axes]
    data = np.transpose(np.squeeze(data) if data.ndim > 4 else data, order) if len(order) == data.ndim else data
    if data.ndim == 3:
        data = data[None]
    voxel, names = None, []
    if xml:
        root = ET.fromstring(xml)
        ns = {"o": root.tag.split("}")[0].strip("{")}
        px = root.find(".//o:Pixels", ns)
        if px is not None:
            if all(px.get(k) for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX")):
                voxel = tuple(float(px.get(k)) for k in ("PhysicalSizeZ", "PhysicalSizeY", "PhysicalSizeX"))
            names = [c.get("Name", "") for c in px.findall("o:Channel", ns)]
    return data, voxel, names


def voxel_of(row: dict, found: tuple[float, float, float] | None) -> tuple[float, float, float]:
    """Voxel size in µm (z, y, x). The Allen crops carry no physical size in their OME header, the size
    is in the column scale_micron of the dataset table (x, y, z; the crops are resampled to isotropic)."""
    if found:
        return found
    try:
        x, y, z = (float(v) for v in ast.literal_eval(row.get("scale_micron") or ""))
        return z, y, x
    except (ValueError, SyntaxError, TypeError):
        raise ValueError(f"no voxel size for cell {row.get('CellId')}") from None


def channel_names(row: dict, kind: str, found: list[str], n: int) -> list[str]:
    try:
        d = ast.literal_eval(row.get("name_dict") or "{}")
        if kind in d and len(d[kind]) == n:
            return [str(x).lower() for x in d[kind]]
    except (ValueError, SyntaxError):
        pass
    return [x.lower() for x in found] if len(found) == n else [f"ch{i}" for i in range(n)]


def pick(names: list[str], *keys: str) -> int:
    for i, nm in enumerate(names):
        if any(k in nm for k in keys):
            return i
    raise KeyError(f"no channel named {keys} in {names}")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def analyse_cell(folder: Path, row: dict) -> list[dict]:
    """Both variants for one cell of selection.csv (empty list if its files are missing)."""
    params = AnalysisParams(sensitivity=False, colocalization=False, polarity=False)
    st, cid = row["structure_name"], row["CellId"]
    raw_p = folder / st / f"{cid}_raw.ome.tiff"
    seg_p = folder / st / f"{cid}_seg.ome.tiff"
    if not (raw_p.exists() and seg_p.exists()):
        return []
    raw, voxel, found = read_ome(raw_p)
    seg, _, found_s = read_ome(seg_p)
    voxel = voxel_of(row, voxel)
    rn = channel_names(row, "crop_raw", found, raw.shape[0])
    sn = channel_names(row, "crop_seg", found_s, seg.shape[0])
    i_dna, i_mem, i_str = pick(rn, "dna"), pick(rn, "mem"), pick(rn, "struct")
    cell_seg = seg[pick(sn, "mem", "cell")] > 0
    nuc_seg = seg[pick(sn, "dna", "nuc")] > 0
    data = raw[[i_str, i_mem, i_dna]].astype(np.uint16)
    chans = [ChannelInfo(0, "structure", "mEGFP", None, None, "#00FF00"),
             ChannelInfo(1, "membrane", "CellMask", None, None, "#FF00FF"),
             ChannelInfo(2, "dna", "NucBlue", None, None, "#0000FF")]
    roles = [ChannelAssignment("Target", st, "#2BD46B"), ChannelAssignment("Membrane", "Membrane", "#FF2BD6"),
             ChannelAssignment("Nucleus", "DNA", "#3D7DFF")]
    info = {"coverslip_first": True, "z_orientation_source": "Allen crops: first plane at the coverslip"}
    out = []
    for variant in ("A", "B"):
        d = data.copy()
        masks = None
        if variant == "A":
            masks = {"cell": cell_seg}
        else:
            d[:, ~cell_seg] = 0
        stack = ImageStack(d, voxel, chans, info=dict(info))
        try:
            r = analyze_cell(stack, roles, params, masks=masks)
        except Exception as exc:              # report, do not stop the run
            out.append({"structure": st, "cell": cid, "variant": variant, "error": str(exc)})
            continue
        v = r.target_values[st]
        z0, y0, x0 = r.origin

        def full(m):
            f = np.zeros(cell_seg.shape, bool)
            f[z0:z0 + m.shape[0], y0:y0 + m.shape[1], x0:x0 + m.shape[2]] = m
            return f
        group = group_of(st)
        res = {"structure": st, "group": group, "cell": cid, "variant": variant,
               "passes": criterion(group, v),
               "dice_nucleus": dice(full(r.masks["nucleus"]), nuc_seg),
               "dice_cell": dice(full(r.masks["cell"]), cell_seg),
               "cell_volume_um3": r.cell_values.get("cell_volume_um3"),
               "notes": "; ".join(r.notes)}
        for key in ("nucleus", "cytoplasm", "envelope_zone", "nucleus_interior", "cortex_shell", "basal_layer"):
            res[f"E_{key}"] = v.get(f"enrichment_{key}")
            res[f"signal_{key}_pct"] = v.get(f"signal_{key}_pct")
        out.append(res)
    return out


def dark_level(raw_dna: np.ndarray) -> float:
    """Protocol 2 (D76): camera dark level of one crop, the 0.1st percentile of the DNA channel. The DNA dye
    stains only nuclei, so its darkest voxels carry no dye. Subtracted from all three channels (same camera
    offset, about 400 in these images). Diffuse background above the dark level stays in, which pulls every
    enrichment towards 1, i.e. against passing."""
    return float(np.percentile(raw_dna, 0.1))


def analyse_cell2(folder: Path, row: dict) -> list[dict]:
    """Protocol 2 (D76): background = dark level, three variants.
    A: Allen cell and Allen nucleus as masks (only zones and measurement of the program are tested).
    B: Allen cell as mask, own nucleus. C: own cell edge and own nucleus; outside the Allen cell the image
    is set to the dark level (neighbour cells of the colony removed, the surroundings look like background)."""
    st, cid = row["structure_name"], row["CellId"]
    raw_p = folder / st / f"{cid}_raw.ome.tiff"
    seg_p = folder / st / f"{cid}_seg.ome.tiff"
    if not (raw_p.exists() and seg_p.exists()):
        return []
    raw, voxel, found = read_ome(raw_p)
    seg, _, found_s = read_ome(seg_p)
    voxel = voxel_of(row, voxel)
    rn = channel_names(row, "crop_raw", found, raw.shape[0])
    sn = channel_names(row, "crop_seg", found_s, seg.shape[0])
    i_dna, i_mem, i_str = pick(rn, "dna"), pick(rn, "mem"), pick(rn, "struct")
    cell_seg = seg[pick(sn, "mem", "cell")] > 0
    nuc_seg = seg[pick(sn, "dna", "nuc")] > 0
    bg = dark_level(raw[i_dna])
    data = raw[[i_str, i_mem, i_dna]].astype(np.uint16)
    chans = [ChannelInfo(0, "structure", "mEGFP", None, None, "#00FF00"),
             ChannelInfo(1, "membrane", "CellMask", None, None, "#FF00FF"),
             ChannelInfo(2, "dna", "NucBlue", None, None, "#0000FF")]
    roles = [ChannelAssignment("Target", st, "#2BD46B"), ChannelAssignment("Membrane", "Membrane", "#FF2BD6"),
             ChannelAssignment("Nucleus", "DNA", "#3D7DFF")]
    params = AnalysisParams(sensitivity=False, colocalization=False, polarity=False,
                            background={st: bg, "Membrane": bg, "DNA": bg})
    info = {"coverslip_first": True, "z_orientation_source": "Allen crops: first plane at the coverslip"}
    out = []
    for variant in ("A", "B", "C"):
        d = data.copy()
        masks = None
        if variant == "A":
            masks = {"cell": cell_seg, "nucleus": nuc_seg & cell_seg}
        elif variant == "B":
            masks = {"cell": cell_seg}
        else:
            d[:, ~cell_seg] = np.uint16(round(bg))
        stack = ImageStack(d, voxel, chans, info=dict(info))
        try:
            r = analyze_cell(stack, roles, params, masks=masks)
        except Exception as exc:              # report, do not stop the run
            out.append({"structure": st, "cell": cid, "variant": variant, "error": str(exc)})
            continue
        v = r.target_values[st]
        z0, y0, x0 = r.origin

        def full(m):
            f = np.zeros(cell_seg.shape, bool)
            f[z0:z0 + m.shape[0], y0:y0 + m.shape[1], x0:x0 + m.shape[2]] = m
            return f
        group = group_of(st)
        res = {"structure": st, "group": group, "cell": cid, "variant": variant,
               "passes": criterion(group, v), "background": bg,
               "dice_nucleus": dice(full(r.masks["nucleus"]), nuc_seg),
               "dice_cell": dice(full(r.masks["cell"]), cell_seg),
               "cell_volume_um3": r.cell_values.get("cell_volume_um3"),
               "nucleus_volume_um3": r.cell_values.get("nucleus_volume_um3"),
               "allen_cell_volume_um3": float(cell_seg.sum() * np.prod(voxel)),
               "allen_nucleus_volume_um3": float(nuc_seg.sum() * np.prod(voxel)),
               "notes": "; ".join(r.notes)}
        for key in ("nucleus", "cytoplasm", "envelope_zone", "nucleus_interior", "cortex_shell", "basal_layer"):
            res[f"E_{key}"] = v.get(f"enrichment_{key}")
            res[f"signal_{key}_pct"] = v.get(f"signal_{key}_pct")
        out.append(res)
    return out


def analyse(folder: Path, limit: int | None, jobs: int = 1, protocol: int = 2) -> list[dict]:
    rows = list(csv.DictReader(open(folder / "selection.csv", encoding="utf-8")))
    rows = rows[:limit] if limit else rows
    fn = analyse_cell2 if protocol == 2 else analyse_cell
    out = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(jobs) as ex:
            for k, res in enumerate(ex.map(fn, [folder] * len(rows), rows), 1):
                out.extend(res)
                print(f"{k}/{len(rows)}", flush=True)
        return out
    for k, row in enumerate(rows, 1):
        out.extend(fn(folder, row))
        print(f"{k}/{len(rows)} {row['structure_name']} {row['CellId']}", flush=True)
    return out


VARIANT_NAMES = {1: {"A": "Allen cell mask, own nucleus, no background subtraction",
                     "B": "own cell edge on the image set to 0 outside the Allen cell, no background subtraction"},
                 2: {"A": "Allen cell and nucleus masks", "B": "Allen cell mask, own nucleus",
                     "C": "own cell edge and nucleus"}}


def summarise(cells: list[dict], folder: Path, protocol: int = 2, label: str = "") -> None:
    ok = [c for c in cells if "error" not in c]
    keys = [k for k in ok[0] if k.startswith(("E_", "dice_"))] if ok else []
    summary, report = [], [f"Positive controls with the Allen Cell hiPSC single-cell images (criteria D75, "
                           f"protocol {protocol}{', D76' if protocol == 2 else ''})", ""]
    for variant, name in VARIANT_NAMES[protocol].items():
        report.append(f"Variant {variant}: {name}")
        for st in sorted({c["structure"] for c in ok}):
            sel = [c for c in ok if c["structure"] == st and c["variant"] == variant]
            if not sel:
                continue
            share = float(np.mean([c["passes"] for c in sel]))
            row = {"variant": variant, "structure": st, "group": sel[0]["group"], "cells": len(sel),
                   "share_passing": round(share, 3)}
            for k in keys:
                vals = np.array([c[k] for c in sel if c[k] is not None], float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    row[f"{k}_median"], row[f"{k}_q1"], row[f"{k}_q3"] = round(med, 3), round(q1, 3), round(q3, 3)
            summary.append(row)
            verdict = "PASS" if share >= PASS_SHARE else "FAIL"
            report.append(f"  {st:10s} {sel[0]['group']:12s} n={len(sel):3d}  {100 * share:5.1f} % meet the "
                          f"criterion  {verdict}")
        report.append("")
    errors = [c for c in cells if "error" in c]
    if errors:
        report.append(f"{len(errors)} analyses failed, see allen_cells.csv (column error).")
    if summary:
        with open(folder / f"allen_summary_run{protocol}{label}.csv", "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in summary for k in r}, key=lambda k: (k not in ("variant", "structure", "group",
                                                                                      "cells", "share_passing"), k))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary)
    (folder / f"allen_report_run{protocol}{label}.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))



# D79 (confirmatory run on new cells): groups with AAVS1 as plasma membrane marker, criteria as separation
# of a structure from a comparison group (AUC = probability that a random cell of the structure has the
# larger value than a random cell of the comparison group, descriptive, no test). Fixed before the run.
GROUPS3 = {
    "nuclear": ("HIST1H2BJ", "FBL", "NPM1", "SON"),
    "envelope": ("LMNB1", "NUP153"),
    "cytoplasmic": ("TOMM20", "ST6GAL1", "LAMP1", "TUBA1B"),
    "er": ("SEC61B",),
    "surface": ("CTNNB1", "TJP1", "ACTB", "AAVS1"),
}
AUC_PASS = 0.9
# group: (value, comparison group, direction): +1 the structure must be higher, -1 lower
CRITERIA3 = {
    "nuclear": ("E_nucleus", "cytoplasmic", +1),
    "envelope": ("envelope_minus_interior", "nuclear", +1),
    "cytoplasmic": ("E_nucleus", "nuclear", -1),
    "er": ("envelope_minus_interior", "cytoplasmic", +1),
    "surface": ("E_cortex_shell", "cytoplasmic", +1),
}


def auc(a, b) -> float:
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    b = np.asarray([x for x in b if x is not None and np.isfinite(x)], float)
    if not a.size or not b.size:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def summarise3(cells: list[dict], folder: Path, label: str = "") -> None:
    """D79: separation criteria per structure and variant, and agreement of variant C with variant A."""
    ok = [dict(c) for c in cells if "error" not in c]
    for c in ok:
        e, i = c.get("E_envelope_zone"), c.get("E_nucleus_interior")
        c["envelope_minus_interior"] = (e - i) if e is not None and i is not None else None
    group = {st: g for g, sts in GROUPS3.items() for st in sts}
    report = ["Positive controls, confirmatory run (D79): separation from a comparison group, pass at AUC "
              f">= {AUC_PASS} (or <= {1 - AUC_PASS:.1f} where the structure must be lower)", ""]
    rows = []
    for variant, name in VARIANT_NAMES[2].items():
        report.append(f"Variant {variant}: {name}")
        sel = [c for c in ok if c["variant"] == variant]
        for st in sorted({c["structure"] for c in sel}, key=lambda x: (list(GROUPS3).index(group[x]), x)):
            g = group[st]
            key, comp, sign = CRITERIA3[g]
            a = [c[key] for c in sel if c["structure"] == st]
            b = [c[key] for c in sel if group[c["structure"]] == comp]
            v = auc(a, b)
            passed = bool(v >= AUC_PASS) if sign > 0 else bool(v <= 1 - AUC_PASS)
            rows.append({"variant": variant, "structure": st, "group": g, "value": key, "against": comp,
                         "auc": round(v, 3), "passes": passed, "cells": len(a)})
            report.append(f"  {st:10s} {g:12s} {key:24s} vs {comp:12s} AUC {v:5.3f}  {'PASS' if passed else 'FAIL'}")
        report.append("")
    report.append("Agreement of variant C (own masks) with variant A (Allen masks), median over the cells:")
    a = {c["cell"]: c for c in ok if c["variant"] == "A"}
    for st in sorted({c["structure"] for c in ok}, key=lambda x: (list(GROUPS3).index(group[x]), x)):
        cs = [c for c in ok if c["variant"] == "C" and c["structure"] == st and c["cell"] in a]
        parts = []
        for key in ("E_nucleus", "E_envelope_zone", "E_cortex_shell"):
            d = [c[key] - a[c["cell"]][key] for c in cs if c[key] is not None and a[c["cell"]][key] is not None]
            parts.append(f"{key} {np.median(d):+.2f}" if d else f"{key} n/a")
        dn = [c["dice_nucleus"] for c in cs]
        dc = [c["dice_cell"] for c in cs]
        report.append(f"  {st:10s} " + ", ".join(parts) + f", Dice nucleus {np.median(dn):.2f}, cell {np.median(dc):.2f}")
    with open(folder / f"allen_summary_run3{label}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (folder / f"allen_report_run3{label}.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cells (a quick look)")
    ap.add_argument("--jobs", type=int, default=1, help="cells analysed in parallel")
    ap.add_argument("--protocol", type=int, default=2, choices=(1, 2, 3),
                    help="1: first run as proposed in D75 (no background subtraction), 2: D76 (default), "
                         "3: D76 analysis with the separation criteria of D79 (confirmatory run)")
    ap.add_argument("--label", default="", help="appended to the output names, e.g. _v1.4")
    a = ap.parse_args()
    cells = analyse(a.folder, a.limit, a.jobs, min(a.protocol, 2))
    if cells:
        fields = sorted({k for c in cells for k in c}, key=lambda k: (k not in ("structure", "group", "cell",
                                                                              "variant", "passes"), k))
        with open(a.folder / f"allen_cells_run{a.protocol}{a.label}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(cells)
        if a.protocol == 3:
            summarise3(cells, a.folder, a.label)
        else:
            summarise(cells, a.folder, a.protocol, a.label)


if __name__ == "__main__":
    main()
