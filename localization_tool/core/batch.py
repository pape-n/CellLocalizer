"""Analyse many cells: the command line run and "analyse all images" / "apply settings to all cells"
in the program (phase 2). Writes the results workbook and the control images.

The work is a generator (iter_cells) so that the program can show progress, store each cell as soon
as it is done and stop in between. Nothing here imports Qt.
"""

from __future__ import annotations

import csv
import hashlib
import io
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .. import __version__
from .analysis import CellResult, analyze_cell
from .crop import CropBox, crop
from .image import ImageStack
from .params import AnalysisParams
from .reader import ACQUISITION_KEYS
from .project_file import LEGACY_PARAMS, AnalysisFile, CellSpec, ImageSpec, load_masks
from .qc_images import control_image
from .reader import read_image

CELL_COLUMNS_FIRST = ["image", "experiment", "condition", "cell", "status", "target",
                      "box_z0", "box_z1", "box_y0", "box_y1", "box_x0", "box_x1"]
RESULTS_NAME = "Results.xlsx"
CONTROL_DIR = "Control images"
AUTOMATIC = "automatic"
EXCLUDED = "excluded"


def _safe(text: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return keep.strip("_")[:80] or "cell"


# ---------------------------------------------------------------------------- one cell
def full_box(stack: ImageStack) -> CropBox:
    nz, ny, nx = stack.shape_zyx
    return CropBox(0, nz, 0, ny, 0, nx)


def effective_params(params: AnalysisParams, spec: CellSpec) -> AnalysisParams:
    return params.with_overrides(spec.overrides)


def analyze_spec(stack: ImageStack, assignments, params: AnalysisParams, spec: CellSpec,
                 masks: dict | None = None) -> tuple[ImageStack, CropBox, CellResult]:
    """Analyse one stored cell of an image with the settings of the analysis and the cell's own factors."""
    box = spec.box or full_box(stack)
    c = crop(stack, box)
    r = analyze_cell(c, assignments, effective_params(params, spec), exclusions_yx=spec.exclusions, masks=masks)
    return c, box, r


# Raised when the meaning of stored numbers changes, so older results are computed again:
# 2 = version 0.5/1.0 (D39: cytoplasm includes actin; empty instead of 0 % for missing compartments)
# 3 = version 1.1 (D51 half-maximum nucleus edge, D52 coverslip gap open, D53 no mask on unstained targets)
# 4 = version 1.2 (D55 channel offsets, D56 sensitivity span, D63 acquisition settings in the rows)
RESULTS_VERSION = 4


def cell_results(image: ImageSpec, spec: CellSpec, box: CropBox, r: CellResult, params: AnalysisParams) -> dict:
    """What is stored with a cell in the analysis file: table rows and the settings used."""
    rows, prof = result_rows(image.path.name, image.experiment, image.condition, spec.name, spec.status, box, r)
    own = ", ".join(f"{k} {v.get('factor'):g}" for k, v in sorted(spec.overrides.items())
                    if isinstance(v, dict) and "factor" in v)
    for row in rows:
        row["own_factors"] = own                  # factors changed for this cell only (reproducibility)
        row["blind_code"] = image.code            # D57
        row["name_revealed"] = image.revealed
    return {"rows": rows, "profile_rows": prof, "params": effective_params(params, spec).to_dict(),
            "results_version": RESULTS_VERSION, "program": __version__,
            "mask_digest": mask_digest(box, r.masks)}


def mask_digest(box: CropBox, masks: dict) -> str:
    """Fingerprint of the box and all masks. Equal fingerprints mean the person checking the cell would
    see exactly the same outlines, so a recomputed cell keeps "accepted" (review of 1.2)."""
    h = hashlib.sha1(repr(box).encode())
    for key in sorted(masks):
        m = np.asarray(masks[key])
        h.update(key.encode() + repr(m.shape).encode())
        h.update(np.packbits(m > 0).tobytes() if m.dtype == bool else np.ascontiguousarray(m).tobytes())
    return h.hexdigest()


# settings that exist since 1.2 and do not change any mask while they have these values
_NEW_IN_4 = {"channel_offsets": [], "sensitivity": None, "bottom_fraction": None, "actin_per_plane": False}


def masks_unchanged(old: dict | None, new: dict) -> bool:
    """True if a recomputed cell has the same masks as before, so its status can stay.

    With a fingerprint from 1.2 or later the fingerprints are compared. Results of 1.1 have none: there
    the masks are the same if every setting 1.1 knew is unchanged and the settings new in 1.2 do not
    move a mask (no channel offset; the sensitivity span only adds columns).
    """
    if not old or "rows" not in old:
        return False
    if old.get("mask_digest"):
        return old["mask_digest"] == new.get("mask_digest")
    if old.get("results_version") != 3:
        return False
    op, np_ = old.get("params") or {}, dict(new.get("params") or {})
    for key, neutral in _NEW_IN_4.items():
        if key not in op:
            value = np_.pop(key, None)
            if neutral is not None and value != neutral:
                return False
    return op == np_


def control_png(c: ImageStack, assignments, r: CellResult, image_name: str, experiment: str, condition: str,
                cell: str) -> bytes:
    buf = io.BytesIO()                    # in memory: no temporary file a virus scanner could hold
    control_image(c.data, assignments, r.masks, r.voxel_um, image_name,
                  control_lines(experiment, condition, cell, r), buf, r.coverslip_first, r.overlays)
    return buf.getvalue()


def needs_update(spec: CellSpec, params: AnalysisParams) -> bool:
    """True if the cell has no stored results or was computed with other settings."""
    if not spec.results or "rows" not in spec.results:
        return True
    if spec.results.get("results_version") != RESULTS_VERSION:
        return True                          # computed by an older version with other definitions
    stored = spec.results.get("params") or {}
    current = effective_params(params, spec).to_dict()
    for key, before in LEGACY_PARAMS.items():        # added later: missing means the older behaviour
        if key not in stored and current.get(key) == before:
            current.pop(key, None)
    return stored != current


# ---------------------------------------------------------------------------- many cells
def iter_cells(af: AnalysisFile, jobs: list[tuple[int, str | None]], base_dir: str | Path | None = None,
               find_assignments: Callable[[ImageStack], list | None] | None = None,
               pending_masks: dict | None = None, png: bool = True, force: bool = False,
               shown_name: Callable[[ImageSpec], str] | None = None) -> Iterator[dict]:
    """Analyse the given cells one after the other.

    shown_name: name of an image in progress texts and reasons (the code during blind review).
    jobs: (image index, cell name); cell name None means: a new automatic cell over the whole image.
    find_assignments: channel roles for an image without stored roles (e.g. a saved panel).
    pending_masks: corrected masks not yet written to disk, keyed by (image path, cell name).
    Without force, cells whose stored results used the current settings are skipped.

    Yields dicts with "kind": "progress", "cell" (with "image", "spec", "png", "assignments"),
    or "skipped" (with "reason"). Images are read once and released before the next one.
    """
    by_image: dict[int, list[str | None]] = {}
    for i, name in jobs:
        by_image.setdefault(i, []).append(name)
    total = len(jobs)
    done = 0
    for i, names in by_image.items():
        image = af.images[i]
        nm = shown_name(image) if shown_name else image.path.name      # a code during blind review (D57)

        def hide(text: str, nm=nm, image=image) -> str:
            return text if nm == image.path.name else text.replace(str(image.path), nm).replace(image.path.name, nm)

        yield {"kind": "progress", "done": done, "total": total, "text": f"Reading {nm} ..."}
        try:
            stack = read_image(image.path)
        except Exception as exc:           # one bad file must never stop the run
            text = str(exc) or exc.__class__.__name__
            for name in names:
                done += 1
                yield {"kind": "skipped", "image": i, "cell": name, "reason": f"{nm}: {hide(text)}"}
            continue
        assignments = image.assignments or (find_assignments(stack) if find_assignments else None) or af.panel
        if not assignments or len(assignments) != stack.n_channels:
            for name in names:
                done += 1
                yield {"kind": "skipped", "image": i, "cell": name,
                       "reason": f"{nm}: no channel roles. Open the image once and assign the roles, "
                                 "or save a panel for these channels."}
            continue
        for name in names:
            done += 1
            if name is None:
                spec = CellSpec(name=_free_name(image), status=AUTOMATIC, box=full_box(stack))
            else:
                spec = next((c for c in image.cells if c.name == name), None)
                if spec is None:
                    continue
                if spec.status == EXCLUDED or (not force and not needs_update(spec, af.params)):
                    yield {"kind": "skipped", "image": i, "cell": name, "reason": "", "unchanged": True}
                    continue
                spec = CellSpec.from_dict(spec.to_dict())
            yield {"kind": "progress", "done": done - 1, "total": total,
                   "text": f"{nm} / {spec.name} ({done} of {total}) ..."}
            try:
                masks = _masks_for(image, spec, base_dir, pending_masks)
                c, box, r = analyze_spec(stack, assignments, af.params, spec, masks)
                previous = spec.results
                spec.box = box
                spec.results = cell_results(image, spec, box, r, af.params)
                if (not _labels_kept(masks) and spec.status != AUTOMATIC and name is not None
                        and not masks_unchanged(previous, spec.results)):
                    spec.status = AUTOMATIC          # other masks than the checked ones: check the cell again
                    for row in spec.results["rows"]:
                        row["status"] = AUTOMATIC
                picture = control_png(c, assignments, r, image.path.name, image.experiment, image.condition,
                                      spec.name) if png else None
            except Exception as exc:           # e.g. MemoryError (no text), a damaged mask file
                text = str(exc) or exc.__class__.__name__
                yield {"kind": "skipped", "image": i, "cell": spec.name,
                       "reason": f"{nm} / {spec.name}: {hide(text)}"}
                continue
            yield {"kind": "cell", "image": i, "path": str(image.path), "spec": spec, "png": picture,
                   "assignments": assignments, "result": r if not png else None}
            del c, r
        del stack
    yield {"kind": "progress", "done": total, "total": total, "text": "Done"}


def _free_name(image: ImageSpec) -> str:
    names = {c.name for c in image.cells}
    k = 1
    while f"cell{k}" in names:
        k += 1
    return f"cell{k}"


def _labels_kept(masks: dict | None) -> bool:
    return bool(masks) and set(masks) - {"cell"} != set()


def _masks_for(image: ImageSpec, spec: CellSpec, base_dir, pending: dict | None) -> dict | None:
    if pending:
        p = pending.get((str(image.path), spec.name))
        if p and p.get("masks") is not None:
            m = p["masks"]
            if "labels" in m:
                lab = m["labels"]
                return {"cell": lab > 0, "nucleus": lab == 1, "actin": (lab == 2) | (lab == 4)}
            return {"cell": m["cell"]}
    if spec.mask_file and base_dir is not None and (Path(base_dir) / spec.mask_file).exists():
        return load_masks(Path(base_dir) / spec.mask_file)
    return None


def run(af: AnalysisFile, out_dir: str | Path, control_images: bool = True,
        progress: Callable[[str], None] | None = None, csv_tables: bool = False,
        base_dir: str | Path | None = None) -> dict:
    """Command line: analyse every cell of the analysis file and write the analysis folder content."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda s: None)
    errors = []
    t0 = time.perf_counter()
    jobs = []
    for i, image in enumerate(af.images):
        if image.cells:
            jobs += [(i, c.name) for c in image.cells if c.status != EXCLUDED]
        else:
            image.cells = [CellSpec(name="cell1")]
            jobs.append((i, "cell1"))
    rows, prof = [], []
    for ev in iter_cells(af, jobs, base_dir=base_dir or out, png=control_images, force=True):
        if ev["kind"] == "skipped":
            if ev["reason"]:
                errors.append({"image": af.images[ev["image"]].path.name, "cell": ev["cell"], "error": ev["reason"]})
                say(f"skipped: {ev['reason']}")
        elif ev["kind"] == "cell":
            image = af.images[ev["image"]]
            spec = ev["spec"]
            rows += [{**r, "status": spec.status} for r in spec.results["rows"]]
            prof += spec.results["profile_rows"]
            if ev["png"]:
                p = out / CONTROL_DIR / control_name(image.path, spec.name, [im.path for im in af.images])
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(ev["png"])
            say(f"{image.path.name} / {spec.name} done")
    images = [{"image": im.path.name, "experiment": im.experiment, "condition": im.condition,
               "channel roles": roles_text(im.assignments or af.panel), "path": str(im.path)} for im in af.images]
    info = run_info(af.params, [vars(a) for a in af.panel] if af.panel else None, images, errors)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    excluded = excluded_rows(af)
    workbook = write_workbook(out, rows, prof, info, excluded)
    result = {"workbook": workbook, "rows": len(rows), "errors": errors}
    if csv_tables:
        result["cells_csv"], result["profiles_csv"] = write_tables(out, rows, prof)
    return result


# ---------------------------------------------------------------------------- tables
def run_info(params, panel, images, errors=None) -> dict:
    """panel: channel roles of the analysis; None when every image has its own (listed with the images)."""
    return {"program": f"Localization Tool {__version__}", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params.to_dict(), "channel roles": panel if panel else "per image, see images",
            "images": images, "errors": errors or []}


def roles_text(assignments) -> str:
    return ", ".join(f"{a.label}={a.role}" for a in assignments or [])


def write_workbook(out_dir: str | Path, rows: list[dict], prof_rows: list[dict], info: dict,
                   excluded: list[dict] | None = None) -> Path:
    from .export_excel import write_results

    return write_results(Path(out_dir) / RESULTS_NAME, rows, prof_rows, info, excluded or [])


def excluded_rows(af: AnalysisFile) -> list[dict]:
    """Cells and images left out on purpose, with the reason (sheet 'Excluded')."""
    return [{"image": im.path.name, "experiment": im.experiment, "condition": im.condition, "cell": c.name,
             "reason": c.note} for im in af.images for c in im.cells if c.status == EXCLUDED]


def result_rows(image: str, experiment: str, condition: str, cell: str, status: str, box: CropBox, r) -> tuple[list, list]:
    """Table rows of one analysed cell: one row per target, and the profile rows."""
    base = {"image": image, "experiment": experiment, "condition": condition, "cell": cell, "status": status,
            "box_z0": box.z0, "box_z1": box.z1, "box_y0": box.y0, "box_y1": box.y1,
            "box_x0": box.x0, "box_x1": box.x1}
    common = {**r.cell_values,
              **{f"threshold_{k}": v for k, v in r.thresholds.items()},
              "coverslip_first": r.coverslip_first, "z_orientation": r.info.get("z_orientation", ""),
              "voxel_x_um": r.voxel_um[2], "voxel_y_um": r.voxel_um[1], "voxel_z_um": r.voxel_um[0],
              "objective": r.info.get("objective", ""), "immersion": r.info.get("immersion", ""),
              "channel_offset_correction": "; ".join(
                  f"{o['channel']} dz {o['dz_um']:g} dy {o['dy_um']:g} dx {o['dx_um']:g} um"
                  for o in r.info.get("channel_offsets", [])) or "none",
              "notes": "; ".join(r.notes)}
    for label, ch in (r.info.get("acquisition") or {}).items():          # D63
        for key in ACQUISITION_KEYS:
            if key in ch:
                common[f"{key}_{label}"] = ch[key]
        if ch.get("airyscan_estimation"):
            common[f"airyscan_mode_{label}"] = ch["airyscan_estimation"]     # automatic or fixed strength
    rows, prof = [], []
    for target, values in r.target_values.items():
        rows.append({**base, "target": target, **values, **common})
        for pname, prow in r.profiles[target].items():
            for p in prow:
                prof.append({"image": image, "experiment": experiment, "condition": condition, "cell": cell,
                             "target": target, "profile": pname, **p})
    return rows, prof


def control_lines(experiment: str, condition: str, cell: str, r) -> list[str]:
    return [f"{experiment} {condition} {cell}".strip(),
            "  ".join(f"{t}: nucleus {_pct_text(v['signal_nucleus_pct'])}, actin {_pct_text(v['signal_actin_pct'])}, "
                      f"cytoplasm {_pct_text(v['signal_cytoplasm_pct'])}" for t, v in r.target_values.items()),
            ("notes: " + "; ".join(r.notes)) if r.notes else "notes: none"]


def _pct_text(v) -> str:
    return "no value" if v is None or v != v else f"{v:.1f} %"


def control_name(image: str | Path, cell: str, images=None) -> str:
    """File name of the control image. If another image of the analysis has the same file name in
    another folder (e.g. ZEN's "New-01.czi" on two days), a short code from the full path is added."""
    path = Path(image)
    name = _safe(path.stem)
    if images is not None and any(Path(o).stem == path.stem and str(o) != str(path) for o in images):
        name += "_" + hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return f"{name}_{_safe(cell)}.png"


def write_tables(out_dir: str | Path, rows: list[dict], prof_rows: list[dict]) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return (_write_csv(out / "cells.csv", rows, CELL_COLUMNS_FIRST),
            _write_csv(out / "profiles.csv", prof_rows, ["image", "experiment", "condition", "cell", "target", "profile"]))


def _write_csv(path: Path, rows: list[dict], first: list[str]) -> Path:
    cols = list(first)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:      # BOM: umlauts right in Excel
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})
    return path


def _fmt(v):
    if isinstance(v, float):
        return "" if v != v else f"{v:.6g}"
    return v
