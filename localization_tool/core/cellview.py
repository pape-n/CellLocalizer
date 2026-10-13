"""One stored cell, ready for the cell viewer (D71): the crop of the image, the masks of the analysis
and the zones. The masks are computed with the settings stored in the analysis (the same call as
"Apply settings to all cells") and kept in the analysis folder ("Cell masks/"), so a cell opens at
once the next time. A kept file is used only if it belongs to the stored result: same fingerprint of
box and masks (D66) or, for results without fingerprint, the same settings. Nothing imports Qt here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import figure
from .analysis import _shift, channel_offset
from .batch import _masks_for, analyze_spec, mask_digest
from .crop import CropBox, crop
from .image import ImageStack
from .project_file import AnalysisFile, CellSpec, ImageSpec
from .reader import read_image

CACHE_DIR = "Cell masks"


@dataclass
class CellData:
    stack: ImageStack                  # crop of the cell, channels moved by the channel offset (whole voxels)
    assignments: list
    masks: dict[str, np.ndarray]       # masks of the analysis on the crop grid
    view_masks: dict[str, np.ndarray]  # masks and zones for showing
    voxel_um: tuple[float, float, float]   # with the axial scale of the analysis
    coverslip_first: bool
    default_z: int
    name: str                          # "image / cell"
    condition: str
    experiment: str


def cache_path(base_dir: str | Path, image: ImageSpec, spec: CellSpec) -> Path:
    return Path(base_dir) / CACHE_DIR / f"{figure.file_stem(image.path.stem)}__{figure.file_stem(spec.name)}.npz"


def _load_cache(path: Path, spec: CellSpec, box: CropBox):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            masks = {k: f[k] for k in f.files if k != "meta"}
    except Exception:                       # a damaged or foreign file: compute again
        return None
    stored = spec.results or {}
    if stored.get("mask_digest"):
        if stored["mask_digest"] != mask_digest(box, masks):
            return None
    elif meta.get("params") != stored.get("params"):
        return None
    return masks, meta


def _save_cache(path: Path, masks: dict, meta: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **masks)
    except OSError:                         # a read-only analysis folder: the viewer still works
        pass


def load_cell(af: AnalysisFile, image: ImageSpec, spec: CellSpec, base_dir: str | Path | None,
              stack: ImageStack | None = None) -> CellData:
    """Read the image, cut out the cell and get its masks (kept file or computed again)."""
    stack = stack if stack is not None else read_image(image.path)
    assignments = af.assignments_for(image)
    box = (spec.box or CropBox(0, stack.shape_zyx[0], 0, stack.shape_zyx[1], 0, stack.shape_zyx[2])).clipped(
        stack.shape_zyx)
    path = cache_path(base_dir, image, spec) if base_dir is not None else None
    cached = _load_cache(path, spec, box) if path is not None else None
    if cached is None:
        corrected = _masks_for(image, spec, base_dir, None)
        c, box, r = analyze_spec(stack, assignments, af.params, spec, corrected)
        masks = {k: np.asarray(v) for k, v in r.masks.items() if v is not None and np.ndim(v) == 3}
        meta = {"params": (spec.results or {}).get("params"), "voxel_um": list(r.voxel_um),
                "coverslip_first": bool(r.coverslip_first)}
        if path is not None:
            _save_cache(path, masks, meta)
    else:
        masks, meta = cached
        c = crop(stack, box)
    voxel = tuple(meta["voxel_um"])
    data = np.asarray(c.data)
    shifted = []
    for i, a in enumerate(assignments):     # the view shows the channels aligned as analysed (D55)
        fl = c.channels[i].fluor if i < len(c.channels) else None
        off = channel_offset(af.params, a.label, fl)
        shifted.append(_shift(data[i], off, c.voxel_um) if off and a.role != "Ignore" else data[i])
    c = ImageStack(np.stack(shifted), c.voxel_um, c.channels, source=c.source, origin=c.origin, info=dict(c.info),
                   is_crop=True)
    p = af.params
    vm = figure.view_masks(masks, voxel, p.envelope_um, p.cortex_um, p.basal_um, bool(meta["coverslip_first"]))
    z = figure.default_plane(c.data, assignments, vm)
    return CellData(c, assignments, masks, vm, voxel, bool(meta["coverslip_first"]), z,
                    f"{image.path.name} / {spec.name}", image.condition, image.experiment)


def cells_of(af: AnalysisFile, accepted_only: bool = False) -> list[tuple[ImageSpec, CellSpec]]:
    """Every stored cell that is not excluded (or only the accepted ones), in the order of the analysis."""
    out = []
    for im in af.images:
        for c in im.cells:
            if c.status == "excluded":
                continue
            if accepted_only and not str(c.status).startswith("accepted"):
                continue
            out.append((im, c))
    return out
