"""Analysis file (.lta.json, D15): images, channel roles, cells and parameters.

Everything needed to reproduce every number. Paths are stored relative to the file when possible.
Masks corrected by hand are added in phase 2 as separate files next to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .channels import ChannelAssignment
from .conditions import DEFAULT_VOCABULARY
from .crop import CropBox
from .params import AnalysisParams

FORMAT_VERSION = 1
SUFFIX = ".lta.json"




# settings added after an analysis may have been saved, with the value that reproduces the older result
LEGACY_PARAMS = {"actin_per_plane": False}          # 1.2.1, D70

def code_number(code: str) -> int:
    """'Image 07' -> 7; no code sorts last."""
    digits = "".join(ch for ch in (code or "") if ch.isdigit())
    return int(digits) if digits else 10 ** 9

@dataclass
class CellSpec:
    name: str
    box: CropBox | None = None                     # None: the whole image is one cell
    exclusions: list = field(default_factory=list)  # polygons of (y, x) in source voxel coordinates
    status: str = "automatic"                       # automatic, accepted, "accepted, corrected", excluded
    note: str = ""
    overrides: dict = field(default_factory=dict)   # parameters changed for this cell only
    mask_file: str = ""                             # corrected masks (.npz next to the analysis file)
    results: dict | None = None                     # stored table rows, so an analysis reopens instantly

    def to_dict(self) -> dict:
        d = {"name": self.name, "box": self.box.to_dict() if self.box else None,
             "exclusions": [[list(map(float, p)) for p in poly] for poly in self.exclusions],
             "status": self.status, "note": self.note}
        if self.overrides:
            d["overrides"] = self.overrides
        if self.mask_file:
            d["mask_file"] = self.mask_file
        if self.results is not None:
            d["results"] = self.results
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CellSpec":
        box = CropBox.from_dict(d["box"]) if d.get("box") else None
        return cls(name=str(d.get("name", "cell")), box=box, exclusions=d.get("exclusions", []) or [],
                   status=str(d.get("status", "automatic")), note=str(d.get("note", "")),
                   overrides=dict(d.get("overrides") or {}), mask_file=str(d.get("mask_file", "")),
                   results=d.get("results"))


@dataclass
class ImageSpec:
    path: Path
    experiment: str = ""
    condition: str = ""
    assignments: list[ChannelAssignment] | None = None   # None: use the analysis panel
    cells: list[CellSpec] = field(default_factory=list)
    code: str = ""                       # blind review (D57): shown instead of name and condition
    revealed: bool = False               # the name was shown during blind review ("Show name")

    def to_dict(self, base: Path | None) -> dict:
        p = self.path
        if base is not None:
            try:
                p = Path(p).resolve().relative_to(base.resolve())
            except ValueError:
                pass
        d = {"path": str(p).replace("\\", "/"), "experiment": self.experiment, "condition": self.condition,
             "cells": [c.to_dict() for c in self.cells]}
        if self.assignments is not None:
            d["assignments"] = [vars(a) for a in self.assignments]
        if self.code:
            d["code"] = self.code
        if self.revealed:
            d["revealed"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict, base: Path | None) -> "ImageSpec":
        p = Path(d["path"])
        if not p.is_absolute() and base is not None:
            p = base / p
        assignments = [ChannelAssignment(**a) for a in d["assignments"]] if d.get("assignments") else None
        return cls(path=p, experiment=str(d.get("experiment", "")), condition=str(d.get("condition", "")),
                   assignments=assignments, cells=[CellSpec.from_dict(c) for c in d.get("cells", [])],
                   code=str(d.get("code", "")), revealed=bool(d.get("revealed", False)))


@dataclass
class AnalysisFile:
    images: list[ImageSpec] = field(default_factory=list)
    panel: list[ChannelAssignment] | None = None
    params: AnalysisParams = field(default_factory=AnalysisParams)
    vocabulary: list[str] = field(default_factory=lambda: list(DEFAULT_VOCABULARY))   # D37
    blind: bool = False                  # blind review (D57); new analyses take the program default

    def assign_codes(self, seed: int | None = None) -> None:
        """Give every image without a code a random code 'Image NN' (D57). The order of the codes is
        random, so going through the images by code does not follow folders or conditions. Numbers are
        unique also when the list grows past 99 images (the new codes then get more digits)."""
        import random

        used = {code_number(im.code) for im in self.images if im.code}
        todo = [im for im in self.images if not im.code]
        if not todo:
            return
        total = len(used) + len(todo)
        free = [k for k in range(1, total + 1) if k not in used]
        k = total
        while len(free) < len(todo):            # numbers above the count are taken (images removed)
            k += 1
            if k not in used:
                free.append(k)
        random.Random(seed).shuffle(free)
        width = max([2, len(str(total))] + [len(im.code.split()[-1]) for im in self.images if im.code])
        for im, k in zip(todo, free):
            im.code = f"Image {k:0{width}d}"

    def blind_order(self) -> list[int]:
        """Indices of the images in the order they are listed: by code while blind review is on, else as
        added (D57). The position in the list then says nothing about folder or condition."""
        order = list(range(len(self.images)))
        if self.blind:
            order.sort(key=lambda k: code_number(self.images[k].code))
        return order

    def shown_name(self, image: ImageSpec) -> str:
        """File name, or the code while blind review hides it."""
        return image.code if self.blind and image.code and not image.revealed else image.path.name

    def assignments_for(self, image: ImageSpec) -> list[ChannelAssignment]:
        a = image.assignments or self.panel
        if a is None:
            raise ValueError(f"{image.path.name}: no channel roles given (neither for the image nor as panel)")
        return a

    def to_dict(self, base: Path | None = None) -> dict:
        return {"format": "Localization Tool analysis", "version": FORMAT_VERSION,
                "panel": [vars(a) for a in self.panel] if self.panel else None,
                "params": self.params.to_dict(), "vocabulary": list(self.vocabulary), "blind": self.blind,
                "images": [im.to_dict(base) for im in self.images]}

    @classmethod
    def from_dict(cls, d: dict, base: Path | None = None) -> "AnalysisFile":
        if int(d.get("version", 0)) > FORMAT_VERSION:
            raise ValueError("This analysis file was written by a newer version of the program.")
        panel = [ChannelAssignment(**a) for a in d["panel"]] if d.get("panel") else None
        pd = dict(d.get("params") or {})
        for key, before in LEGACY_PARAMS.items():   # settings added later: a saved analysis keeps how it was computed
            if pd and key not in pd:
                pd[key] = before
        params = AnalysisParams.from_dict(pd)
        vocab = d.get("vocabulary")
        return cls(images=[ImageSpec.from_dict(i, base) for i in d.get("images", [])], panel=panel, params=params,
                   vocabulary=list(vocab) if vocab is not None else list(DEFAULT_VOCABULARY),
                   blind=bool(d.get("blind", False)))


def save_masks(path: str | Path, cell: "np.ndarray | None" = None, labels: "np.ndarray | None" = None) -> Path:
    """Corrected masks of one cell on the crop grid. labels: 1 nucleus, 2 actin, 3 cytoplasm."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if cell is not None:
        data["cell"] = np.asarray(cell, bool)
    if labels is not None:
        data["labels"] = np.asarray(labels, np.uint8)
    np.savez_compressed(path, **data)
    return path


def load_masks(path: str | Path) -> dict:
    """Masks for analyze_cell from a file written by save_masks."""
    import numpy as np

    with np.load(path) as f:
        if "labels" in f:
            lab = f["labels"]
            return {"cell": lab > 0, "nucleus": lab == 1, "actin": (lab == 2) | (lab == 4)}
        if "cell" in f:
            return {"cell": f["cell"].astype(bool)}
    return {}


def _path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.name.endswith(SUFFIX) else path.with_name(path.name + SUFFIX)


def dumps(af: AnalysisFile, path: str | Path) -> str:
    """The file content for saving at path (paths relative to its folder). Quick, so the program can
    take this snapshot and write it in the background."""
    return json.dumps(af.to_dict(_path(path).parent), indent=1)


def write_text(path: str | Path, text: str) -> Path:
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for k in range(5):                       # Windows virus scanners hold new files for a moment
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if k == 4:
                    raise
                import time
                time.sleep(0.3)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def save(af: AnalysisFile, path: str | Path) -> Path:
    return write_text(path, dumps(af, path))


def load(path: str | Path) -> AnalysisFile:
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8-sig"))      # -sig: also files saved with a BOM
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name} could not be read as analysis file ({exc}).") from exc
    if not isinstance(d, dict) or "images" not in d:
        raise ValueError(f"{path.name} is not an analysis file of the Localization Tool.")
    try:
        return AnalysisFile.from_dict(d, path.parent)
    except ValueError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"{path.name} is damaged or incomplete ({exc.__class__.__name__}: {exc}).") from exc
