"""Find the images of an experiment folder (0.4.0a1 feedback: add a whole day at once).

Rules, so that a folder as it comes from the microscope can be added as it is:
- Subfolders are searched, too.
- Only files the program can read: processed CZI (Airyscan raw data, several scenes or broken files
  are listed as skipped with the reason), otherwise TIFF.
- If a folder tree holds CZI files, TIFF files in it are ignored: next to the CZI they are usually
  exports, reslices or projections made in Fiji, and would be counted twice.
- Hidden files and files in folders named like raw data ("non processed", "unprocessed", "raw") are left out.
"""

from __future__ import annotations

import re
from pathlib import Path

# whole words only: "raw data" or "Raw" count, "crawling" or "withdrawn" do not
RAW_FOLDER = re.compile(r"(?<![a-z0-9])(raw|rawdata|unprocessed|non[ _-]?processed)(?![a-z0-9])")


def probe_czi(path: Path) -> str | None:
    """Reason why a CZI cannot be analysed, or None. Reads only the header, not the image data."""
    try:
        import czifile

        with czifile.CziFile(path) as czi:
            scenes = czi.scenes
            if not scenes:
                return "no image data"
            if len(scenes) > 1:
                return f"{len(scenes)} scenes"
            if "H" in "".join(scenes[0].dims):
                return "unprocessed Airyscan raw data"
    except Exception as exc:  # broken or incomplete file
        return f"cannot be opened ({exc.__class__.__name__})"
    return None


def probe_nd2(path: Path) -> str | None:
    """Reason to skip a Nikon ND2 file without reading the image data (D90), or None."""
    try:
        import nd2

        with nd2.ND2File(path) as f:
            sizes = dict(f.sizes)
            if f.is_rgb:
                return "RGB colour image"
            if sizes.get("P", 1) > 1:
                return f"{sizes['P']} positions"
            if sizes.get("T", 1) > 1:
                return f"time series ({sizes['T']} time points)"
    except Exception as exc:  # broken or incomplete file
        return f"cannot be opened ({exc.__class__.__name__})"
    return None


def is_raw_folder_name(name: str) -> bool:
    return RAW_FOLDER.search(name.lower()) is not None


def _in_raw_folder(path: Path, root: Path) -> bool:
    return any(is_raw_folder_name(p) for p in path.relative_to(root).parts[:-1])


def find_images(folder: str | Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """(images to add, skipped files with reason), sorted by path."""
    root = Path(folder)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    czi = [p for p in files if p.suffix.lower() in (".czi", ".nd2")]     # microscope files (D90: also Nikon)
    tif = [p for p in files if p.suffix.lower() in (".tif", ".tiff")]
    skipped: list[tuple[Path, str]] = []
    images: list[Path] = []
    for p in czi:
        if _in_raw_folder(p, root):
            skipped.append((p, "in a raw data folder"))
            continue
        reason = probe_nd2(p) if p.suffix.lower() == ".nd2" else probe_czi(p)
        if reason:
            skipped.append((p, reason))
        else:
            images.append(p)
    if czi:
        skipped += [(p, "TIFF next to CZI or ND2 files (export or reslice)") for p in tif]
    else:
        for p in tif:
            if _in_raw_folder(p, root):
                skipped.append((p, "in a raw data folder"))
            else:
                images.append(p)
    return images, skipped
