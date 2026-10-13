"""Download a sample of the Allen Cell hiPSC single-cell image dataset for the positive controls (D72).

Viana et al. (2023) Nature 613:345, doi:10.1038/s41586-022-05563-7. Data: Allen Institute for Cell Science,
quilt package "aics/hipsc_single_cell_image_dataset" in s3://allencell (Allen Cell terms of use; cite the
paper and the dataset). Nothing is uploaded anywhere, the files only land in the output folder.

Run on a computer with internet access (not inside the analysis program):

    python3 -m venv allen-env && source allen-env/bin/activate
    pip install quilt3 pandas
    python validation/allen/download_allen.py "<folder>/Allen positive controls"

Confirmatory sample (D79): new cells with seed 1, none of the cells of the first sample:

    python ".../validation/allen/download_allen.py" ".../Allen positive controls seed1" --seed 1 \
        --exclude ".../Allen positive controls/selection.csv"

Per structure it takes N interphase cells that do not touch the field edge (fixed random seed, so the
selection is reproducible) and saves for each cell the raw crop (DNA dye, membrane dye, tagged structure)
and the Allen segmentation crop (nucleus, cell, structure). Already downloaded files are skipped, so the
script can simply be started again after an interruption (incomplete files are downloaded again, network
timeouts are retried). About 20 to 70 MB per cell.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import quilt3

N_PER_STRUCTURE = 20
SEED = 0
# expected localization (Viana et al. 2023, Fig. 1): the answer key of the positive controls
STRUCTURES = {
    "LMNB1": "nuclear envelope (lamina)",
    "NUP153": "nuclear envelope (pores)",
    "HIST1H2BJ": "nucleus (chromatin)",
    "FBL": "nucleus (nucleolus, dense fibrillar component)",
    "NPM1": "nucleus (nucleolus, granular component)",
    "SON": "nucleus (speckles)",
    "TOMM20": "cytoplasm (mitochondria)",
    "SEC61B": "cytoplasm (ER, continuous with the nuclear envelope)",
    "ST6GAL1": "cytoplasm (Golgi)",
    "LAMP1": "cytoplasm (lysosomes)",
    "TUBA1B": "cytoplasm (microtubules)",
    "CTNNB1": "cell surface (adherens junctions)",
    "TJP1": "cell surface (tight junctions)",
    "ACTB": "cell cortex (actin)",
    "AAVS1": "plasma membrane (mEGFP-CAAX in the AAVS1 safe harbour, Allen Cell Catalog AICS-54)",
}


def fetch(entry, target: Path, attempts: int = 6) -> None:
    """Download one file into a .part file and rename it when complete, so an interrupted download never
    looks finished. A file that is already there with the size of the package entry is kept. Network
    timeouts are retried with growing pauses, the run continues where it stopped when started again."""
    size = getattr(entry, "size", None)
    if target.exists() and target.stat().st_size > 0 and (size is None or target.stat().st_size == size):
        return
    part = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            if part.exists():
                part.unlink()
            entry.fetch(part)
            if size is not None and part.stat().st_size != size:
                raise OSError(f"incomplete: {part.stat().st_size} of {size} bytes")
            part.replace(target)
            return
        except Exception as exc:          # network errors of boto3, urllib3 and quilt3 differ by version
            if attempt == attempts:
                raise
            wait = 10 * attempt
            print(f"  {target.name}: {type(exc).__name__}, new attempt in {wait} s ({attempt}/{attempts})", flush=True)
            time.sleep(wait)


def main(out: Path, seed: int = SEED, exclude: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pkg = quilt3.Package.browse("aics/hipsc_single_cell_image_dataset", registry="s3://allencell")
    meta = pkg["metadata.csv"]()
    print(f"{len(meta)} cells in the dataset, columns: {', '.join(meta.columns[:40])} ...")
    sel = meta[meta["structure_name"].isin(STRUCTURES)]
    if "cell_stage" in sel.columns:                     # interphase only
        sel = sel[sel["cell_stage"] == "M0"]
    for col in ("edge_flag", "fov_edge"):               # cells touching the field edge are incomplete
        if col in sel.columns:
            sel = sel[sel[col] == 0]
    # one sample per structure (works with pandas 2 and 3, apply no longer passes the grouping column in 3)
    if exclude is not None:                             # an independent sample: none of these cells again
        old = set(pd.read_csv(exclude)["CellId"].astype(str))
        sel = sel[~sel["CellId"].astype(str).isin(old)]
        print(f"{len(old)} cells of {exclude} left out")
    picked = pd.concat([g.sample(n=min(N_PER_STRUCTURE, len(g)), random_state=seed)
                        for _, g in sel.groupby("structure_name")])
    missing = sorted(set(STRUCTURES) - set(picked["structure_name"]))
    if missing:
        print("Not in the dataset (skipped):", ", ".join(missing))
    picked = picked.assign(expected=picked["structure_name"].map(STRUCTURES))
    picked.to_csv(out / "selection.csv", index=False)
    print(f"{len(picked)} cells selected, list in {out / 'selection.csv'}")
    for k, row in enumerate(picked.itertuples(), 1):
        folder = out / row.structure_name
        folder.mkdir(exist_ok=True)
        for column, suffix in (("crop_raw", "raw"), ("crop_seg", "seg")):
            sub, name = getattr(row, column).split("/", 1)
            fetch(pkg[sub][name], folder / f"{row.CellId}_{suffix}.ome.tiff")
        print(f"{k}/{len(picked)} {row.structure_name} cell {row.CellId}", flush=True)
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Allen Cell single-cell crops for the positive controls")
    ap.add_argument("out", type=Path, help="output folder")
    ap.add_argument("--seed", type=int, default=SEED, help="random selection (0: first sample, 1: confirmatory, D79)")
    ap.add_argument("--exclude", type=Path, default=None, help="selection.csv of cells that must not be drawn again")
    a = ap.parse_args()
    main(a.out.expanduser(), a.seed, a.exclude.expanduser() if a.exclude else None)
