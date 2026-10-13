# Positive controls: Allen Cell hiPSC single-cell images

Checks that the Localization Tool finds known subcellular localizations (decision D72, criteria D75,
protocol D76, both fixed and committed before the runs they apply to).

Data: Allen Institute for Cell Science, hiPSC single-cell image dataset (Viana et al. 2023, Nature 613:345,
doi:10.1038/s41586-022-05563-7), quilt package `aics/hipsc_single_cell_image_dataset` in `s3://allencell`.
The images are not part of this repository. Sources:
dataset https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset,
Allen Cell data page https://allencell.org/data-downloading.html, AWS Open Data registry
https://registry.opendata.aws/allen-cell-imaging-collections/, terms of use
https://www.allencell.org/terms-of-use.html, structure annotations per cell line in the Allen Cell Catalog
(e.g. AAVS1: https://cell-catalog.allencell.org/cell-line/AICS-54-91/).

1. `download_allen.py <folder> [--seed 1 --exclude <first selection.csv>]`: 20 interphase cells per
   structure, fixed random selection (seed 0 for the first sample, seed 1 without overlap for the
   confirmatory sample, D79), raw crop (DNA dye, membrane dye, tagged structure) and Allen segmentation crop.
2. `run_allen.py <folder> [--protocol 2|3] [--jobs 2] [--label _x]`: analyses every cell and writes
   `allen_cells_runN.csv`, `allen_summary_runN.csv` and `allen_report_runN.txt`.

Protocol 1 (first run, D75 as proposed): program defaults, background 0. Kept for the record: the images
carry a camera dark level of about 400 in every channel, so every enrichment comes out close to 1.

Protocol 2 (D76, default): the camera dark level of each crop (0.1st percentile of the DNA channel) is
subtracted from all channels through the program's background setting. Variant A uses the Allen cell and
nucleus masks (zones and measurement only), B the Allen cell mask and the program's own nucleus, C the
program's own cell edge and nucleus on the image set to the dark level outside the Allen cell.

Protocol 3 (D79, confirmatory run on the seed 1 sample, criteria committed before the download): analysis
as protocol 2, each structure passes when it separates from a comparison group (AUC at least 0.9, or at
most 0.1 where it must be lower): nuclear E nucleus against cytoplasmic, envelope (E envelope zone minus
E nucleus interior) against nuclear, cytoplasmic E nucleus against nuclear, SEC61B envelope minus interior
against cytoplasmic, surface including AAVS1 E cortex shell against cytoplasmic. Plus the agreement of
variant C with variant A.

Protocols 1 and 2: a structure passes when at least 90 % of its cells meet the criterion of its group (D75):
nuclear E nucleus > 1.5; nuclear envelope E envelope zone > 1.5 and > E nucleus interior; cytoplasmic
organelles E nucleus < 0.5; ER E nucleus < 0.5 and E envelope zone > E nucleus interior; junctions and
actin E cortex shell > 1.2; diffuse mEGFP 0.5 < E nucleus < 2 and no zone above 1.5.

Notes from the runs of 12 Oct 2026 (details in docs/METRICS.md §4.13):

- The OME headers carry no voxel size. It is read from the column `scale_micron` (isotropic 0.108 µm).
- AAVS1 in this dataset is mEGFP-CAAX at the plasma membrane (Allen Cell Catalog AICS-54), not diffuse
  mEGFP as assumed in D75.
- E nucleus below 0.5 is not reached by cytoplasmic structures even with the Allen masks (axial blur in
  cells whose nucleus fills most of the cell height). A confirmatory run on new cells with criteria based
  on group separation is proposed in D79.
