# Validation with public images of known localization

The program was checked on the Allen Cell hiPSC single-cell image dataset (Viana et al. 2023, Nature 613:345, doi:10.1038/s41586-022-05563-7; data: https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset). Each cell line carries one endogenously tagged structure whose localization is known (Allen Cell Catalog, https://cell-catalog.allencell.org/). Images: live hiPS cells, spinning-disk confocal, resampled to isotropic voxels of 0.108 µm, with a DNA dye, a membrane dye and the tagged structure. Scripts: `validation/allen/`.

Sample: 20 interphase cells per structure that do not touch the field edge, fixed random selection, 15 structures, 300 cells. Roles: DNA dye = Nucleus, membrane dye = Membrane, tagged structure = Target. Program defaults otherwise, not tuned to these images. Three variants per cell: (A) the Allen cell and nucleus segmentations as masks, so only zones and measurement are tested; (B) the Allen cell, the program's own nucleus; (C) the program's own cell edge and nucleus.

## Runs

The criteria of each run were fixed and committed in the development repository before the run. Every run is reported.

1. **Run 1**, program defaults (background 0): failed for 29 of 30 structure and variant combinations. The camera images carry a dark level of about 400 grey values in every channel; without subtracting it every enrichment is close to 1. This led to the dark level estimate and the background hint of version 2.0.
2. **Run 2**, dark level subtracted, criteria as fixed thresholds per cell (for example enrichment in the nucleus above 1.5 for nuclear proteins, below 0.5 for cytoplasmic ones, in at least 90 % of the cells). With the Allen masks all nuclear and nuclear envelope proteins and β-catenin passed. Cytoplasmic proteins did not reach an enrichment below 0.5 in the nucleus even with the Allen masks (median 0.59 to 0.87): the nucleus fills most of the height of these cells, and the axial blur of the spinning disk carries the signal of organelles above and below it into the nucleus. The threshold was not reachable with these images, independent of the program. AAVS1 turned out to be a plasma membrane marker (mEGFP-CAAX), not diffuse GFP as assumed.
3. **Run 3**, same images, version with the new noise test, criteria as separation of each structure from a comparison group (area under the ROC curve at least 0.9; for example nuclear proteins against cytoplasmic proteins in nuclear enrichment). These criteria were written after seeing run 2, so on these cells the result is exploratory: with the Allen masks all 15 structures passed, with the program's own masks 14 of 15. SEC61B (ER) failed with the own nucleus, because the nucleus was too tall (see below).

A confirmatory run with the same criteria on 300 new cells (no overlap with the first sample) is under way; its result will be added here.

## Agreement of the program's masks with the Allen masks

Cell: Dice 0.96 (median), cell volume 7 to 8 % larger (the cell edge lies at the outer foot of the blurred staining). Nucleus: Dice 0.80 to 0.85, volume 1.41 times larger. In the plane the nucleus edge lies on the Allen edge; in z the nucleus reaches about 1 µm beyond the lamina above and below, because the DNA signal of these images falls to 20 % only 0.65 to more than 4 µm beyond the nucleus. On synthetic nuclei the half-maximum edge keeps the volume within 2 % for axial blur up to a Gaussian sigma of 0.3 µm (Airyscan range), +4 % at 0.45 µm, +10 % at 0.6 µm and +26 % at 0.9 µm (`tests/test_v14.py`).

## What this means for users

Enrichments and signal shares rank localizations reliably and separate groups of proteins completely. Their absolute values depend on optics, background and mask edges: compare conditions recorded the same way, and do not carry fixed thresholds across microscopes. With strong axial blur, read the nuclear envelope zone and the nucleus interior with care.
