# Localization Tool

Desktop program that measures where a protein sits in a cell, in confocal 3D stacks. For every cell it reports which share of the signal lies in the nucleus and in the cytoplasm, how much of it in the actin part of the cytoplasm, and how the signal is enriched near the nuclear envelope, below the cell surface and above the coverslip. Which channel is nucleus, actin, membrane or target is set per data set, so the program is not tied to one protein or cell type.

- Loads Zeiss CZI (Airyscan-processed or confocal) and ImageJ or OME TIFF with voxel size, channel metadata and z direction
- Guides through five steps per cell: channels, cell edge, compartments, check, accept; masks can be adjusted with factors and corrected by hand
- Computes signal shares, enrichments, distance zones and profiles in 3D, plus polarity, Pearson and Manders coefficients and an optional crosstalk correction
- Analyses whole experiment folders, suggests experiment and condition from the names, supports blind review
- Writes `Results.xlsx` with sheets ready for GraphPad Prism (one column per condition, replicate means), a control image per cell and a complete methods text (English and German, also as Word file)
- Shows cells as planes, vertical sections or in 3D and exports figures (PNG) and the recorded values (TIFF)
- Reports anomalies as hints without changing any value (cut-off stack, second nucleus, unstained target, possible crosstalk, background not subtracted)

The measured quantities, masks and their rules are described by the methods text in the program (Help > Methods text). Validation with public images of known localization: [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Download

Installers for Windows (no administrator rights needed, or portable zip) and macOS (Apple Silicon) are on the [releases page](https://github.com/pape-n/CellLocalizer/releases). First start of the unsigned programs and recording advice: [`docs/INSTALL.md`](docs/INSTALL.md). What is new: [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md).

## From the source code

Python 3.12:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     (Windows: .venv\Scripts\pip ...)
.venv/bin/python -m localization_tool
```

Command line without the user interface:

```
python -m localization_tool init "<image folder>" --out analysis.lta.json --roles "DAPI=Nucleus,488=Actin,647=Target:MyProtein"
python -m localization_tool analyze analysis.lta.json --out results/
```

Tests: `pytest -q` (the user interface tests need a display; on Linux `xvfb-run -a pytest -q`). Self-test of an installed program: `--self-test report.txt`.

## Citation

If you publish results obtained with this program, please cite the version given in the methods text (see [`CITATION.cff`](CITATION.cff)) and let the author know (npape2@uni-muenster.de).

## License

BSD 3-Clause License, see [`LICENSE`](LICENSE). Built with napari, NumPy, SciPy, scikit-image, tifffile, czifile (BSD), PySide6 (LGPL) and openpyxl (MIT).

Developed by Nils Pape, Institute of Physiology II, University of Münster.
