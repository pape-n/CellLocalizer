# Installation

## Windows (no administrator rights needed)

1. Download `CellLocalizer-<version>-Windows-Setup.exe` from the release page.
2. Double-click it. Windows SmartScreen reports "Windows protected your PC" because the program is not signed: click "More info", then "Run anyway".
3. The program is installed for the current user in `%LOCALAPPDATA%\Programs\CellLocalizer` (an update of a version before 2.0.1, then called Localization Tool, stays in its folder and replaces the old shortcuts).

Without installation: unpack the portable zip and start `CellLocalizer.exe`.

## macOS (Apple Silicon)

1. Open the DMG and drag "CellLocalizer" into Applications. An older "Localization Tool" there can be deleted; settings and analyses stay.
2. On the first start macOS reports that the developer cannot be verified: right-click the program in Finder, choose "Open" and confirm. On recent macOS versions: System Settings, Privacy & Security, "Open Anyway".

If the program starts but the image stays black, or it runs over Remote Desktop: use "CellLocalizer (safe graphics)" in the Windows start menu, or start with `--software-opengl` (draws without the graphics card, slower).

## From the source code

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     (Windows: .venv\Scripts\pip ...)
.venv/bin/python -m localization_tool
```

Self-test without the user interface: `python -m localization_tool --self-test report.txt`. Tests: `pytest -q` (the user interface tests need a display, on Linux `xvfb-run -a pytest -q`).

## Recording images for comparable values

The program reads these settings from Zeiss CZI files and reports when they differ within one experiment (sheet "Acquisition" in Results.xlsx).

- Airyscan processing with a **fixed strength** instead of "Auto". The automatic strength changes sharpness and noise from image to image.
- The same laser power, detector gain and pixel size for all cells of an experiment.
- A z range of at least 1 µm above and below the cell, so **start in the glass below the cell**. A stack that starts inside the cell misses its lowest layer, which affects the basal layer and the cortex shell.
- Pixel size for Airyscan as recommended by ZEN ("Optimal", about 35 to 40 nm with a 63×/1.4 objective).
- Note the mounting medium for the methods text.
- **Background:** Airyscan-processed images are clipped at 0 and need no background. Images from cameras (spinning disk, widefield) carry a dark level in every channel: open an image, then Settings > Channels > "Estimate dark level from the open image". Without it every enrichment comes out close to 1, and the program shows the hint "background not subtracted?".

## Measuring the channel offset once

Different colours are focused at slightly different heights. One plane (0.13 µm) of offset changes the basal enrichment by a few percent up to about 20 %.

1. Put multicolour beads (for example TetraSpeck 0.1 µm) on a coverslip with the same mounting medium as the cells.
2. Record them with the same objective, the same channels in the same order and the same z step as the cells, with the same Airyscan processing. 10 to 50 beads in the image are enough.
3. In the program: Settings > Channel offset > choose the reference channel (for example actin) > "Measure from a bead image ...". The program shows the offset of every channel in nm and the spread over the beads.
4. Accept with OK. Bead channels are matched by dye name, otherwise by order.
5. For all new analyses: do the same under Settings on the start page. There the offset is stored per dye (for example "Alexa Fluor 647") and applies to every channel with that dye.
