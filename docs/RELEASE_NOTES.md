# CellLocalizer 2.1.0

13 October 2026. Changes after the first use in the lab. CellLocalizer 2.0.1 was the first public release (2.0.0 as Localization Tool). Open source under the BSD 3-Clause License. Analyses saved with earlier versions open and keep their settings; the program offers to recompute their cells so that new columns appear, and cells whose outlines stay the same keep "accepted".

## What it does

The program measures in confocal 3D stacks (Zeiss CZI, Nikon ND2, also ImageJ and OME TIFF) where a protein sits in a cell: the share of its signal in the nucleus and the cytoplasm, in the actin part of the cytoplasm, in zones (nuclear envelope, below the cell surface, above the coverslip, front half) and as enrichment (1 = as dense as on average in the cell). The result is a folder with `Results.xlsx` (sheets ready for GraphPad Prism), a control image for every cell and the analysis file to open again.

## Installation

Windows: `CellLocalizer-2.1.0-Windows-Setup.exe`, no administrator rights needed, or the portable zip. macOS (Apple Silicon): the DMG. The programs are not signed: see `INSTALL.md` for the first start.

## Getting started

1. On the start page "New analysis". "Add folder ...": add an experiment day with all subfolders. Experiment and condition are suggested from the folder and file names.
2. Open an image and press "Next". The channel roles are guessed from the file name ("Act488 TTN647"): check them and "Save as panel ...".
3. Check the cell edge, exclude neighbouring cells with "Draw". Check the compartments, adjust edges only with the factors.
4. In step 5 accept the cell or exclude it with a reason. "Save results ..." at the bottom right. From then on every cell is saved automatically.
5. Many images: "Analyze all images" computes every image without a cell, "Next image" leads through the cells "to check". Only accepted cells go into the Prism sheets.

The guide is under Help and on the start page, in English and German. Help > "Methods text" writes the complete methods section with all settings and explains how every value is computed. "Practice with a synthetic cell" on the start page opens a synthetic cell to learn the steps.

## What to watch for

- **Orange hints** in step 5 and in the control image change no number. They say what to check: stack cut off at the top, possibly two cells (second nucleus), much DAPI outside the nucleus mask, target without staining (control), very high correlation with actin (crosstalk?), background not subtracted.
- **Background:** 0 fits Airyscan-processed images, which are clipped at 0. For camera images (spinning disk, widefield) set the dark level per channel (Settings > Channels > "Estimate dark level from the open image"), otherwise all enrichments come out close to 1.
- **Cell edge:** per channel where the staining fades into the background (Li), i.e. outside the actin cortex. "Half maximum" stays selectable under Settings > Masks > Cell.
- **Nucleus edge:** where the DAPI signal is halfway between nucleus and cytoplasm. With strong axial blur (far worse than Airyscan) the nucleus becomes too tall; then read the nuclear envelope zone and the nucleus interior with care.
- **Empty cells** in Excel mean "no value", for example without a nucleus or actin channel. They are never 0.
- **Statistics:** cells of one coverslip are not independent replicates. The sheet "Replicates" holds the means per experiment and condition for the statistics in Prism.
- **Absolute values:** enrichments and shares rank localizations reliably, but their absolute values depend on optics, background and the mask edges. Compare conditions recorded the same way; do not carry fixed thresholds across microscopes.

## Known limits

- An image is analysed as a whole; the largest connected structure is the cell. Several cells per image are analysed one after another with "Next cell in this image" and exclusion polygons.
- Very large images (tile scans above about 500 MB) need much memory; the program warns before a series.
- The programs are not signed; Windows and macOS warn on the first start.

## New in 2.1.0

- **Nikon ND2 files:** z-stacks from NIS-Elements open directly, with voxel size, channel names, wavelengths and colours. The coverslip side comes from the z direction and the stand (inverted Ti/Ti2, upright Ni); otherwise the setting "Coverslip side" decides. Time series and multi-position files are skipped with a reason. This reader is new: please report a file that does not open or shows wrong sizes (Help > Report a problem).
- **Comparison channel:** a new channel role "Comparison" for a second stained protein (for example Vimentin). It is measured like a target and gets its own mask (colour, factor in step 4, shown in 3D and in the cell viewer). Every target is also measured in the stained area of the comparison channel: share of the signal there, enrichment, and the share of the target mask inside the area. Results.xlsx has a Prism sheet "<target> Enr in <name>".
- **Target values from the target mask:** the shares now count only the target signal inside the target mask, so correcting the mask with its factor changes the numbers, and diffuse background and the faint halo of bright structures no longer count. The sensitivity span also moves the target edge. Settings > Masks > "Target signal counted" switches back to all signal. Saved analyses keep how they were computed: with the older settings 2.1.0 gives the same values as 2.0.1. A target without staining above noise (secondary antibody control) gets no mask, its values stay empty.
- **Polarity:** the front direction points to the actin-rich mask (its factor and painted masks move it). The arrow is long and bold for a clear front and short and thin for a nearly even cell. New column polarity_clear (1 from the polarity index set in Settings > Extras on, provisional default 0.25: set it from resting cells). New: enrichment of the rear half and the ratio rear / front (density, 1 = even).
- **Nuclear envelope:** the envelope zone is also split into an inner half (inside the nucleus mask) and an outer half. "Highest enrichment at ... µm from the nucleus edge" says on which side the signal peaks, averaged over the whole nucleus; empty when the profile has no maximum near the edge. Optional: envelope zones only at the side of the nucleus, without top and bottom where the axial blur is largest (Settings > Zones).
- **Stability:** every display setting stays as you left it when you move between the steps (the cytoplasm mask no longer comes back after the results in 3D); the cell viewer frees its memory and its 3D view when closed and computes 3D surfaces in the background, so the window no longer freezes; 3D surfaces of a previous cell no longer appear; the program waits for a running calculation before it closes; a crash on Windows is now kept in crash_previous.log.

## New in 2.0.1

- The program is now called **CellLocalizer** (window, installer, start menu). Analyses, settings and file formats stay the same; an update replaces the old shortcuts.

## New in 2.0.0

- **Public and self-standing:** BSD 3-Clause License, citation file, English documents, Help > "Report a problem" with the link to the issue page.
- **Cell viewer:** a colour for every mask and zone, target masks first, zones after a line; the picture shows only the cell (1 µm around it), as in the preview; the saved 3D picture shows the same frame as the window; even lighting of the 3D surfaces from every side; the viewer remembers its settings per analysis; the size of the saved picture is shown.
- **Background hint and dark level estimate** for camera images.
- **Noise test from neighbouring voxels** (1.4.0): weak but clearly visible stains in crowded images are no longer taken for empty channels. Airyscan images compute exactly as before.
- General words in the condition list (WT, KO, KD, control, treated, untreated, vehicle, stimulated, resting); add your own under Settings.

## Feedback

Help > "Report a problem ..." says what to report and copies the details of the computer. Issues: https://github.com/pape-n/CellLocalizer/issues
