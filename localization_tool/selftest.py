"""Self-test of the installed program: python -m localization_tool --self-test report.txt

Runs without a window. Checks that all libraries load in the build and that the core works.
Exit code 0 means all checks passed.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path


def _checks():
    import numpy as np

    from localization_tool.core.channels import suggest_assignments, validate
    from localization_tool.core.crop import CropBox, crop
    from localization_tool.core.preview import otsu_mask
    from localization_tool.core.reader import read_image, write_imagej_tiff
    from localization_tool.core.synthetic import make_synthetic_cell

    stack = make_synthetic_cell()
    yield "synthetic cell", f"shape {stack.data.shape}, {stack.nbytes_mb:.1f} MB"

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "selftest.tif"
        write_imagej_tiff(stack, p)
        back = read_image(p)
        assert back.data.shape == stack.data.shape, "TIFF shape changed"
        assert np.array_equal(back.data, stack.data), "TIFF values changed"
        assert np.allclose(back.voxel_um, stack.voxel_um, rtol=1e-4), f"voxel size {back.voxel_um}"
        assert [c.short for c in back.channels] == [c.short for c in stack.channels], "channel names"
    yield "TIFF write and read", "values, voxel size and channel names identical"

    box = CropBox(0, stack.shape_zyx[0], 40, 150, 30, 160)
    cell = crop(stack, box)
    assert cell.origin == (0, 40, 30)
    assert np.array_equal(cell.data, stack.data[:, :, 40:150, 30:160])
    yield "crop", "no resampling, origin kept"

    assignments = suggest_assignments(stack.channels, "TTN647 Act488 DAPI")
    assert assignments[2].role == "Nucleus"
    assignments[0].role = "Target"
    assert validate(assignments) == [], validate(assignments)
    yield "channel roles", "DAPI recognised as Nucleus"

    t = time.perf_counter()
    mask, thr = otsu_mask(stack.data[2], stack.voxel_um)
    assert mask.sum() > 0
    yield "mask preview", f"{int(mask.sum())} voxels, {time.perf_counter() - t:.2f} s"

    from localization_tool.core.analysis import analyze_cell
    from localization_tool.core.channels import ChannelAssignment
    from localization_tool.core.synthetic import make_golden_cell

    t = time.perf_counter()
    golden, truth = make_golden_cell()
    roles = [ChannelAssignment("Target", "T", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
             ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]
    res = analyze_cell(golden, roles, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    dev = max(abs(res.target_values["T"][k] - v) for k, v in truth["expected"].items() if k.endswith("_pct"))
    assert dev < 0.5, f"golden cell deviates by {dev:.2f} percentage points"
    auto = analyze_cell(golden, roles)
    assert auto.masks["nucleus"].any() and auto.masks["actin"].any(), "automatic masks empty"
    yield "analysis", f"golden cell within {dev:.2f} percentage points, {time.perf_counter() - t:.1f} s"

    with tempfile.TemporaryDirectory() as tmp:
        from localization_tool.core.qc_images import control_image

        png = control_image(golden.data, roles, auto.masks, auto.voxel_um, "self-test", ["golden cell"],
                            Path(tmp) / "control.png")
        assert png.stat().st_size > 1000
    yield "control image", "PNG written"

    with tempfile.TemporaryDirectory() as tmp:
        from localization_tool.core.batch import result_rows
        from localization_tool.core.crop import CropBox
        from localization_tool.core.export_excel import write_results

        rows, prof = result_rows("golden", "E1", "C1", "cell1", "accepted", CropBox(0, 1, 0, 1, 0, 1), auto)
        xlsx = write_results(Path(tmp) / "Results.xlsx", rows, prof, {"test": "self-test"})
        assert xlsx.stat().st_size > 3000
    yield "Excel export", "Results.xlsx written"

    from localization_tool.gui.surfaces import mask_surface
    from localization_tool.paths import resource

    mesh = mask_surface(auto.masks["nucleus"], auto.voxel_um)
    assert mesh is not None and len(mesh[1]) > 100, "3D surface of the nucleus empty"
    assert resource("icon.png").exists(), "program icon missing"
    yield "3D surfaces and resources", f"{len(mesh[1])} faces for the nucleus, icon found"

    with tempfile.TemporaryDirectory() as tmp:
        from localization_tool.core.methods_text import methods_text, save_docx
        from localization_tool.core.offsets import estimate_offsets, synthetic_beads

        doc = save_docx(methods_text(lang="de"), Path(tmp) / "methods.docx")
        assert doc.stat().st_size > 5000, "Word file too small"
        beads = synthetic_beads(shifts_um=((0, 0, 0), (0.13, 0.0, 0.0)))
        off = estimate_offsets(beads, (0.13, 0.05, 0.05), 0, ["A", "B"])["channels"]["B"]
        assert abs(off["dz_um"] - 0.13) < 0.04, f"bead offset {off}"
    yield "methods text and bead offset", f"Word file written, bead offset z {off['dz_um'] * 1000:.0f} nm"

    from localization_tool.core import figure

    data = np.asarray(stack.data)
    cell_m = data[-1] > 0
    vs = figure.FigureSettings(mode="section", angle_deg=30,
                               channels=[figure.ChannelView(c.short, "#FFFFFF", high=float(data[i].max() or 1))
                                         for i, c in enumerate(stack.channels)],
                               masks=[figure.MaskView("cell", True)], show_legend=True, title="self-test")
    view = figure.make_view(data, {"cell": cell_m}, stack.voxel_um, vs)
    pic = figure.render(view, vs)
    with tempfile.TemporaryDirectory() as tmp:
        figure.save_png(pic, Path(tmp) / "v.png")
        figure.save_tiff(view, [c.short for c in stack.channels], Path(tmp) / "v.tif")
    yield "cell viewer pictures", f"section {pic.shape[1]} x {pic.shape[0]} px, PNG and TIFF written"

    import imagecodecs

    a = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 7) % 4000
    assert imagecodecs.zstd_decode(imagecodecs.zstd_encode(a.tobytes())) == a.tobytes(), "zstd"
    assert np.array_equal(imagecodecs.jpegxr_decode(imagecodecs.jpegxr_encode(a, level=1.0)), a), "JPEG XR"
    yield "CZI compression", f"zstd and JPEG XR encode and decode (imagecodecs {imagecodecs.__version__})"

    import czifile  # noqa: F401
    import openpyxl  # noqa: F401
    import scipy  # noqa: F401
    import skimage  # noqa: F401
    import tifffile  # noqa: F401
    yield "core libraries", "czifile, tifffile, scipy, scikit-image, openpyxl"

    t = time.perf_counter()
    import napari
    import qtpy
    import vispy
    yield "viewer libraries", (f"napari {napari.__version__}, vispy {vispy.__version__}, "
                               f"Qt API {qtpy.API_NAME}, import {time.perf_counter() - t:.1f} s")


def main(report_path: str | None = None) -> int:
    from localization_tool import APP_NAME, __version__

    lines = [f"{APP_NAME} {__version__} self-test",
             f"Python {sys.version.split()[0]} on {platform.platform()}", ""]
    ok = True
    try:
        for name, detail in _checks():
            lines.append(f"OK    {name}: {detail}")
    except Exception:
        ok = False
        lines.append("FAIL")
        lines.append(traceback.format_exc())
    lines.append("")
    lines.append("RESULT: " + ("PASSED" if ok else "FAILED"))
    text = "\n".join(lines)
    if report_path:
        Path(report_path).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if ok else 1
