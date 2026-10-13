"""Cell viewer (D71): figures of planes and sections, recorded values in the TIFF, mask cache."""

from pathlib import Path

import numpy as np
import pytest

from localization_tool.core import batch, cellview, figure, project_file
from localization_tool.core.channels import ChannelAssignment
from localization_tool.core.params import AnalysisParams
from localization_tool.core.reader import write_imagej_tiff
from localization_tool.core.synthetic import make_practice_cell

ROLES = [ChannelAssignment("Target", "TTN", "#FF2BD6"), ChannelAssignment("Target", "VIM", "#FFAA00"),
         ChannelAssignment("Actin", "Actin", "#2BD46B"), ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]


def make_analysis(folder: Path) -> Path:
    """A saved analysis with one accepted cell of the practice image."""
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / "WT_cell.tif"
    write_imagej_tiff(make_practice_cell(), img)
    params = AnalysisParams(sensitivity=False, colocalization=False)
    im = project_file.ImageSpec(path=img, experiment="2026-10-12", condition="WT", assignments=list(ROLES))
    af = project_file.AnalysisFile(images=[im], params=params)
    from localization_tool.core.reader import read_image

    stack = read_image(img)
    spec = project_file.CellSpec(name="cell1", status="accepted")
    c, box, r = batch.analyze_spec(stack, im.assignments, params, spec)
    spec.box = box
    spec.results = batch.cell_results(im, spec, box, r, params)
    im.cells.append(spec)
    return project_file.save(af, folder / "analysis.lta.json")


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    return make_analysis(tmp_path_factory.mktemp("viewer"))


def test_cell_opens_and_masks_are_kept_for_next_time(analysis):
    af = project_file.load(analysis)
    im, spec = cellview.cells_of(af)[0]
    first = cellview.load_cell(af, im, spec, analysis.parent)
    cache = cellview.cache_path(analysis.parent, im, spec)
    assert cache.exists()
    again = cellview.load_cell(af, im, spec, analysis.parent)      # from the kept file
    assert np.array_equal(first.masks["cell"], again.masks["cell"])
    for k in ("cell", "nucleus", "envelope_zone", "cortex_shell", "basal_layer", "target_TTN"):
        assert first.view_masks[k].any()
    assert first.view_masks["nucleus"].sum() + first.view_masks["cytoplasm"].sum() == first.view_masks["cell"].sum()


def test_kept_masks_of_another_result_are_not_used(analysis, tmp_path):
    af = project_file.load(analysis)
    im, spec = cellview.cells_of(af)[0]
    cellview.load_cell(af, im, spec, analysis.parent)
    cache = cellview.cache_path(analysis.parent, im, spec)
    with np.load(cache) as f:
        data = {k: f[k] for k in f.files}
    data["cell"] = np.zeros_like(data["cell"])                       # does not fit the stored fingerprint
    np.savez_compressed(cache, **data)
    cell = cellview.load_cell(af, im, spec, analysis.parent)
    assert cell.masks["cell"].any()


def test_plane_and_section_pictures_and_recorded_values(analysis, tmp_path):
    af = project_file.load(analysis)
    im, spec = cellview.cells_of(af)[0]
    cell = cellview.load_cell(af, im, spec, analysis.parent)
    s = figure.FigureSettings(channels=figure.default_channel_views(cell.stack.data, cell.assignments,
                                                                   cell.view_masks["cell"]),
                              masks=figure.default_mask_views(list(cell.view_masks)), show_legend=True, title="WT")
    s.crop_to_cell = False
    plane = figure.make_view(cell.stack.data, cell.view_masks, cell.voxel_um, s, cell.coverslip_first, cell.default_z)
    assert np.array_equal(plane.raw, cell.stack.data[:, cell.default_z])          # the recorded values
    s.crop_to_cell = True
    rgb = figure.render(plane, s)                     # title above, scale bar below, legend right: in margins
    assert rgb.shape[0] > plane.raw.shape[1] * s.scale and rgb.shape[1] > plane.raw.shape[2] * s.scale
    bare = figure.FigureSettings(channels=s.channels, masks=s.masks, show_scale_bar=False, mode="plane")
    assert figure.render(plane, bare).shape[:2] == (plane.raw.shape[1] * s.scale, plane.raw.shape[2] * s.scale)
    s.mode, s.angle_deg = "section", 90.0
    sec = figure.make_view(cell.stack.data, cell.view_masks, cell.voxel_um, s, cell.coverslip_first)
    assert sec.raw.shape[1] == cell.stack.shape_zyx[0]                            # one row per plane
    assert set(np.unique(sec.raw[3])) <= set(np.unique(cell.stack.data[3]))       # nearest voxel, no new values
    bare.mode = "section"
    pic = figure.render(sec, bare)
    assert pic.shape[0] == pytest.approx(sec.raw.shape[1] * sec.row_stretch * s.scale, abs=s.scale)  # true aspect
    out = figure.save_tiff(sec, [v.label for v in s.channels], tmp_path / "sec.tif")
    import tifffile

    with tifffile.TiffFile(out) as t:
        assert t.series[0].shape == sec.raw.shape and t.imagej_metadata["Labels"][0] == "TTN"
    png = figure.save_png(pic, tmp_path / "sec.png", 300)
    from PIL import Image

    assert Image.open(png).info["dpi"][0] == pytest.approx(300, abs=1)


def test_section_through_the_coverslip_side_is_at_the_bottom():
    data = np.zeros((1, 6, 20, 20), np.uint16)
    data[0, 0] = 100                                      # bright plane at the coverslip
    cell = np.zeros((6, 20, 20), bool)
    cell[:, 5:15, 5:15] = True
    s = figure.FigureSettings(mode="section", channels=[figure.ChannelView("A", "#FFFFFF", high=100)])
    v = figure.make_view(data, {"cell": cell}, (0.13, 0.05, 0.05), s, coverslip_first=True)
    assert (v.raw[0, -1] == 100).any() and not (v.raw[0, 0] == 100).any()


def test_settings_file_round_trip(tmp_path):
    s = figure.FigureSettings(mode="section", angle_deg=30, channels=[figure.ChannelView("TTN", "#FF00FF", low=5)],
                              masks=[figure.MaskView("cell", True, "fill", "#FFFFFF")])
    back = figure.FigureSettings.load(s.save(tmp_path / "f.json"))
    assert back == s
    with pytest.raises(ValueError):
        figure.FigureSettings.from_dict({"mode": "hologram"})
