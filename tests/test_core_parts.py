"""Unit tests for distances, segmentation helpers, notes, parameters and the raw-data guarantee."""

import numpy as np
import pytest

from localization_tool.core import distances, qc, segmentation
from localization_tool.core.analysis import analyze_cell
from localization_tool.core.channels import ChannelAssignment
from localization_tool.core.crop import CropBox, crop
from localization_tool.core.params import AnalysisParams, MaskParams
from localization_tool.core.reader import z_orientation
from localization_tool.core.synthetic import make_golden_cell

VOX = (0.13, 0.05, 0.05)
ASSIGN = [ChannelAssignment("Target", "T", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
          ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]


def _sphere(radius, voxel=VOX, shape=(50, 120, 120)):
    nz, ny, nx = shape
    z = (np.arange(nz) - nz / 2 + 0.5)[:, None, None] * voxel[0]
    y = (np.arange(ny) - ny / 2 + 0.5)[None, :, None] * voxel[1]
    x = (np.arange(nx) - nx / 2 + 0.5)[None, None, :] * voxel[2]
    r = np.sqrt(z ** 2 + y ** 2 + x ** 2)
    return r <= radius, r


def test_signed_distance_sphere_accuracy():
    # the voxel grid limits the accuracy: at most about 3/4 of the axial voxel size,
    # typically below the lateral voxel size (METRICS §5)
    mask, r = _sphere(2.0)
    d = distances.signed_distance(mask, VOX)
    near = np.abs(r - 2.0) < 1.0
    err = np.abs(d[near] - (r[near] - 2.0))
    assert err.max() <= 0.75 * max(VOX)
    assert np.median(err) < min(VOX)


def test_signed_distance_empty_mask_is_nan():
    d = distances.signed_distance(np.zeros((3, 4, 4), bool), VOX)
    assert np.isnan(d).all()


def test_depth_below_surface_sphere():
    mask, r = _sphere(2.5)
    depth = distances.depth_below_surface(mask, VOX)
    inside = mask & (r < 2.4)
    err = np.abs(depth[inside] - (2.5 - r[inside]))
    assert err.max() <= 0.75 * max(VOX)
    assert np.median(err) < min(VOX)
    assert np.isnan(depth[~mask]).all()


def test_height_above_substrate_both_orientations():
    cell = np.zeros((10, 4, 4), bool)
    cell[2:7] = True
    h = distances.height_above_substrate(cell, 0.2, coverslip_first=True)
    assert h[2, 0, 0] == pytest.approx(0.0)
    assert h[6, 0, 0] == pytest.approx(0.8)
    h2 = distances.height_above_substrate(cell, 0.2, coverslip_first=False)
    assert h2[6, 0, 0] == pytest.approx(0.0)
    assert h2[2, 0, 0] == pytest.approx(0.8)
    assert np.isnan(h[0, 0, 0])


def test_exclusion_polygon_is_voxel_exact_in_source_coordinates():
    # polygon covering source columns 10..14 and rows 5..9 (voxel centres inside)
    poly = [(4.5, 9.5), (4.5, 14.5), (9.5, 14.5), (9.5, 9.5)]
    m = segmentation.exclusion_mask([poly], origin_zyx=(0, 3, 8), shape_zyx=(2, 10, 10))
    # in the crop with origin y=3, x=8: rows 2..6, columns 2..6
    expected = np.zeros((10, 10), bool)
    expected[2:7, 2:7] = True
    assert np.array_equal(m[0], expected)
    assert np.array_equal(m[1], expected)


def test_exclusion_removes_a_neighbour_from_the_cell():
    stack, truth = make_golden_cell()
    data = stack.data.copy()
    nz, ny, nx = stack.shape_zyx
    data[:, 5:30, 2:20, 2:20] = 0
    data[0, 5:30, 2:20, 2:20] = 300       # bright neighbour fragment in a corner, separate from the cell
    data[1, 5:30, 2:20, 2:20] = 300
    from localization_tool.core.image import ImageStack
    s2 = ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))
    r_plain = analyze_cell(s2, ASSIGN)
    assert not r_plain.masks["cell"][10, 10, 10]           # largest component keeps the real cell
    # an exclusion polygon cuts the cap off the cell
    x0 = int(np.argmax(truth["grid_um"][2].ravel() > 3.5))
    poly = [(0, x0), (0, nx), (ny, nx), (ny, x0)]
    r = analyze_cell(s2, ASSIGN, exclusions_yx=[poly])
    assert not r.masks["cell"][:, :, x0 + 1:].any()
    assert r.masks["cell"][:, :, : x0 - 1].any()


def test_raw_data_are_never_changed_and_masks_stay_on_the_crop_grid():
    stack, _ = make_golden_cell()
    box = CropBox(0, stack.shape_zyx[0], 7, stack.shape_zyx[1] - 5, 3, stack.shape_zyx[2] - 9)
    c = crop(stack, box)
    before = c.data.copy()
    r = analyze_cell(c, ASSIGN, AnalysisParams(background={"T": 5.0}))
    assert np.array_equal(c.data, before)
    for m in r.masks.values():
        assert m.shape == c.shape_zyx
    assert r.origin == (0, 7, 3)


def test_targets_are_measured_on_raw_values_not_on_smoothed_ones():
    stack, truth = make_golden_cell()
    a = analyze_cell(stack, ASSIGN, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    p = AnalysisParams(cell=MaskParams(sigma_um=0.5, method="li"))
    b = analyze_cell(stack, ASSIGN, p, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    # with fixed masks the smoothing setting must not change any measured value
    assert a.target_values["T"]["signal_nucleus_pct"] == b.target_values["T"]["signal_nucleus_pct"]
    assert a.target_values["T"]["total_intensity"] == float(stack.data[0][truth["cell"]].sum())


def test_largest_component_and_remove_small():
    m = np.zeros((3, 20, 20), bool)
    m[:, 1:4, 1:4] = True
    m[:, 10:18, 10:18] = True
    big = segmentation.largest_component(m)
    assert big[:, 12, 12].all() and not big[:, 2, 2].any()
    kept = segmentation.remove_small(m, (1.0, 1.0, 1.0), 30.0)
    assert not kept[:, 2, 2].any() and kept[:, 12, 12].all()


def test_small_holes_filled_large_kept():
    m = np.ones((1, 30, 30), bool)
    m[0, 5:7, 5:7] = False          # 4 px hole
    m[0, 12:22, 12:22] = False      # 100 px hole
    out = segmentation.fill_holes_per_plane(m, max_area_px=10)
    assert out[0, 5, 5] and not out[0, 15, 15]


def test_auto_threshold_constant_gives_empty_mask():
    assert segmentation.auto_threshold(np.ones(100), "otsu") == float("inf")


def test_flags_top_truncation_and_saturation():
    stack, truth = make_golden_cell()
    cell = truth["cell"].copy()
    top = np.flatnonzero(cell.reshape(cell.shape[0], -1).any(axis=1))
    mid = (top[0] + top[-1]) // 2
    cut = cell[: mid + 1]
    notes = qc.flags(cut, truth["nucleus"][: mid + 1], {}, None, True, 0.2, 0.001, True)
    assert "stack cut off at the top" in notes
    full = qc.flags(cell, truth["nucleus"], {}, None, True, 0.2, 0.001, True)
    assert "stack cut off at the top" not in full
    raw = np.zeros(cell.shape, np.uint16)
    raw[cell] = 65535
    notes = qc.flags(cell, truth["nucleus"], {"T": raw}, 65535.0, True, 0.2, 0.001, True)
    assert any(n.startswith("saturated voxels in T") for n in notes)


def test_flags_no_nucleus():
    cell = np.zeros((4, 10, 10), bool)
    cell[:, 3:7, 3:7] = True
    assert "no nucleus found" in qc.flags(cell, np.zeros_like(cell), {}, None, True, 0.2, 0.001, True)
    assert qc.flags(np.zeros_like(cell), np.zeros_like(cell), {}, None, True, 0.2, 0.001, True) == ["empty cell mask"]


def test_params_round_trip_and_validation():
    p = AnalysisParams(envelope_um=0.4, background={"TTN": 12.0})
    p.nucleus.factor = 0.9
    q = AnalysisParams.from_dict(p.to_dict())
    assert q == p
    assert AnalysisParams.from_dict({"cell": {"method": "otsu"}}).cell.sigma_um == AnalysisParams().cell.sigma_um
    with pytest.raises(ValueError):
        AnalysisParams.from_dict({"nonsense": 1})
    with pytest.raises(ValueError):
        MaskParams(sigma_um=0.1, method="manual")
    with pytest.raises(ValueError):
        AnalysisParams(axial_scale=0)


def test_z_orientation_from_czi_metadata():
    xml = ("<Information><Image><Dimensions><Z><ZAxisDirection>FromObjectiveToSpecimen</ZAxisDirection>"
           "<Positions><Interval><Start>-2</Start><Increment>0.13</Increment></Interval></Positions>"
           "</Z></Dimensions></Image></Information><Type>Inverted</Type>")
    o = z_orientation(xml)
    assert o["coverslip_first"] is True and "inverted" in o["z_orientation_source"]
    assert z_orientation(xml.replace("0.13", "-0.13"))["coverslip_first"] is False
    assert z_orientation(xml.replace("FromObjectiveToSpecimen", "FromSpecimenToObjective"))["coverslip_first"] is False
    assert z_orientation("<x/>") == {}


def test_z_reference_setting_overrides_metadata():
    stack, truth = make_golden_cell()
    r = analyze_cell(stack, ASSIGN, AnalysisParams(z_reference="last_plane"),
                     masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    assert r.coverslip_first is False
    assert "setting" in r.info["z_orientation"]


def test_analysis_needs_a_target():
    stack, _ = make_golden_cell()
    bad = [ChannelAssignment("Other", "x", "#FFFFFF")] + ASSIGN[1:]
    with pytest.raises(ValueError):
        analyze_cell(stack, bad)
