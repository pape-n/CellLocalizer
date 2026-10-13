"""Version 1.1: nucleus edge at half maximum (D51), coverslip gap open (D52), no target mask on an
unstained channel (D53)."""

import numpy as np
import pytest

from localization_tool.core import segmentation
from localization_tool.core.channels import ChannelAssignment
from localization_tool.core.image import ImageStack
from localization_tool.core.params import AnalysisParams, MaskParams
from localization_tool.core.synthetic import blur_and_noise, make_golden_cell

from localization_tool.core import analysis as _analysis
from localization_tool.core.params import AnalysisParams as _Params


def analyze_cell(stack, assignments, params=None, **kw):
    """These tests check analytic values of the whole target signal (D30). Since D89 that is the
    setting target_signal "all"; counting inside the target mask is tested in test_v21.py."""
    p = params or _Params()
    if p.target_signal != "all":
        p = p.with_overrides({"target_signal": "all"})
    return _analysis.analyze_cell(stack, assignments, p, **kw)


ASSIGN = [ChannelAssignment("Target", "T", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
          ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]


def _replace_channel(stack, index, values):
    data = stack.data.copy()
    data[index] = np.asarray(values).astype(data.dtype)
    return ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))


def test_default_nucleus_threshold_is_half_maximum():
    p = AnalysisParams()
    assert p.nucleus.method == "halfmax"
    assert p.nucleus.sigma_um == pytest.approx(0.2)
    assert p.nucleus_dim_hole_fraction == 0.0               # optional rule, off (review before 1.1)


@pytest.mark.parametrize("cell_axes", [(5.0, 4.0, 3.0), (3.0, 2.7, 2.4)])
def test_half_maximum_does_not_depend_on_cell_size(cell_axes):
    """The same nucleus (DAPI 800) in a large and in a compact cell: the threshold sits at half the
    nuclear level and the volume is right in both. Otsu depends on the share of nucleus voxels."""
    stack, truth = make_golden_cell(cell_axes=cell_axes, nucleus_radius=2.0, nucleus_shift_x=0.0,
                                    front_x=cell_axes[0] - 0.8)
    blurred = blur_and_noise(stack)
    r = analyze_cell(blurred, ASSIGN)
    assert r.thresholds["nucleus"] == pytest.approx(400, rel=0.1)
    assert r.masks["nucleus"].sum() / truth["nucleus"].sum() == pytest.approx(1.0, abs=0.04)


def test_halfmax_threshold_of_a_plain_step():
    """Two levels, 100 outside and 900 inside: the threshold lies at 500."""
    sm = np.full((10, 40, 40), 100.0, np.float32)
    sm[2:8, 10:30, 10:30] = 900.0
    cell = np.ones(sm.shape, bool)
    thr = segmentation.halfmax_threshold(sm, cell, (0.13, 0.1, 0.1))
    assert thr == pytest.approx(500.0, abs=1.0)


def test_dim_chromatin_hole_is_filled():
    """A column of dim but DAPI-positive chromatin through the nucleus (an open tunnel in 3D, larger
    than the small-hole rule) becomes nucleus with the optional rule: its DAPI reaches half the
    threshold (D51)."""
    stack, truth = make_golden_cell()
    z, y, x = truth["grid_um"]
    zc, _, xs = truth["nucleus_centre_um"]
    column = (((x - xs) ** 2 + y ** 2) <= 0.8 ** 2) & truth["nucleus"]
    inner = column & (np.abs(z - zc) < 1.5)  # where the column ends at the nucleus surface it is open
    dapi = stack.data[2].astype(float)
    dapi[column] = 240.0                     # well below the threshold (about 400), above half of it
    s = _replace_channel(stack, 2, dapi)
    filled = analyze_cell(s, ASSIGN, AnalysisParams(nucleus_dim_hole_fraction=0.5)).masks["nucleus"]
    assert filled[inner].mean() > 0.98
    off = analyze_cell(s, ASSIGN, AnalysisParams(nucleus_dim_hole_fraction=0.0)).masks["nucleus"]
    assert off[inner].mean() < 0.6


def test_dim_hole_rule_keeps_a_dark_hole_open():
    mask = np.zeros((3, 30, 30), bool)
    mask[:, 5:25, 5:25] = True
    mask[:, 10:20, 10:20] = False            # 100 px hole
    sm = np.where(mask, 1000.0, 0.0).astype(np.float32)
    sm[:, 10:20, 10:20] = 100.0
    assert not segmentation.fill_dim_holes(mask, sm, 200.0)[:, 15, 15].any()
    assert segmentation.fill_dim_holes(mask, sm, 50.0)[:, 15, 15].all()


def test_ring_centre_stays_open_by_default():
    """A small DAPI-free lumen in a blurred ring nucleus: the default keeps it cytoplasm. The optional
    dim-hole rule judges the mean per plane and can fill it, which is why it is off."""
    stack, truth = make_golden_cell(ring_nucleus=True, nucleus_radius=1.4)
    blurred = blur_and_noise(stack)
    z, y, x = truth["grid_um"]
    zc, _, xs = truth["nucleus_centre_um"]
    lumen = (((x - xs) ** 2 + y ** 2) <= 0.3 ** 2) & (np.abs(z - zc) < 0.2) & ~truth["nucleus"]
    assert lumen.any()
    r = analyze_cell(blurred, ASSIGN)
    assert r.masks["nucleus"][lumen].mean() < 0.5


def test_dapi_with_noise_only_gives_no_nucleus():
    stack, _ = make_golden_cell()
    noise = np.random.default_rng(2).poisson(3.0, size=stack.data[2].shape)
    r = analyze_cell(_replace_channel(stack, 2, noise), ASSIGN)
    assert not r.masks["nucleus"].any()
    assert "no nucleus found" in r.notes
    assert np.isnan(r.target_values["T"]["signal_nucleus_pct"])


def test_coverslip_orientation_unknown_or_numpy_bool():
    """Unknown orientation closes both ends, a numpy boolean works like a Python one (D52)."""
    mask = _gap_block()
    voxel = (0.13, 0.1, 0.1)
    unknown = segmentation._fill_enclosed(mask, voxel, 0.2, coverslip_first=None)
    assert unknown[0:3, 12, 16].all()                       # unknown orientation: both ends closed
    assert not segmentation._fill_enclosed(mask, voxel, 0.2, coverslip_first=np.bool_(True))[0:3, 12, 16].any()


def _gap_block():
    """Nucleus block with a cavity that opens only onto plane 0 (gap between two lobes on the glass)."""
    mask = np.zeros((6, 24, 34), bool)
    mask[:, 2:22, 2:32] = True
    mask[0:3, 6:18, 10:22] = False
    return mask


def test_gap_open_to_the_coverslip_stays_open():
    voxel = (0.13, 0.1, 0.1)
    mask = _gap_block()
    out = segmentation._fill_enclosed(mask, voxel, 0.2, coverslip_first=True)
    assert not out[0:3, 12, 16].any()
    # the same cavity at the far end of the stack (stack cut inside the nucleus) is filled
    out = segmentation._fill_enclosed(mask, voxel, 0.2, coverslip_first=False)
    assert out[0:3, 12, 16].all()


def test_unstained_target_has_no_mask():
    """Secondary antibody control: a target channel with noise only gets no target mask and empty
    mask values, the main values stay (D53)."""
    stack, _ = make_golden_cell()
    rng = np.random.default_rng(1)
    noise = rng.poisson(3.0, size=stack.data[0].shape)
    r = analyze_cell(_replace_channel(stack, 0, noise), ASSIGN)
    assert not r.masks["target_T"].any()
    v = r.target_values["T"]
    assert np.isnan(v["target_mask_volume_pct"]) and np.isnan(v["target_mask_signal_pct"])
    assert np.isfinite(v["signal_nucleus_pct"])
    assert any("no staining above noise" in n for n in r.notes)


def test_stained_target_still_has_a_mask():
    stack, _ = make_golden_cell()
    r = analyze_cell(stack, ASSIGN)
    assert r.masks["target_T"].any()
    assert np.isfinite(r.target_values["T"]["target_mask_volume_pct"])


def test_half_maximum_only_for_the_nucleus():
    with pytest.raises(ValueError):
        AnalysisParams(actin=MaskParams(sigma_um=0.1, method="halfmax"))
    with pytest.raises(ValueError):
        AnalysisParams(nucleus_dim_hole_fraction=1.5)
    old = AnalysisParams.from_dict({"nucleus": {"method": "otsu", "sigma_um": 0.3}})
    assert old.nucleus.method == "otsu"                      # older analyses keep their settings
    assert old.nucleus.sigma_um == 0.3
    assert old.nucleus_dim_hole_fraction == 0.0              # and get no new fill rule
