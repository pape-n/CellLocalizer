"""Golden tests: synthetic cells with analytically known compartments (METRICS §9, ROADMAP phase 1).

Tolerances (percentage points of the signal share):
- given the true masks: 0.2 pp (only voxelisation of the analytic shapes)
- automatic masks, sharp image: 1.5 pp
- automatic masks, blurred (Airyscan-like PSF) and noisy image: 3 pp. Blur moves signal across
  compartment borders, so even the true masks deviate by about 1 pp (partial volume effect).
"""

import numpy as np
import pytest

from localization_tool.core.channels import ChannelAssignment
from localization_tool.core.crop import CropBox, crop
from localization_tool.core.params import AnalysisParams
from localization_tool.core.synthetic import GOLDEN_DENSITY, blur_and_noise, make_golden_cell

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


def _true_masks(truth):
    return {k: truth[k] for k in ("cell", "nucleus", "actin")}


def _max_dev(values, expected):
    return max(abs(values[k] - v) for k, v in expected.items() if k.endswith("_pct"))


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


@pytest.fixture(scope="module")
def golden():
    return make_golden_cell()


@pytest.fixture(scope="module")
def golden_blurred(golden):
    return blur_and_noise(golden[0])


def test_true_masks_reproduce_analytic_shares(golden):
    stack, truth = golden
    r = analyze_cell(stack, ASSIGN, masks=_true_masks(truth))
    v = r.target_values["T"]
    assert _max_dev(v, truth["expected"]) < 0.2
    for k in ("nucleus", "actin", "cytoplasm"):
        assert v[f"enrichment_{k}"] == pytest.approx(truth["expected"][f"enrichment_{k}"], rel=0.02)
    total = v["signal_nucleus_pct"] + v["signal_cytoplasm_pct"]          # D39: nucleus + cytoplasm = 100
    assert total == pytest.approx(100.0, abs=1e-9)
    parts = v["signal_actin_pct"] + v["signal_cytoplasm_without_actin_pct"]
    assert parts == pytest.approx(v["signal_cytoplasm_pct"], abs=1e-9)
    assert r.cell_values["cell_volume_um3"] == pytest.approx(truth["volumes_um3"]["cell"], rel=0.01)
    assert r.cell_values["nucleus_volume_um3"] == pytest.approx(truth["volumes_um3"]["nucleus"], rel=0.01)


def test_automatic_masks_sharp_image(golden):
    stack, truth = golden
    r = analyze_cell(stack, ASSIGN)
    assert _dice(r.masks["cell"], truth["cell"]) > 0.94
    assert _dice(r.masks["nucleus"], truth["nucleus"]) > 0.98
    assert _dice(r.masks["actin"], truth["actin"]) > 0.94
    assert _max_dev(r.target_values["T"], truth["expected"]) < 1.5


def test_automatic_masks_blurred_noisy_image(golden, golden_blurred):
    _, truth = golden
    r = analyze_cell(golden_blurred, ASSIGN)
    assert _dice(r.masks["cell"], truth["cell"]) > 0.94
    assert _dice(r.masks["nucleus"], truth["nucleus"]) > 0.97
    assert _max_dev(r.target_values["T"], truth["expected"]) < 3.0
    v = r.target_values["T"]
    # the ranking of the enrichment must survive blur and noise
    assert v["enrichment_actin"] > v["enrichment_nucleus"] > v["enrichment_cytoplasm_without_actin"]


def test_ring_nucleus_keeps_cytoplasm_in_its_centre():
    stack, truth = make_golden_cell(ring_nucleus=True)
    r = analyze_cell(stack, ASSIGN)
    z, y, x = truth["grid_um"]
    zc, _, xs = truth["nucleus_centre_um"]
    iz = int(np.argmin(np.abs(z.ravel() - zc)))
    iy = int(np.argmin(np.abs(y.ravel())))
    ix = int(np.argmin(np.abs(x.ravel() - xs)))
    assert truth["labels"][iz, iy, ix] == 3                 # the hole is cytoplasm by construction
    assert r.masks["labels"][iz, iy, ix] == 3               # and must not be filled
    # smoothing of 0.3 µm (D40) rounds the thin synthetic ring a little
    assert _dice(r.masks["nucleus"], truth["nucleus"]) > 0.94
    assert _max_dev(r.target_values["T"], truth["expected"]) < 1.5


def test_result_does_not_depend_on_crop_margin():
    # crops leaving 1.2 to 2.9 µm around the cell; the threshold is recomputed on the cell's
    # bounding box plus 1 µm, so the size of the drawn rectangle must not matter
    stack = blur_and_noise(make_golden_cell(margin_um=3.0)[0])
    nz, ny, nx = stack.shape_zyx
    results = []
    for m in (2, 20, 36):
        box = CropBox(0, nz, m, ny - m, m, nx - m)
        r = analyze_cell(crop(stack, box), ASSIGN)
        results.append(r.target_values["T"])
    for k in ("signal_nucleus_pct", "signal_actin_pct", "signal_cytoplasm_pct", "signal_basal_layer_pct"):
        vals = [v[k] for v in results]
        assert max(vals) - min(vals) < 0.5, k


def test_envelope_zone_matches_analytic_shells(golden):
    stack, truth = golden
    p = AnalysisParams()
    r = analyze_cell(stack, ASSIGN, p, masks=_true_masks(truth))
    R, w = truth["nucleus_radius_um"], p.envelope_um
    v_in = 4 / 3 * np.pi * (R ** 3 - (R - w) ** 3)
    v_out = 4 / 3 * np.pi * ((R + w) ** 3 - R ** 3)
    vols = truth["volumes_um3"]
    total = sum(GOLDEN_DENSITY[k] * vols[k] for k in ("nucleus", "actin", "cytoplasm"))
    expected = 100 * (GOLDEN_DENSITY["nucleus"] * v_in + GOLDEN_DENSITY["cytoplasm"] * v_out) / total
    interior = 100 * GOLDEN_DENSITY["nucleus"] * 4 / 3 * np.pi * (R - w) ** 3 / total
    v = r.target_values["T"]
    assert v["signal_envelope_zone_pct"] == pytest.approx(expected, abs=1.0)
    assert v["signal_nucleus_interior_pct"] == pytest.approx(interior, abs=1.0)


def test_cortex_shell_and_basal_layer_volumes(golden):
    stack, truth = golden
    p = AnalysisParams()
    r = analyze_cell(stack, ASSIGN, p, masks=_true_masks(truth))
    a, b, c = truth["cell_axes_um"]
    t = p.cortex_um
    shell = 1 - (a - t) * (b - t) * (c - t) / (a * b * c)        # inner ellipsoid approximation
    assert r.target_values["T"]["volume_cortex_shell_pct"] == pytest.approx(100 * shell, abs=1.5)
    # basal layer: cap of the ellipsoid up to h_basal above the lowest voxel plane of the cell
    dz = stack.voxel_um[0]
    z = truth["grid_um"][0].ravel()
    base = z[np.flatnonzero(truth["cell"].reshape(len(z), -1).any(axis=1))[0]]
    zc = truth["cell_centre_z_um"]
    top = base + p.basal_um + dz / 2                             # plane centres up to h_basal, half a plane above
    hc = top - (zc - c)
    cap = np.pi * a * b * hc ** 2 * (3 * c - hc) / (3 * c ** 2)
    assert r.target_values["T"]["volume_basal_layer_pct"] == pytest.approx(100 * cap / truth["volumes_um3"]["cell"], abs=1.0)


def test_shell_definition_of_actin_rich():
    stack, truth = make_golden_cell()
    p = AnalysisParams(actin_definition="shell")
    r = analyze_cell(stack, ASSIGN, p, masks={"cell": truth["cell"], "nucleus": truth["nucleus"]})
    v = r.target_values["T"]
    assert v["volume_actin_pct"] > 0
    # actin-rich (shell) and cortex shell are the same region, minus the nucleus
    assert v["volume_actin_pct"] <= v["volume_cortex_shell_pct"] + 1e-9


def test_background_offset_is_removed(golden):
    stack, truth = golden
    shifted = stack.data.astype(np.int32)
    shifted[0] += 37                                   # a constant offset on the target only
    from localization_tool.core.image import ImageStack
    s2 = ImageStack(shifted.astype(np.uint16), stack.voxel_um, stack.channels, info=dict(stack.info))
    p = AnalysisParams(background={"T": 37.0})
    a = analyze_cell(stack, ASSIGN, masks=_true_masks(truth)).target_values["T"]
    b = analyze_cell(s2, ASSIGN, p, masks=_true_masks(truth)).target_values["T"]
    for k in ("signal_nucleus_pct", "signal_actin_pct", "enrichment_cytoplasm"):
        assert b[k] == pytest.approx(a[k], abs=1e-9)


def test_axial_scale_changes_volumes_not_shares(golden):
    stack, truth = golden
    a = analyze_cell(stack, ASSIGN, masks=_true_masks(truth))
    b = analyze_cell(stack, ASSIGN, AnalysisParams(axial_scale=0.8), masks=_true_masks(truth))
    assert b.cell_values["cell_volume_um3"] == pytest.approx(0.8 * a.cell_values["cell_volume_um3"])
    assert b.cell_values["cell_height_um"] == pytest.approx(0.8 * a.cell_values["cell_height_um"])
    assert b.target_values["T"]["signal_nucleus_pct"] == pytest.approx(a.target_values["T"]["signal_nucleus_pct"])


def test_profiles_are_normalised(golden):
    stack, truth = golden
    r = analyze_cell(stack, ASSIGN, masks=_true_masks(truth))
    for rows in r.profiles["T"].values():
        assert rows
        assert sum(x["signal_fraction"] for x in rows) == pytest.approx(1.0)
        assert sum(x["volume_fraction"] for x in rows) == pytest.approx(1.0)


def test_dim_chromatin_open_only_to_the_stack_end_is_filled():
    """A dim region inside the nucleus that reaches the top of a truncated stack is nucleus,
    not cytoplasm (seen in real cells with stacks ending inside the nucleus)."""
    from localization_tool.core.image import ImageStack

    stack, truth = make_golden_cell()
    z, y, x = truth["grid_um"]
    zc, _, xs = truth["nucleus_centre_um"]
    dim = (((x - xs) ** 2 + y ** 2) <= 0.6 ** 2) & (z >= zc - 0.5) & truth["nucleus"]
    data = stack.data.copy()
    data[2][dim] = 0
    top = int(np.argmin(np.abs(z.ravel() - (zc + 0.8))))
    cut = ImageStack(data[:, : top + 1], stack.voxel_um, stack.channels, info=dict(stack.info))
    r = analyze_cell(cut, ASSIGN)
    dim_cut = dim[: top + 1]
    assert r.masks["nucleus"][dim_cut].mean() > 0.95
    assert "nucleus cut off at the top" in r.notes


def test_cell_edge_follows_any_stained_channel():
    """A thin protrusion stained only in the actin channel belongs to the cell (D28)."""
    from localization_tool.core.image import ImageStack

    stack, truth = make_golden_cell(margin_um=2.0)
    z, y, x = truth["grid_um"]
    a, b, c = truth["cell_axes_um"]
    zc = truth["cell_centre_z_um"]
    protrusion = (np.abs(y) < 0.4) & (x > a - 0.2) & (x < a + 1.5) & (np.abs(z - zc) < 0.3)   # at the equator
    protrusion &= ~truth["cell"]
    data = stack.data.copy()
    data[1][protrusion] = 300                    # actin only, weaker than the actin front
    s2 = ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))
    r = analyze_cell(s2, ASSIGN)
    assert r.masks["cell"][protrusion].mean() > 0.8


def test_actin_rich_lies_inside_actin_and_is_reported():
    stack, truth = make_golden_cell()
    r = analyze_cell(stack, ASSIGN)
    assert r.masks["actin_rich"].any() and not (r.masks["actin_rich"] & ~r.masks["actin"]).any()
    v = r.target_values["T"]
    assert 0 < v["volume_actin_rich_pct"] <= v["volume_actin_pct"] + v["volume_nucleus_pct"]
    assert v["enrichment_actin_rich"] > 1


def test_empty_noise_channel_does_not_inflate_the_cell():
    """A channel with only noise (e.g. a secondary-antibody control) must not grow the cell mask."""
    from localization_tool.core.image import ImageStack

    stack, truth = make_golden_cell()
    rng = np.random.default_rng(4)
    data = stack.data.copy()
    data[0] = rng.poisson(3.0, size=data[0].shape).astype(np.uint16)     # target channel: noise only
    s2 = ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))
    r = analyze_cell(s2, ASSIGN)
    assert r.masks["cell"].sum() < 1.3 * truth["cell"].sum()
    assert _dice(r.masks["cell"], truth["cell"]) > 0.85


def test_cell_mask_alone_matches_the_full_analysis():
    from localization_tool.core.analysis import cell_mask

    stack, _ = make_golden_cell()
    m, thr = cell_mask(stack, ASSIGN)
    r = analyze_cell(stack, ASSIGN)
    assert np.array_equal(m, r.masks["cell"])
    assert set(thr) == {"cell_T", "cell_Actin", "cell_Nucleus"}


def test_target_mask_marks_clear_target_signal():
    """The golden target has densities 5 (actin cap), 2 (nucleus) and 1 (rest of the cytoplasm).
    The target mask (Li, D40) marks the clearly stained regions and leaves the dimmest one out."""
    stack, truth = make_golden_cell()
    r = analyze_cell(stack, ASSIGN, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    t = r.masks["target_T"]
    rest = truth["labels"] == 3
    assert (t & truth["actin"]).sum() / truth["actin"].sum() > 0.99
    assert (t & truth["nucleus"]).sum() / truth["nucleus"].sum() > 0.7
    assert (t & rest).sum() / rest.sum() < 0.05
    v = r.target_values["T"]
    assert v["target_mask_signal_pct"] > v["signal_actin_pct"]
    # the main shares do not depend on the target mask
    p = AnalysisParams.from_dict({"target": {"factor": 0.5}})
    r2 = analyze_cell(stack, ASSIGN, p, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    assert r2.target_values["T"]["signal_nucleus_pct"] == v["signal_nucleus_pct"]
