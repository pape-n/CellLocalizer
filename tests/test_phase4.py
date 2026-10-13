"""Phase 4 measures against synthetic references: puncta (D34), colocalization (D35), polarity (D36),
crosstalk correction (D38) and suggestions from file names (D37)."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from localization_tool.core import coloc, crosstalk, polarity
from localization_tool.core.analysis import analyze_cell
from localization_tool.core.channels import ChannelAssignment
from localization_tool.core.conditions import DEFAULT_VOCABULARY, suggest_condition, suggest_experiment
from localization_tool.core.distances import signed_distance
from localization_tool.core.image import ImageStack
from localization_tool.core.params import AnalysisParams
from localization_tool.core.synthetic import blur_and_noise, make_golden_cell

ASSIGN = [ChannelAssignment("Target", "T", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
          ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]


@pytest.fixture(scope="module")
def golden():
    return make_golden_cell()


# ---------------------------------------------------------------------------- colocalization
def _structures(shape=(30, 60, 60), frac=0.12, seed=0):
    rng = np.random.default_rng(seed)
    field = ndi.gaussian_filter(rng.normal(size=shape), 2.0)
    return field > np.quantile(field, 1 - frac)


def test_coloc_identical_and_independent():
    rng = np.random.default_rng(1)
    s = _structures()
    cell = np.ones(s.shape, bool)
    a = 100.0 * s + rng.normal(10, 3, s.shape)
    c = coloc.colocalization(a, a, cell)
    assert c["pcc"] == pytest.approx(1.0)
    assert c["m1"] > 0.9 and c["m2"] > 0.9
    other = _structures(seed=5)
    b = 100.0 * other + rng.normal(10, 3, s.shape)
    c2 = coloc.colocalization(a, b, cell)
    assert abs(c2["pcc"]) < 0.1


def test_coloc_manders_with_costes_threshold():
    rng = np.random.default_rng(2)
    s = _structures()
    cell = np.ones(s.shape, bool)
    a = 100.0 * s + rng.normal(10, 3, s.shape)
    # b shares half of the structures of a
    half = s.copy()
    half[:, :, :30] = False
    b = 80.0 * half + rng.normal(10, 3, s.shape)
    c = coloc.colocalization(a, b, cell)
    expected_m1 = a[half].sum() / a.sum()      # share of a where b has signal
    expected_m2 = b[half].sum() / b.sum()      # share of b where a has signal (all of b's structure)
    assert c["m1"] == pytest.approx(expected_m1, abs=0.05)
    assert c["m2"] == pytest.approx(expected_m2, abs=0.08)
    assert c["pcc"] > 0.4


def test_coloc_anticorrelated_has_no_costes_threshold():
    rng = np.random.default_rng(3)
    s = _structures()
    a = 100.0 * s + rng.normal(10, 3, s.shape)
    b = 100.0 * ~s + rng.normal(10, 3, s.shape)
    c = coloc.colocalization(a, b, np.ones(s.shape, bool))
    assert c["pcc"] < 0
    assert np.isnan(c["m1"]) and np.isnan(c["m2"])


# ---------------------------------------------------------------------------- polarity
def _cap_centroid(a, x0):
    num = (a ** 2 - x0 ** 2) / 2 - (a ** 4 - x0 ** 4) / (4 * a ** 2)
    den = (a - x0) - (a ** 3 - x0 ** 3) / (3 * a ** 2)
    return num / den


def test_polarity_axis_and_index_analytic(golden):
    stack, truth = golden
    a, b, c = truth["cell_axes_um"]
    front_x = 3.0
    v_cell = truth["volumes_um3"]["cell"]
    v_front = truth["volumes_um3"]["actin"]
    # actin = 100 everywhere in the cell plus 1000 in the cap x > 3 µm
    x_act = 1000 * v_front * _cap_centroid(a, front_x) / (100 * v_cell + 1000 * v_front)
    expected_index = x_act / np.sqrt(a * b)
    p = polarity.polarity(truth["cell"], stack.data[1].astype(float), truth["nucleus"], stack.voxel_um)
    assert p["polarity_axis_deg"] == pytest.approx(0.0, abs=1.0)          # pointing to +x (right)
    assert p["polarity_index"] == pytest.approx(expected_index, rel=0.03)
    assert p["volume_front_pct"] == pytest.approx(50.0, abs=1.0)
    assert p["nucleus_position_on_axis"] < 0                             # nucleus shifted to -x


def test_polarity_even_cortex_has_low_index(golden):
    stack, truth = golden
    actin = np.where(truth["cell"], 100.0, 0.0)
    p = polarity.polarity(truth["cell"], actin, truth["nucleus"], stack.voxel_um)
    assert p["polarity_index"] < 0.01


def test_target_front_share(golden):
    stack, truth = golden
    data = stack.data.copy()
    _, _, x = truth["grid_um"]
    data[0] = np.where(truth["cell"], 100, 0).astype(np.uint16)
    s = ImageStack(data=data, voxel_um=stack.voxel_um, channels=stack.channels, source=None, info=dict(stack.info))
    r = analyze_cell(s, ASSIGN, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    assert r.target_values["T"]["enrichment_front"] == pytest.approx(1.0, abs=0.02)
    data[0] = np.where(truth["cell"] & (x > 0), 100, 0).astype(np.uint16)
    s = ImageStack(data=data, voxel_um=stack.voxel_um, channels=stack.channels, source=None, info=dict(stack.info))
    r = analyze_cell(s, ASSIGN, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    assert r.target_values["T"]["signal_front_pct"] == pytest.approx(100.0, abs=0.5)
    assert r.target_values["T"]["enrichment_front"] == pytest.approx(2.0, abs=0.05)
    assert "polarity" in r.overlays


def test_phase4_switches_off(golden):
    stack, truth = golden
    # "spots" of 0.4.0a1 analysis files is ignored (puncta removed)
    p = AnalysisParams.from_dict({"spots": {"enabled": True}, "colocalization": False, "polarity": False})
    r = analyze_cell(stack, ASSIGN, p, masks={k: truth[k] for k in ("cell", "nucleus", "actin")})
    v = r.target_values["T"]
    assert "pcc_actin" not in v and "enrichment_front" not in v


# ---------------------------------------------------------------------------- crosstalk
def test_crosstalk_estimate_and_correction(golden):
    stack, truth = golden
    src = stack.data[1].astype(float)                      # actin as source dye
    rng = np.random.default_rng(4)
    leak = 0.06 * src + rng.normal(0, 0.5, src.shape)       # single-stained control: only crosstalk
    est = crosstalk.estimate_k(src, leak, truth["cell"])
    assert est["k"] == pytest.approx(0.06, abs=0.003)
    fixed = crosstalk.apply(leak, src, est["k"])
    assert fixed[truth["cell"]].mean() < 0.02 * leak[truth["cell"]].mean() + 0.5


def test_crosstalk_param_applies_to_target(golden):
    stack, truth = golden
    masks = {k: truth[k] for k in ("cell", "nucleus", "actin")}
    data = stack.data.astype(float)
    data[0] = data[0] + 0.1 * data[1]                       # actin leaks into the target channel
    s = ImageStack(data=np.clip(data, 0, 65535).astype(np.uint16), voxel_um=stack.voxel_um,
                   channels=stack.channels, source=None, info=dict(stack.info))
    leaky = analyze_cell(s, ASSIGN, masks=masks).target_values["T"]
    p = AnalysisParams.from_dict({"crosstalk": [{"into": "T", "from": "Actin", "k": 0.1}]})
    fixed = analyze_cell(s, ASSIGN, p, masks=masks)
    clean = analyze_cell(stack, ASSIGN, masks=masks).target_values["T"]
    assert leaky["signal_actin_pct"] > clean["signal_actin_pct"] + 1
    assert fixed.target_values["T"]["signal_actin_pct"] == pytest.approx(clean["signal_actin_pct"], abs=0.2)
    assert any("crosstalk" in n for n in fixed.notes)


def test_crosstalk_param_validation():
    with pytest.raises(ValueError):
        AnalysisParams.from_dict({"crosstalk": [{"into": "T", "from": "T", "k": 0.1}]})
    with pytest.raises(ValueError):
        AnalysisParams.from_dict({"crosstalk": [{"into": "T", "from": "A", "k": 1.5}]})


# ---------------------------------------------------------------------------- file names
def test_suggestions_from_file_names():
    base = "Images"
    p = f"{base}/09.02.2026 Target staining/WT+treated/06_25_2026 Cells WT+treated DAPI-01.czi"
    assert suggest_condition(p, DEFAULT_VOCABULARY) == "WT treated"
    assert suggest_experiment(p) == "09.02.2026 Target staining"      # day and month ambiguous
    p = f"{base}/20260226 - Cells KO Target Phalloidin/Processed images/cell01/26_02_2026_Cells-01.czi"
    assert suggest_experiment(p) == "2026-02-26"
    assert suggest_condition(p, DEFAULT_VOCABULARY) == "KO"
    p = f"{base}/06.25.2026 Target staining stimulated cells/Cell4/x-4.czi"
    assert suggest_experiment(p) == "2026-06-25"
    assert suggest_condition("a/il-8 wt.czi", ["WT", "IL8"]) == "WT IL8"
    assert suggest_condition("a/b.czi", ["WT"]) == ""
