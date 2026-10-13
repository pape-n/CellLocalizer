"""Version 1.2: half-maximum cell edge and enrichment accuracy (D54), channel offset (D55),
sensitivity span (D56), acquisition settings (D63)."""

from pathlib import Path

import numpy as np
import pytest

from localization_tool.core import offsets, reader
from localization_tool.core.analysis import _shift
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
SHAPES = {"compact": dict(cell_axes=(4.0, 3.6, 3.0), nucleus_radius=1.8, nucleus_shift_x=-0.5, front_x=3.0),
          "flat": dict(cell_axes=(7.0, 5.5, 1.6), nucleus_radius=1.4, nucleus_shift_x=-2.0, front_x=5.5)}
HALFMAX = AnalysisParams().with_overrides({"cell": {"method": "halfmax"}})
LI = AnalysisParams().with_overrides({"cell": {"method": "li"}})


def _uniform_target(shape: str, blur: bool):
    """Target spread evenly over the cell: the true enrichment is 1 everywhere."""
    stack, truth = make_golden_cell(**SHAPES[shape])
    data = stack.data.copy()
    data[0] = np.where(truth["cell"], 100, 0).astype(data.dtype)
    s = ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))
    return (blur_and_noise(s) if blur else s), truth


@pytest.fixture(scope="module")
def uniform():
    return {(shape, blur): _uniform_target(shape, blur) for shape in SHAPES for blur in (False, True)}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_enrichment_of_an_even_target_is_one_with_the_half_maximum_edge(uniform, shape):
    """Golden test for the enrichment (D54): sharp image within 0.03, Airyscan-like blur and noise
    within 0.1 (the rest is partial volume, about 0.05 even with the true masks)."""
    for blur, tol in ((False, 0.03), (True, 0.10)):
        stack, truth = uniform[(shape, blur)]
        r = analyze_cell(stack, ASSIGN, HALFMAX)
        v = r.target_values["T"]
        assert r.masks["cell"].sum() / truth["cell"].sum() == pytest.approx(1.0, abs=0.04)
        assert v["enrichment_nucleus"] == pytest.approx(1.0, abs=tol)
        assert v["enrichment_cytoplasm"] == pytest.approx(1.0, abs=0.03)


def test_half_maximum_edge_removes_most_of_the_bias_and_the_shape_effect(uniform):
    li = {s: analyze_cell(uniform[(s, True)][0], ASSIGN, LI).target_values["T"]["enrichment_nucleus"] for s in SHAPES}
    hm = {s: analyze_cell(uniform[(s, True)][0], ASSIGN, HALFMAX).target_values["T"]["enrichment_nucleus"]
          for s in SHAPES}
    for s in SHAPES:
        assert abs(hm[s] - 1) < 0.6 * abs(li[s] - 1)          # Li: 1.14 and 1.18 (METRICS §4.10)
    assert abs(hm["flat"] - hm["compact"]) < 0.05


def test_half_maximum_cell_edge_on_the_golden_cell():
    stack, truth = make_golden_cell()
    for s, dice_min, dev_max in ((stack, 0.98, 1.5), (blur_and_noise(stack), 0.97, 3.0)):
        r = analyze_cell(s, ASSIGN, HALFMAX)
        m = r.masks["cell"]
        assert 2 * (m & truth["cell"]).sum() / (m.sum() + truth["cell"].sum()) > dice_min
        dev = max(abs(r.target_values["T"][k] - x) for k, x in truth["expected"].items() if k.endswith("_pct"))
        assert dev < dev_max


# ------------------------------------------------------------------------------- channel offset (D55)
def test_shift_moves_whole_voxels_and_keeps_values():
    v = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    out = _shift(v, (0.26, -0.1, 0.0), (0.13, 0.1, 0.1))        # back by 2 planes and +1 row
    assert out[0, 1, 0] == v[2, 0, 0]
    assert set(np.unique(out)) <= set(np.unique(v))                # no interpolated values
    assert (out[-2:] == out[-3]).all() and (out[:, 0] == out[:, 1]).all()   # border repeated, no 0 inside a cell


def test_offset_correction_restores_the_aligned_result():
    stack, _ = make_golden_cell()
    data = stack.data.copy()
    data[0] = np.roll(data[0], 2, axis=0)                          # target channel 2 planes too high
    data[0][:2] = 0
    shifted = ImageStack(data, stack.voxel_um, stack.channels, info=dict(stack.info))
    ref = analyze_cell(stack, ASSIGN, AnalysisParams(sensitivity=False)).target_values["T"]
    bad = analyze_cell(shifted, ASSIGN, AnalysisParams(sensitivity=False)).target_values["T"]
    fixed = analyze_cell(shifted, ASSIGN, AnalysisParams(sensitivity=False, channel_offsets=[
        {"channel": "T", "dz_um": 2 * stack.voxel_um[0], "dy_um": 0, "dx_um": 0}])).target_values["T"]
    key = "enrichment_basal_layer"
    assert abs(bad[key] - ref[key]) > 0.05
    assert fixed[key] == pytest.approx(ref[key], abs=0.01)


def test_bead_offsets_are_recovered():
    shifts = ((0, 0, 0), (0.15, 0.03, -0.02), (-0.1, 0.0, 0.05))
    data = offsets.synthetic_beads(shifts_um=shifts)
    res = offsets.estimate_offsets(data, (0.13, 0.05, 0.05), 0, ["A", "B", "C"])
    for lab, sh in zip(("B", "C"), shifts[1:]):
        o = res["channels"][lab]
        assert o["beads"] >= 10
        assert o["dz_um"] == pytest.approx(sh[0], abs=0.03)
        assert o["dy_um"] == pytest.approx(sh[1], abs=0.015)
        assert o["dx_um"] == pytest.approx(sh[2], abs=0.015)
    p = AnalysisParams(channel_offsets=offsets.offsets_as_params(res))
    assert {o["channel"] for o in p.channel_offsets} == {"B", "C"}


def test_implausible_offset_is_rejected():
    with pytest.raises(ValueError):
        AnalysisParams(channel_offsets=[{"channel": "T", "dz_um": 9.0, "dy_um": 0, "dx_um": 0}])
    with pytest.raises(ValueError):                                 # lateral: more than the 1 µm crop margin
        AnalysisParams(channel_offsets=[{"channel": "T", "dz_um": 0, "dy_um": 1.5, "dx_um": 0}])


def test_offset_found_by_fluorophore():
    from localization_tool.core.analysis import channel_offset
    from localization_tool.core.image import ChannelInfo

    p = AnalysisParams(channel_offsets=[{"channel": "AF647", "fluor": "Alexa Fluor 647", "dz_um": 0.13,
                                         "dy_um": 0, "dx_um": 0}])
    assert channel_offset(p, "TTN", "alexa fluor 647") == (0.13, 0.0, 0.0)
    assert channel_offset(p, "TTN", "DAPI") is None
    stack, _ = make_golden_cell()
    chans = [ChannelInfo(index=c.index, name=c.name, fluor=("Alexa Fluor 647" if c.index == 0 else c.fluor))
             for c in stack.channels]
    tagged = ImageStack(stack.data, stack.voxel_um, chans, info=dict(stack.info))
    r = analyze_cell(tagged, ASSIGN, AnalysisParams(sensitivity=False, channel_offsets=p.channel_offsets))
    assert [o["channel"] for o in r.info["channel_offsets"]] == [ASSIGN[0].label]


# ------------------------------------------------------------------------------- sensitivity (D56)
def test_sensitivity_span_contains_the_value_and_can_be_switched_off():
    stack, _ = make_golden_cell()
    s = blur_and_noise(stack)
    v = analyze_cell(s, ASSIGN).target_values["T"]
    for key in ("enrichment_nucleus", "signal_nucleus_pct", "enrichment_actin", "signal_cytoplasm_pct"):
        assert v[f"{key}_min"] <= v[key] <= v[f"{key}_max"]
    assert v["enrichment_nucleus_max"] > v["enrichment_nucleus_min"]
    off = analyze_cell(s, ASSIGN, AnalysisParams(sensitivity=False)).target_values["T"]
    assert "enrichment_nucleus_min" not in off


# ------------------------------------------------------------------------------- acquisition (D63)
XML = """<ImageDocument><Metadata><Information>
<Instrument><Objectives><Objective Id="Objective:0" Name="EC Plan-Neofluar 20x/0.50"/>
<Objective Id="Objective:1" Name="Plan-Apochromat 63x/1.40 Oil DIC M27"/></Objectives></Instrument>
<Image><ObjectiveSettings><Medium>Oil</Medium><RefractiveIndex>1.518</RefractiveIndex><ObjectiveRef Id="Objective:1"/></ObjectiveSettings>
<Dimensions><Channels>
<Channel Id="Channel:0" Name="AF647"><Fluor>Alexa Fluor 647</Fluor><DetectorSettings><Voltage>850</Voltage><Offset>0</Offset></DetectorSettings>
<LightSourcesSettings><LightSourceSettings><Attenuation>0.99</Attenuation></LightSourceSettings></LightSourcesSettings></Channel>
<Channel Id="Channel:2" Name="DAPI"><Fluor>DAPI</Fluor></Channel>
</Channels></Dimensions></Image>
<Processing><Airyscan><Channels>
<Channel Id="Channel:2"><SuperResolutionParameter>6.5</SuperResolutionParameter><ParameterEstimationType>Automatic</ParameterEstimationType></Channel>
<Channel Id="Channel:0"><SuperResolutionParameter>8.2</SuperResolutionParameter><ParameterEstimationType>Automatic</ParameterEstimationType></Channel>
</Channels></Airyscan></Processing>
</Information></Metadata></ImageDocument>"""


def test_acquisition_settings_are_read_by_channel_id():
    a = reader.acquisition_settings(XML)
    assert a["immersion"] == "Oil" and a["refractive_index"] == 1.518
    assert a["objective"].startswith("Plan-Apochromat 63x")                        # the referenced one
    c0, c1 = a["channels"]
    assert c0["airyscan_strength"] == 8.2 and c1["airyscan_strength"] == 6.5      # matched by Id, not order
    assert c0["laser_transmission_pct"] == pytest.approx(1.0)
    assert c0["detector_voltage"] == 850
    assert reader.acquisition_settings("not xml") == {}


def test_differences_within_an_experiment_are_reported():
    from localization_tool.core.export_excel import acquisition_notes

    rows = [{"experiment": "E1", "image": "img1", "notes": "", "airyscan_strength_TTN": 8.2, "detector_voltage_TTN": 850},
            {"experiment": "E1", "image": "img2", "notes": "x", "airyscan_strength_TTN": 6.6, "detector_voltage_TTN": 850},
            {"experiment": "E2", "image": "img3", "notes": "", "airyscan_strength_TTN": 9.0}]
    out = acquisition_notes(rows)
    e1 = [r for r in out if r["experiment"] == "E1"]
    assert all("Airyscan strength TTN 6.6 to 8.2" in r["notes"] for r in e1)
    assert not any("detector" in r["notes"] for r in e1)
    assert e1[1]["notes"].startswith("x; ")
    assert next(r for r in out if r["experiment"] == "E2")["notes"] == ""   # one image: nothing to compare
    assert rows[0]["notes"] == ""                                            # the input is not changed



def test_automatic_airyscan_strength_and_equal_file_names():
    from localization_tool.core.export_excel import acquisition_notes

    auto = [{"experiment": "E1", "condition": c, "image": "Image 1.czi", "notes": "", "airyscan_mode_TTN": "Automatic",
             "airyscan_strength_TTN": v, "laser_transmission_pct_TTN": t}
            for c, v, t in (("WT", 8.2, 1.6), ("KO", 6.6, 2.0))]
    out = acquisition_notes(auto)
    assert all("Airyscan" not in r["notes"] for r in out)            # automatic: differs by design
    assert all("laser % TTN 1.6 to 2" in r["notes"] for r in out)    # same file name in two folders still compared


# ------------------------------------------------------------------------------- methods text (D58)
def _analysis_with_results():
    from localization_tool.core import batch, project_file
    from localization_tool.core.crop import CropBox

    stack, _ = make_golden_cell()
    r = analyze_cell(stack, ASSIGN)
    im = project_file.ImageSpec(path=__import__("pathlib").Path("x/img1.czi"), experiment="2026-06-25",
                                condition="WT", assignments=ASSIGN)
    spec = project_file.CellSpec(name="cell1", status="accepted")
    nz, ny, nx = stack.shape_zyx
    spec.results = batch.cell_results(im, spec, CropBox(0, nz, 0, ny, 0, nx), r, AnalysisParams())
    im.cells = [spec, project_file.CellSpec(name="cell2", status="excluded", note="Out of focus")]
    af = project_file.AnalysisFile(images=[im], blind=True)
    af.assign_codes(seed=1)
    return af


@pytest.mark.parametrize("lang", ["en", "de"])
def test_methods_text_is_complete_and_cites_what_it_uses(lang):
    import re

    from localization_tool import __version__
    from localization_tool.core.methods_text import REFERENCES, methods_text, summary_from_analysis

    af = _analysis_with_results()
    text = methods_text(af.params, summary_from_analysis(af), lang)
    assert __version__ in text and "npape2@uni-muenster.de" in text
    cited = {int(n) for group in re.findall(r"\[(\d+(?:, \d+)*)\]", text) for n in group.split(", ")}
    listed = re.findall(r"^(\d+)\. ", text, re.M)
    assert cited == set(range(1, len(listed) + 1))                # every citation is listed, nothing else
    assert all(any(r in line for r in REFERENCES.values()) for line in text.splitlines() if re.match(r"^\d+\. ", line))
    assert "Out of focus (1)" in text
    assert ("blinded" in text) if lang == "en" else ("ohne Kenntnis" in text)
    # defaults only: counts are placeholders
    plain = methods_text(AnalysisParams(), None, lang)
    assert "[n]" in plain and "{" not in plain


def test_methods_text_follows_the_settings():
    from localization_tool.core.methods_text import methods_text

    p = AnalysisParams(crosstalk=[{"into": "VIM", "from": "Actin", "k": 0.06}],
                       channel_offsets=[{"channel": "T", "dz_um": 0.13, "dy_um": 0, "dx_um": 0}])
    t = methods_text(p.with_overrides({"cell": {"method": "halfmax"}}), None, "en")
    assert "k = 0.06" in t and "multicolour beads" in t and "half maximum" in t
    assert "No correction for chromatic offset" not in t
    assert "No correction for chromatic offset" in methods_text(AnalysisParams(), None, "en")


def test_blind_codes_are_random_stable_and_hide_names():
    from pathlib import Path

    from localization_tool.core import project_file

    af = project_file.AnalysisFile(images=[project_file.ImageSpec(path=Path(f"d/{c}.czi"), condition=c)
                                           for c in ("WT1", "WT2", "TC1", "TC2", "TC3")], blind=True)
    af.assign_codes(seed=3)
    codes = [im.code for im in af.images]
    assert len(set(codes)) == 5 and all(c.startswith("Image ") for c in codes)
    assert codes != sorted(codes)                                   # not in folder order
    assert af.shown_name(af.images[0]) == codes[0]
    af.images[0].revealed = True
    assert af.shown_name(af.images[0]) == "WT1.czi"
    again = project_file.AnalysisFile.from_dict(af.to_dict())
    assert [im.code for im in again.images] == codes and again.blind and again.images[0].revealed
    af.images.append(project_file.ImageSpec(path=Path("d/new.czi")))
    af.assign_codes()
    assert [im.code for im in af.images[:5]] == codes and af.images[5].code not in codes


# ------------------------------------------------------------------------------- review of 1.2
def test_bead_channels_are_matched_by_fluorophore_then_position():
    chans = [("TTN", "Alexa Fluor 647", "Target"), ("T-PMT", "", "Ignore"), ("Actin", "Alexa Fluor 488", "Actin"),
             ("Nucleus", "DAPI", "Nucleus")]
    labs, how = offsets.match_channels(["DAPI", "Alexa Fluor 647", "Alexa Fluor 488"], chans)
    assert labs == ["Nucleus", "TTN", "Actin"] and how == "by fluorophore"
    labs, how = offsets.match_channels(["A", "B", "C", "D"], chans)
    assert labs == ["TTN", "T-PMT", "Actin", "Nucleus"] and how.startswith("by position")
    labs, how = offsets.match_channels(["A", "B"], chans)
    assert labs is None and "A, B" in how
    assert offsets.match_channels(["DAPI"], [])[0] == ["DAPI"]


def test_blind_codes_stay_unique_past_99_images():
    from localization_tool.core.project_file import AnalysisFile, ImageSpec, code_number

    af = AnalysisFile(images=[ImageSpec(path=Path(f"/x/{i}.czi")) for i in range(95)], blind=True)
    af.assign_codes(1)
    af.images += [ImageSpec(path=Path(f"/y/{i}.czi")) for i in range(10)]
    af.assign_codes(2)
    nums = [code_number(im.code) for im in af.images]
    assert len(set(nums)) == 105 and max(nums) == 105
    order = af.blind_order()
    assert [code_number(af.images[k].code) for k in order] == sorted(nums)


def test_accepted_cell_keeps_status_when_masks_do_not_change(tmp_path):
    from localization_tool.core import batch, project_file
    from localization_tool.core.reader import write_imagej_tiff

    stack, _ = make_golden_cell()
    path = tmp_path / "cell.tif"
    write_imagej_tiff(stack, path)
    af = project_file.AnalysisFile(params=AnalysisParams(sensitivity=False), panel=ASSIGN)
    af.images.append(project_file.ImageSpec(path=path))
    out = [e for e in batch.iter_cells(af, [(0, None)], png=False) if e["kind"] == "cell"]
    spec = out[0]["spec"]
    spec.status = "accepted"
    af.images[0].cells.append(spec)
    af.params = AnalysisParams(sensitivity=True)               # only adds columns
    again = [e for e in batch.iter_cells(af, [(0, spec.name)], png=False) if e["kind"] == "cell"][0]["spec"]
    assert again.status == "accepted" and again.results["rows"][0]["status"] == "accepted"
    assert "enrichment_nucleus_min" in again.results["rows"][0]
    af.images[0].cells[0] = again
    af.params = AnalysisParams(sensitivity=True, nucleus=MaskParams(sigma_um=0.2, method="halfmax", factor=1.4))
    third = [e for e in batch.iter_cells(af, [(0, spec.name)], png=False) if e["kind"] == "cell"][0]["spec"]
    assert third.status == "automatic"                          # other nucleus: check again
    # a cell from 1.1: no fingerprint, same settings as 1.1 knew them -> kept
    old = dict(again.results)
    old.pop("mask_digest")
    old["results_version"] = 3
    old["params"] = {k: v for k, v in old["params"].items() if k not in ("channel_offsets", "sensitivity")}
    assert batch.masks_unchanged(old, again.results)
    old["params"]["cortex_um"] = 0.8
    assert not batch.masks_unchanged(old, again.results)


def test_halfmax_edge_does_not_cut_dim_cytoplasm_inside_the_cell(monkeypatch):
    """Granular cells (review of 1.2): cytoplasm between bright granules and near the coverslip is
    dimmer than half the rim level in every channel, but above Li. The half maximum may trim the blur
    shell (0.5 µm), not the cell interior. Here the lower half of one side of the cell is dim."""
    from scipy import ndimage as ndi

    from localization_tool.core import segmentation

    stack, truth = make_golden_cell(nucleus_radius=1.2, nucleus_shift_x=-2.5)
    v = stack.voxel_um
    nz, nx = stack.data.shape[1], stack.data.shape[3]
    z = np.arange(nz)[:, None, None] * v[0]
    x = (np.arange(nx)[None, None, :] - nx / 2) * v[2]
    data = stack.data.astype(np.float32).copy()
    dim = truth["cell"] & (z <= 1.2) & (x > 0.0)
    data[:, dim] *= 0.4
    blurred = blur_and_noise(ImageStack(data, v, stack.channels, info=dict(stack.info)))
    chans = list(blurred.data.astype(np.float32))
    deep = dim & (ndi.distance_transform_edt(truth["cell"], sampling=v) > 0.6)
    mask, _ = segmentation.segment_cell(chans, v, HALFMAX.cell)
    assert mask[deep].mean() > 0.95
    assert 0.95 < mask.sum() / truth["cell"].sum() < 1.06
    monkeypatch.setattr(segmentation, "EDGE_TRIM_UM", 1e9)         # without the limit: interior cut
    unbounded, _ = segmentation.segment_cell(chans, v, HALFMAX.cell)
    assert unbounded[deep].mean() < 0.8


def test_methods_text_claims_equal_settings_only_from_metadata():
    """Review of 1.2: without laser and detector values in the files (TIFF) the text asks for them
    instead of stating that they were identical; a 50 nm pixel stays 50 nm."""
    from localization_tool.core.methods_text import Summary, methods_text

    s = Summary(cells=3, images=3, voxel_um=(0.05, 0.05, 0.13))
    t = methods_text(None, s, "en")
    assert "identical within each experiment" not in t and "not in the image metadata" in t
    assert "pixel size of 50 nm" in t
    s.acquisition_read = True
    assert "identical within each experiment" in methods_text(None, s, "en")
    assert "Pixelgröße von 50 nm" in methods_text(None, s, "de")


def test_practice_cell_has_no_bottom_plane_note():
    """The practice cell starts above empty planes, like a well-recorded stack (D67)."""
    from localization_tool.core.analysis import analyze_cell
    from localization_tool.core.channels import ChannelAssignment
    from localization_tool.core.synthetic import make_practice_cell

    roles = [("Target", "TTN"), ("Target", "VIM"), ("Actin", "Actin"), ("Nucleus", "Nucleus")]
    r = analyze_cell(make_practice_cell(), [ChannelAssignment(a, b, "#FFFFFF") for a, b in roles], AnalysisParams())
    assert r.cell_values["bottom_plane_ratio"] < 0.6


def _flat_cell_with_bright_base():
    """Flat cell: a bright actin layer at the coverslip (planes 0 to 2), a four times dimmer cortex ring
    in the planes above, dim cytoplasm inside."""
    nz, ny, nx = 16, 120, 120
    z, y, x = np.ogrid[:nz, :ny, :nx]
    r = np.sqrt((y - 60) ** 2 + (x - 60) ** 2) * 0.05
    cell = (r <= 2.5) & (z < 14)
    cortex = cell & (r > 2.2) & (z >= 3)
    actin = np.where(cell, 200.0, 5.0)
    actin[cortex] = 1500.0
    actin[:3][cell[:3]] = 6000.0
    return np.asarray(actin, np.float32), cell, cortex


def test_actin_per_plane_finds_the_dim_cortex_above_a_bright_base():
    """D70: one threshold for the stack takes only the bright base, the per-plane threshold also takes the
    cortex of the upper planes; the base stays actin."""
    from localization_tool.core import segmentation

    actin, cell, cortex = _flat_cell_with_bright_base()
    vox, mp = (0.13, 0.05, 0.05), MaskParams(sigma_um=0.0, method="li")
    one, _ = segmentation.segment_actin(actin, vox, mp, cell)
    both, thr, per_plane = segmentation.segment_actin_planes(actin, vox, mp, cell)
    assert one[cortex].mean() < 0.5 < 0.95 < both[cortex].mean()
    assert both[:3][cell[:3]].mean() > 0.99                    # the bright layer at the coverslip stays actin
    inner = cell & ~cortex
    inner[:3] = False
    assert both[inner].mean() < 0.02                           # the dim cytoplasm does not become actin
    assert (per_plane <= thr + 1e-6).all()


def test_saved_analysis_keeps_one_actin_threshold():
    """An analysis saved before 1.2.1 keeps its actin rule and its cells are not offered for recomputing."""
    from localization_tool.core import batch, project_file

    old = AnalysisParams(actin_per_plane=False).to_dict()
    old.pop("actin_per_plane")
    af = project_file.AnalysisFile.from_dict({"format": "Localization Tool analysis", "version": 1, "params": old,
                                              "images": []})
    assert af.params.actin_per_plane is False and AnalysisParams().actin_per_plane is True
    spec = project_file.CellSpec(name="cell1", status="accepted",
                                 results={"rows": [{}], "params": old, "results_version": batch.RESULTS_VERSION})
    assert not batch.needs_update(spec, af.params)
    assert batch.needs_update(spec, AnalysisParams())
