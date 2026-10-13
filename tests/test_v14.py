"""1.4: noise test from neighbouring voxels (D77)."""

import numpy as np
import pytest

from localization_tool.core import segmentation as seg

VOXEL = (0.1, 0.1, 0.1)


def _crowded_nuclei(contrast: float, noise: float, seed: int = 0) -> np.ndarray:
    """Many dim nuclei filling about half of the image, camera offset 400, white noise."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:30, :80, :80]
    img = np.full(z.shape, 400.0)
    for cy in (20, 60):
        for cx in (20, 60):
            img[((y - cy) ** 2 + (x - cx) ** 2 <= 17 ** 2) & (np.abs(z - 15) <= 10)] += contrast
    return img + rng.normal(0, noise, img.shape)


def test_noise_width_from_neighbours_matches_the_true_noise():
    rng = np.random.default_rng(1)
    raw = 400 + rng.normal(0, 5.0, (40, 80, 80))
    assert seg.smoothed_noise(raw, VOXEL, 0.0) == pytest.approx(5.0, rel=0.05)
    sm = seg.smooth(raw, VOXEL, 0.2)                      # away from the edges, where "nearest" adds variance
    assert seg.smoothed_noise(raw, VOXEL, 0.2) == pytest.approx(float(sm[10:-10, 10:-10, 10:-10].std()), rel=0.1)


def test_visible_stain_filling_much_of_the_image_counts_as_signal():
    raw = _crowded_nuclei(contrast=25, noise=5)          # step 25, noise 5: clearly visible
    sm = seg.smooth(raw, VOXEL, 0.2)
    assert not seg.has_signal(sm)                          # spread of all voxels: rejected (D77)
    assert seg.has_signal(sm, raw, VOXEL, 0.2)


def test_noise_only_stays_without_signal():
    rng = np.random.default_rng(2)
    raw = 400 + rng.normal(0, 5.0, (30, 80, 80))
    assert not seg.has_signal(seg.smooth(raw, VOXEL, 0.2), raw, VOXEL, 0.2)


def test_data_clipped_at_zero_keep_the_old_test():
    rng = np.random.default_rng(3)
    raw = np.maximum(rng.normal(0, 3.0, (30, 80, 80)), 0)       # half of the voxels exactly 0 (Airyscan)
    raw[10:20, 30:50, 30:50] += 6
    sm = seg.smooth(raw, VOXEL, 0.2)
    assert seg.smoothed_noise(raw, VOXEL, 0.2) is None
    assert seg.has_signal(sm, raw, VOXEL, 0.2) == seg.has_signal(sm)


@pytest.mark.parametrize("sigma_z, tol", [(0.15, 0.05), (0.3, 0.05), (0.6, 0.15)])
def test_nucleus_volume_with_stronger_axial_blur(sigma_z, tol):
    """D78: with axial blur in the Airyscan range (Gaussian sigma up to 0.3 µm) the half-maximum nucleus
    keeps its volume within 2 % (test 5 %). Stronger axial blur widens the blurred top and bottom of the
    nucleus: sigma 0.45 µm +4 %, 0.6 µm +10 %, 0.9 µm +26 %."""
    from scipy import ndimage as ndi

    from localization_tool.core.params import AnalysisParams

    rng = np.random.default_rng(0)
    zz, yy, xx = np.mgrid[:100, :90, :90] * 0.1
    cell = (((zz - 4.5) / 4.0) ** 2 + ((yy - 4.5) / 4.2) ** 2 + ((xx - 4.5) / 4.2) ** 2 <= 1) & (zz >= 1.0)
    nuc = ((zz - 4.0) / 2.0) ** 2 + ((yy - 4.5) / 2.6) ** 2 + ((xx - 4.5) / 2.6) ** 2 <= 1
    img = ndi.gaussian_filter(np.where(nuc, 4.0, 1.0) * cell, (sigma_z / 0.1, 0.8, 0.8)) * 100
    img = np.clip(img + rng.normal(0, 15, img.shape), 0, None)
    mask, _ = seg.segment_nucleus(img, VOXEL, AnalysisParams().nucleus, cell)
    assert mask.sum() / nuc.sum() == pytest.approx(1.0, abs=tol)
    assert np.flatnonzero(mask.any((1, 2)))[[0, -1]] == pytest.approx(np.flatnonzero(nuc.any((1, 2)))[[0, -1]], abs=2)


def test_picture_only_around_the_cell_and_true_scale_bar():
    """Viewer feedback 13.10.: the picture shows the cell plus 1 µm, not the whole analysed region, and the
    scale bar is as long as the voxel size of the image says."""
    from localization_tool.core import figure

    data = np.zeros((1, 5, 100, 120), np.uint16)
    cell = np.zeros((5, 100, 120), bool)
    cell[1:4, 40:60, 30:70] = True
    data[0][cell] = 100
    voxel = (0.2, 0.05, 0.05)
    s = figure.FigureSettings(channels=[figure.ChannelView("A", "#FFFFFF", high=100)], scale=3,
                              scale_bar_um=1.0, show_scale_text=False)
    v = figure.make_view(data, {"cell": cell}, voxel, s, True, 2)
    assert v.raw.shape[1:] == (20 + 2 * 20, 40 + 2 * 20)                  # 1 µm = 20 pixels on each side
    assert v.masks["cell"].sum() == cell[2].sum()
    rgb = figure.render(v, s)
    h = v.raw.shape[1] * 3
    bar_rows = rgb[h:, :, 0] == 255                                        # the bar below the picture
    row = bar_rows[bar_rows.any(axis=1)][0]
    assert row.sum() == pytest.approx(1.0 / 0.05 * 3, abs=1)                # 1 µm at 0.05 µm per voxel, 3 px each
    s.mode = "section"
    sec = figure.make_view(data, {"cell": cell}, voxel, s, True)
    assert sec.raw.shape[2] == 40 + 2 * 20 + 1 or sec.raw.shape[2] == 40 + 2 * 20
    s.crop_to_cell = False
    assert figure.make_view(data, {"cell": cell}, voxel, s, True).raw.shape[2] >= 119     # the whole line through the image


def test_hint_for_a_camera_offset_that_is_not_subtracted():
    """D80: images with an offset and background 0 get a hint, clipped images and set backgrounds do not."""
    from localization_tool.core import qc
    from localization_tool.core.analysis import analyze_cell
    from localization_tool.core.channels import ChannelAssignment
    from localization_tool.core.image import ImageStack
    from localization_tool.core.params import AnalysisParams
    from localization_tool.core.synthetic import make_practice_cell

    rng = np.random.default_rng(4)
    cam = 400 + rng.normal(0, 5, (10, 40, 40))
    cam[3:7, 10:30, 10:30] += 300
    assert qc.dark_level(cam) == pytest.approx(400 - 3 * 5, abs=6)
    assert "background not subtracted" in qc.dark_level_note("A", cam, 0.0)
    assert qc.dark_level_note("A", cam, 390.0) is None
    clipped = np.maximum(cam - 420, 0)
    assert qc.dark_level_note("A", clipped, 0.0) is None
    stack = make_practice_cell()
    roles = [ChannelAssignment("Target", "TTN", "#FF2BD6"), ChannelAssignment("Target", "VIM", "#FFAA00"),
             ChannelAssignment("Actin", "Actin", "#2BD46B"), ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]
    params = AnalysisParams(sensitivity=False, colocalization=False)
    assert not any("background not subtracted" in n for n in analyze_cell(stack, roles, params).notes)
    shifted = ImageStack(stack.data.astype(np.uint16) + 400, stack.voxel_um, stack.channels, info=dict(stack.info))
    notes = analyze_cell(shifted, roles, params).notes
    assert sum("background not subtracted" in n for n in notes) == 4
