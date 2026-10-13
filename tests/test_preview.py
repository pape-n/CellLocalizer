import numpy as np

from localization_tool.core.preview import otsu_mask
from localization_tool.core.synthetic import make_synthetic_cell


def test_nucleus_preview_finds_one_component():
    s = make_synthetic_cell()
    mask, thr = otsu_mask(s.data[2], s.voxel_um, 0.15)
    assert thr > 0
    vol = mask.sum() * np.prod(s.voxel_um)
    # synthetic nucleus: ellipsoid 2.0 x 1.7 x 1.0 µm semi-axes, partly below plane 0 -> roughly 8 to 15 µm³
    assert 5 < vol < 20


def test_empty_volume():
    mask, thr = otsu_mask(np.zeros((4, 10, 10), np.uint16), (0.1, 0.1, 0.1))
    assert not mask.any()
