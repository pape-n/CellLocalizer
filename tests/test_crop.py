import numpy as np
import pytest

from localization_tool.core.crop import CropBox, box_from_rectangle, crop
from localization_tool.core.synthetic import make_synthetic_cell


def test_crop_copies_values_and_keeps_origin():
    s = make_synthetic_cell(shape_zyx=(10, 60, 70))
    box = CropBox(2, 8, 5, 40, 10, 50)
    c = crop(s, box)
    assert c.origin == (2, 5, 10)
    assert np.array_equal(c.data, s.data[:, 2:8, 5:40, 10:50])
    assert c.translate_um == pytest.approx((2 * s.voxel_um[0], 5 * s.voxel_um[1], 10 * s.voxel_um[2]))
    c.data[0, 0, 0, 0] += 1
    assert s.data[0, 2, 5, 10] != c.data[0, 0, 0, 0], "crop must not share memory with the source"


def test_box_from_rectangle_widens_to_whole_voxels_and_clips():
    box = box_from_rectangle(np.array([[10.4, -3.0], [10.4, 20.2], [30.6, 20.2], [30.6, -3.0]]), (8, 25, 40))
    assert (box.y0, box.y1, box.x0, box.x1) == (10, 25, 0, 21)
    assert (box.z0, box.z1) == (0, 8)


def test_z_range_is_inclusive():
    box = box_from_rectangle(np.array([[0, 0], [5, 5]]), (10, 20, 20), z_range=(2, 4))
    assert (box.z0, box.z1) == (2, 5)


def test_empty_box_rejected():
    with pytest.raises(ValueError):
        CropBox(0, 5, 30, 40, 0, 5).clipped((5, 20, 20))


def test_crop_of_crop_rejected():
    s = make_synthetic_cell(shape_zyx=(6, 40, 40))
    c = crop(s, CropBox(0, 6, 0, 20, 0, 20))
    with pytest.raises(ValueError):
        crop(c, CropBox(0, 6, 0, 10, 0, 10))


def test_border_flag():
    assert CropBox(0, 5, 0, 10, 3, 9).touches_border((5, 20, 20))
    assert not CropBox(0, 5, 1, 10, 3, 9).touches_border((5, 20, 20))


def test_dict_roundtrip():
    b = CropBox(1, 2, 3, 4, 5, 6)
    assert CropBox.from_dict(b.to_dict()) == b
