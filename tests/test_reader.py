import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from localization_tool.core.reader import ReaderError, _to_czyx, read_image, write_imagej_tiff
from localization_tool.core.synthetic import make_synthetic_cell


def test_imagej_roundtrip(tmp_path):
    stack = make_synthetic_cell(shape_zyx=(12, 64, 80))
    p = tmp_path / "cell.tif"
    write_imagej_tiff(stack, p)
    back = read_image(p)
    assert back.data.shape == stack.data.shape
    assert np.array_equal(back.data, stack.data)
    assert np.allclose(back.voxel_um, stack.voxel_um, rtol=1e-4)
    assert back.calibrated
    assert [c.name for c in back.channels] == [c.short for c in stack.channels]


def test_uncalibrated_tiff_is_flagged(tmp_path):
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, np.zeros((5, 20, 30), np.uint16))
    s = read_image(p)
    assert s.data.shape == (1, 5, 20, 30)
    assert not s.calibrated


def test_axes_reordering():
    data = np.arange(2 * 3 * 4 * 5).reshape(3, 2, 4, 5)   # Z C Y X
    out = _to_czyx(data, "ZCYX", "x")
    assert out.shape == (2, 3, 4, 5)
    assert out[1, 2, 3, 4] == data[2, 1, 3, 4]


def test_singleton_axes_dropped_and_missing_added():
    out = _to_czyx(np.zeros((1, 1, 6, 7)), "TBYX", "x")
    assert out.shape == (1, 1, 6, 7)


@pytest.mark.parametrize("axes,shape,text", [
    ("HCZYX", (4, 2, 3, 5, 5), "Airyscan raw"),
    ("TCZYX", (3, 2, 3, 5, 5), "time series"),
    ("MCZYX", (2, 2, 3, 5, 5), "tiles or positions"),
])
def test_unsupported_dimensions(axes, shape, text):
    with pytest.raises(ReaderError, match=text):
        _to_czyx(np.zeros(shape), axes, "f.czi")


def test_unknown_suffix(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"0")
    with pytest.raises(ReaderError, match="not supported"):
        read_image(p)


def test_missing_file(tmp_path):
    with pytest.raises(ReaderError, match="not found"):
        read_image(tmp_path / "nope.czi")


def test_broken_czi(tmp_path):
    p = tmp_path / "broken.czi"
    p.write_bytes(b"ZISRAWFILE" + b"\0" * 100)
    with pytest.raises(ReaderError):
        read_image(p)


REAL = os.environ.get("LT_TEST_DATA")


@pytest.mark.realdata
@pytest.mark.skipif(not REAL, reason="LT_TEST_DATA not set")
def test_real_czi_files():
    files = sorted(Path(REAL).rglob("*.czi"))
    assert files
    for f in files:
        try:
            s = read_image(f)
        except ReaderError as exc:
            assert "Airyscan raw" in str(exc) or "incomplete" in str(exc), f"{f.name}: {exc}"
            continue
        assert s.calibrated, f.name
        assert 0.01 < s.voxel_um[1] < 0.5 and 0.05 < s.voxel_um[0] < 2, (f.name, s.voxel_um)
        assert all(c.fluor for c in s.channels), f.name
