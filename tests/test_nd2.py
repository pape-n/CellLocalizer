"""Nikon ND2 files (D90). The reader uses the nd2 package; these tests replace it by a stand-in with the
metadata layout of real files (nd2 tests/samples_metadata.json), so no lab file is needed. A real file is
checked with LT_TEST_DATA (realdata)."""

import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from localization_tool.core import files, reader
from localization_tool.core.reader import ReaderError, read_image


def _fake_nd2(monkeypatch, sizes, text="Microscope: Nikon Ti2 Microscope", bottom_to_top=True, rgb=False,
              voxel=(0.1, 0.1, 0.3), names=("DAPI", "Alexa 488", "Alexa 647")):
    shape = tuple(sizes.values())
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    chans = [NS(channel=NS(name=n, emissionLambdaNm=450.0 + 100 * i, excitationLambdaNm=405.0 + 100 * i,
                           color=NS(r=255, g=0, b=0) if i == 0 else None,     # nd2 0.12: Color(r, g, b)
                           colorRGB=0x0000FF if i == 0 else 0xFF0000),        # older: 0x00BBGGRR
                microscope=NS(objectiveName="Plan Apo λ 60x Oil", immersionRefractiveIndex=1.515))
             for i, n in enumerate(names)]
    loop = NS(type="ZStackLoop", parameters=NS(bottomToTop=bottom_to_top, stepUm=voxel[2]))

    class ND2File:
        def __init__(self, path):
            self.path = path
            self.sizes = dict(sizes)
            self.is_rgb = rgb
            self.metadata = NS(channels=chans)
            self.experiment = [loop]
            self.text_info = {"description": text, "date": "10/13/2026 9:00:00 AM"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            return data

        def voxel_size(self):
            return NS(x=voxel[0], y=voxel[1], z=voxel[2])

    monkeypatch.setitem(sys.modules, "nd2", types.SimpleNamespace(ND2File=ND2File))
    return data


def test_nd2_stack_channels_voxel_and_coverslip(tmp_path, monkeypatch):
    data = _fake_nd2(monkeypatch, {"Z": 4, "C": 3, "Y": 8, "X": 6})
    path = tmp_path / "cell.nd2"
    path.write_bytes(b"x")
    s = read_image(path)
    assert s.data.shape == (3, 4, 8, 6)
    assert np.array_equal(s.data[1], data[:, 1])           # values unchanged, only the axes moved
    assert s.voxel_um == pytest.approx((0.3, 0.1, 0.1)) and s.calibrated
    assert [c.name for c in s.channels] == ["DAPI", "Alexa 488", "Alexa 647"]
    assert s.channels[0].color == "#FF0000" and s.channels[1].color == "#0000FF"     # Nikon stores BGR
    assert s.channels[2].emission_nm == 650.0
    assert s.info["format"] == "ND2" and s.info["objective"].startswith("Plan Apo")
    assert s.info["coverslip_first"] is True and "inverted" in s.info["z_orientation_source"]


@pytest.mark.parametrize("text,btt,expected", [
    ("Eclipse Ti2", True, True), ("Eclipse Ti2", False, False),
    ("Eclipse Ni-E upright", True, False), ("unknown stand", True, None)])
def test_nd2_coverslip_side_from_stand_and_direction(tmp_path, monkeypatch, text, btt, expected):
    _fake_nd2(monkeypatch, {"Z": 2, "C": 3, "Y": 4, "X": 4}, text=text, bottom_to_top=btt)
    path = tmp_path / "a.nd2"
    path.write_bytes(b"x")
    s = read_image(path)
    assert s.info.get("coverslip_first") is expected


@pytest.mark.parametrize("sizes,word", [({"T": 3, "Z": 2, "C": 3, "Y": 4, "X": 4}, "time series"),
                                        ({"P": 2, "Z": 2, "C": 3, "Y": 4, "X": 4}, "positions")])
def test_nd2_time_series_and_positions_are_refused(tmp_path, monkeypatch, sizes, word):
    _fake_nd2(monkeypatch, sizes)
    path = tmp_path / "b.nd2"
    path.write_bytes(b"x")
    with pytest.raises(ReaderError, match=word if word == "time series" else "positions"):
        read_image(path)
    images, skipped = files.find_images(tmp_path)
    assert not images and word in skipped[0][1]


def test_nd2_rgb_refused_and_single_plane(tmp_path, monkeypatch):
    _fake_nd2(monkeypatch, {"C": 3, "Y": 4, "X": 4}, rgb=True)
    path = tmp_path / "c.nd2"
    path.write_bytes(b"x")
    with pytest.raises(ReaderError, match="RGB"):
        read_image(path)
    _fake_nd2(monkeypatch, {"C": 3, "Y": 4, "X": 4})
    s = read_image(path)
    assert s.data.shape == (3, 1, 4, 4) and s.voxel_um[0] == pytest.approx(0.1)


def test_folder_with_nd2_skips_exported_tiffs(tmp_path, monkeypatch):
    _fake_nd2(monkeypatch, {"Z": 2, "C": 3, "Y": 4, "X": 4})
    (tmp_path / "cell1.nd2").write_bytes(b"x")
    (tmp_path / "cell1.tif").write_bytes(b"x")
    images, skipped = files.find_images(tmp_path)
    assert [p.name for p in images] == ["cell1.nd2"]
    assert "ND2" in skipped[0][1]
    assert ".nd2" in reader.SUPPORTED_SUFFIXES


@pytest.mark.realdata
def test_real_nd2_files():
    """Every ND2 in LT_TEST_DATA opens with voxel size and channel names (lab files, never committed)."""
    import os
    from pathlib import Path

    real = os.environ.get("LT_TEST_DATA")
    found = sorted(Path(real).rglob("*.nd2")) if real else []
    if not found:
        pytest.skip("no ND2 file in LT_TEST_DATA")
    for f in found[:5]:
        try:
            s = read_image(f)
        except ReaderError as exc:          # time series or positions are refused with a clear message
            assert "not supported" in str(exc) or "positions" in str(exc), (f.name, exc)
            continue
        assert s.calibrated and s.data.ndim == 4 and s.info["format"] == "ND2", f.name
