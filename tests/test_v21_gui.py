"""Version 2.1 in the program: comparison channel in steps 4 and 5 (D83), front arrow (D84), settings for
D86, D88, D89. Needs OpenGL: run with xvfb-run on Linux."""

import os
import sys
import time

import pytest

pytestmark = pytest.mark.gui

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("no display (use xvfb-run)", allow_module_level=True)


def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication

    t = time.time()
    while time.time() - t < seconds:
        QApplication.processEvents()
        time.sleep(0.01)


def _idle(w, seconds=120):
    t = time.time()
    while time.time() - t < seconds:
        _pump(0.05)
        if not w.state.computing and not w.panel._working:
            return
    raise AssertionError("the program did not become idle")


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("LT_CONFIG_DIR", str(tmp_path / "cfg"))
    import napari

    from localization_tool.gui import app as A
    from localization_tool.gui import theme

    viewer = napari.Viewer(show=True)
    viewer.theme = theme.register_napari_theme()
    w = A.MainWindow(viewer)
    w.setup()
    yield w
    viewer.close()


def test_practice_cell_with_a_comparison_channel(window):
    w, p = window, window.panel
    w.load_example()
    _idle(w)
    roles = {a.label: a.role for a in w.state.assignments}
    assert roles["VIM"] == "Comparison"
    p.go(2)
    _idle(w)
    p.go(3)
    _idle(w)
    assert set(p.target_box) == {"TTN", "VIM"}
    assert p.target_box["VIM"].text() == "Comparison mask VIM"
    assert p.factor["comparison"].isEnabled()
    assert "count the target signal inside it" in p.target_hint.text()
    # the comparison mask can be shown like a target mask
    p.target_box["VIM"].setChecked(True)
    assert w._target_layers["VIM"].visible
    # the arrow: drawn in 2D with the width that says how clear the front is
    p.front_box.setChecked(True)
    front = w._layer(w.FRONT_LAYER)
    pol = w.state.result.overlays["polarity"]
    assert front.visible and len(front.data) == 2
    assert front.edge_width[0] == pytest.approx(pol["width"])
    # step 5: the target in the VIM area, rear half, envelope halves, the counting
    p.go(4)
    _pump(0.3)
    names = [p.results.item(r, 0).text().strip() for r in range(p.results.rowCount())]
    assert "In VIM area" in names and "Rear half" in names and "inner half" in names and "outer half" in names
    assert "inside the target mask" in p.result_facts.text()
    v = w.state.result.target_values["TTN"]
    assert "enrichment_in_VIM" in v and "rear_to_front_ratio" in v


def test_settings_dialog_round_trip_of_the_new_values(window):
    from localization_tool.core.params import AnalysisParams
    from localization_tool.gui.settings import SettingsDialog

    p = AnalysisParams(target_signal="all", polarity_from="actin_signal", polarity_clear_index=0.4,
                       envelope_lateral_only=True)
    dlg = SettingsDialog(None, p, ["TTN"], [])
    back, _ = dlg.values()
    assert back.target_signal == "all" and back.polarity_from == "actin_signal"
    assert back.polarity_clear_index == pytest.approx(0.4) and back.envelope_lateral_only
    dlg.target_signal.setCurrentIndex(dlg.target_signal.findData("mask"))
    assert dlg.values()[0].target_signal == "mask"
    assert "comparison" in dlg.mask_w
    dlg.close()


def test_shortcut_for_3d_goes_through_the_program(window):
    """D87: napari's own 2D/3D switch prepares the layers like the 3D button (no contours in 3D)."""
    w, p = window, window.panel
    w.load_example()
    _idle(w)
    p.go(2)
    _idle(w)
    cell = w._layer(w.CELL_LAYER)
    w.viewer.dims.ndisplay = 3                   # as napari's shortcut does
    _pump(0.2)
    assert w.viewer.dims.ndisplay == 3 and cell.contour == 0
    w.viewer.dims.ndisplay = 2
    _pump(0.2)
    assert cell.contour == 1
