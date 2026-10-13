"""D87: display state and user choices survive moving around the program.

Rule (D87): every display setting and every user choice stays as the user left it until the user changes it,
across step changes in any order, cell switches, recomputation, 'Apply settings to all cells', saving and
the cell viewer; and what is shown always belongs to the current data.

Assumption on scope: display settings (mask checkboxes, target masks, front arrow, outlines, 'Only' menu,
channel visibility, 2D/3D) are global for the session; factors, exclusions and status belong to one cell
(reset by 'Next cell', restored by 'Open cell'); slice and zoom belong to one image (a new image may reset them).

Each test drives the real main window with the synthetic practice cell. Run with
    xvfb-run -a -s "-screen 0 1600x1000x24" python -m pytest -q tests/test_d87_state.py
The real-data test uses LT_D87_CZI (default: the first CZI in LT_TEST_DATA) and is skipped without one.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gui

if not os.environ.get("LT_D87"):
    # long and memory hungry (about 25 and 20 min, each test best in its own process):
    #   LT_D87=1 xvfb-run -a pytest -q tests/test_d87_state.py -k <name>
    pytest.skip("D87 audit tests: set LT_D87=1", allow_module_level=True)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("no display (use xvfb-run)", allow_module_level=True)

# exploration only: LT_D87_IGNORE="text|text" skips mismatches containing these texts in the random walk,
# LT_D87_EXPLORE=1 skips the front arrow error in 3D (known defect 1) so the walk can go on
IGNORE = [x for x in os.environ.get("LT_D87_IGNORE", "").split("|") if x]

def _first_czi() -> str:
    folder = os.environ.get("LT_TEST_DATA")
    found = sorted(Path(folder).rglob("*.czi")) if folder else []
    return str(found[0]) if found else "/no/real/file.czi"


REAL_CZI = Path(os.environ.get("LT_D87_CZI") or _first_czi())     # lab files are never committed


# ---------------------------------------------------------------------------- helpers
def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication

    t = time.time()
    while time.time() - t < seconds:
        QApplication.processEvents()
        time.sleep(0.01)


def _idle(w, seconds=180):
    """Wait until no debounce timer, calculation, surface job or save is running."""
    p = w.panel
    t = time.time()
    _pump(0.05)
    while time.time() - t < seconds:
        busy = (w.state.computing or p._working or p._debounce.isActive() or p._cell_debounce.isActive()
                or w._batch is not None or w._saving or w.state.full is None and w._after_load is not None)
        if not busy:
            _pump(0.05)
            return
        _pump(0.05)
    raise AssertionError("the program did not become idle")


def _wait_surfaces(w, seconds=120):
    t = time.time()
    while time.time() - t < seconds:
        r = w.state.result
        if r is not None and (w._surface_key is not None and w._surface_key[0] is r and w._surface_key[1] == w.state.box) and not w.panel._working:
            return
        _pump(0.1)
    raise AssertionError("3D surfaces were not computed")


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import napari

    from localization_tool.gui import app as A
    from localization_tool.gui import theme

    viewer = napari.Viewer(show=True)
    viewer.theme = theme.register_napari_theme()
    w = A.MainWindow(viewer)
    w.setup()
    w.messages = []
    monkeypatch.setattr(w, "message", lambda text, *a, **k: w.messages.append(text))
    monkeypatch.setattr(w, "ask", lambda *a, **k: False)
    yield w
    _idle(w)
    viewer.close()


def _open_example_at_step4(w):
    w.load_example()
    w.panel.go(2)
    _idle(w)
    w.panel.go(3)
    _idle(w)
    assert w.panel.step == 3 and w.state.result is not None


def _checkboxes(layout):
    out = []
    for k in range(layout.count()):
        item = layout.itemAt(k)
        if item.widget() is not None and hasattr(item.widget(), "setChecked"):
            out.append(item.widget())
        elif item.layout() is not None:
            out += _checkboxes(item.layout())
    return out


def _channel_box(w, name, layout=None):
    layout = layout or w.panel.comp_channels
    return next(b for b in _checkboxes(layout) if b.text() == name)


def _alpha(layer, value):
    return float(layer.colormap.color_dict[value][3])


def display_problems(w) -> list[str]:
    """Everything shown must follow the panel and the current data. Returns a list of mismatches."""
    p, s, out = w.panel, w.state, []
    if s.full is None:
        return out
    nd = w.viewer.dims.ndisplay
    three_d = nd == 3
    step = p.step
    lab, cell, front = w._layer(w.LABEL_LAYER), w._layer(w.CELL_LAYER), w._layer(w.FRONT_LAYER)
    ex = w._layer(w.EXCLUDE_LAYER)
    current = [i for i, b in enumerate(p.header.buttons) if b.property("step") == "current"]
    if current != [step]:
        out.append(f"step bar shows step {current}, panel is in step {step}")
    if p.view_2d.isChecked() != (nd == 2) or p.view_3d.isChecked() != (nd == 3):
        out.append("2D/3D buttons do not match the view")
    if ex is not None and ex.visible != (nd == 2):
        out.append(f"drawn exclusions visible={ex.visible} in {nd}D")
    vis = {layer.name: layer.visible for layer in w.image_layers()}
    row = {1: p.display_box, 2: p.cell_channels, 3: p.comp_channels}.get(step)    # only the page on screen
    for b in _checkboxes(row) if row is not None else []:
        if b.text() in vis and b.isChecked() != vis[b.text()]:
            out.append(f"step {step + 1} channel box {b.text()} checked={b.isChecked()}, "
                       f"layer visible={vis[b.text()]}")
    if s.computing or p._working:
        return out
    surfaces_on = [n for n, l in w._surfaces.items() if l in w.viewer.layers and l.visible]
    targets_on = [n for n, l in w._target_layers.items() if l.visible]
    if step == 3 and s.result is not None:
        v = p.visible_compartments()
        b = s.box
        expected = s.result.masks["labels"].astype(np.uint8)
        expected[(expected == 2) & s.result.masks["actin_rich"]] = 4
        if not np.array_equal(lab.data[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1], expected):
            out.append("compartment layer does not show the current result")
        for label, tl in w._target_layers.items():
            t = s.result.masks.get(f"target_{label}")
            if t is not None and not np.array_equal(tl.data[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] > 0, t):
                out.append(f"target mask {label} does not show the current result")
        if not three_d:
            if not lab.visible:
                out.append("compartment layer hidden in step 4 (2D)")
            want = {1: v["nucleus"], 3: v["cytoplasm"], 2: v["actin"] or v["cytoplasm"],
                    4: v["actin_rich"] or v["actin"] or v["cytoplasm"]}
            for value, on in want.items():
                if (_alpha(lab, value) > 0) != on:
                    out.append(f"label {value} shown={_alpha(lab, value) > 0}, checkboxes say {on}")
            if lab.contour != (1 if p.outline_box.isChecked() else 0):
                out.append(f"compartment contour {lab.contour}, outlines box {p.outline_box.isChecked()}")
            if cell.visible != v["cell"]:
                out.append(f"cell edge visible={cell.visible}, box {v['cell']}")
            for label, tl in w._target_layers.items():
                if tl.visible != v.get(f"target_{label}", False):
                    out.append(f"target mask {label} visible={tl.visible}, box {v.get(f'target_{label}')}")
            want_front = v["front"] and len(front.data) > 0
            if front.visible != want_front:
                out.append(f"front arrow visible={front.visible}, expected {want_front}")
            if surfaces_on:
                out.append(f"3D surfaces visible in 2D: {surfaces_on}")
        else:
            if lab.visible or cell.visible or targets_on or front.visible:
                out.append(f"voxel masks visible in 3D: labels={lab.visible} cell={cell.visible} "
                           f"targets={targets_on} front={front.visible}")
            if (w._surface_key is not None and w._surface_key[0] is s.result and w._surface_key[1] == b):
                for name, layer in w._surfaces.items():
                    if layer.visible != v.get(name, False):
                        out.append(f"3D surface {name} visible={layer.visible}, box {v.get(name, False)}")
    elif step == 2 and s.cell is not None:
        b = s.box
        if not cell.visible:
            out.append("cell edge hidden in step 3")
        if not np.array_equal(cell.data[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] > 0, s.cell):
            out.append("cell edge layer does not show the current cell")
        if lab.visible or targets_on or front.visible or surfaces_on:
            out.append("step 4 masks visible in step 3")
        if not three_d and (cell.contour != 1 or abs(cell.opacity - 0.9) > 1e-6):
            out.append(f"cell edge in 2D drawn with contour={cell.contour}, opacity={cell.opacity}")
    elif step in (0, 1, 4):
        if lab.visible or cell.visible or targets_on or front.visible or surfaces_on:
            out.append(f"masks visible in step {step + 1}: labels={lab.visible} cell={cell.visible} "
                       f"targets={targets_on} front={front.visible} surfaces={surfaces_on}")
    return out


def shown_choices(w) -> dict:
    p = w.panel
    return {
        "show": {k: cb.isChecked() for k, cb in p.show_box.items()},
        "targets": {k: cb.isChecked() for k, cb in p.target_box.items()},
        "front": p.front_box.isChecked(),
        "outline": p.outline_box.isChecked(),
        "channels": {layer.name: layer.visible for layer in w.image_layers()},
        "ndisplay": w.viewer.dims.ndisplay,
        "factors": {k: round(v, 4) for k, v in p.factors().items()},
    }


# ---------------------------------------------------------------------------- targeted tests
def test_cytoplasm_off_survives_results_and_back_2d(window):
    """The reported case in 2D: cytoplasm off, step 5, back to step 4 (step bar and Back/Next)."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.show_box["cytoplasm"].setChecked(False)
    p.header.buttons[4].click()
    _idle(w)
    p.header.buttons[3].click()
    _idle(w)
    assert not p.show_box["cytoplasm"].isChecked() and display_problems(w) == []
    p.back_btn.click()
    _idle(w)
    p.next_btn.click()
    _idle(w)
    assert not p.show_box["cytoplasm"].isChecked() and display_problems(w) == []


def test_cytoplasm_off_survives_results_and_back_3d(window):
    """The reported case in 3D: going back to step 4 in 3D raises in show_compartments (front arrow added to
    a 2D shapes layer while the view is 3D), enter_step stops half way and shows the voxel cell and
    compartment volumes instead of the chosen surfaces; the step bar still marks step 5."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    assert w.state.result.overlays.get("polarity")          # the practice cell has a front arrow
    p.show_box["cytoplasm"].setChecked(False)
    p.show_box["cell"].setChecked(False)
    w.set_ndisplay(3)
    _wait_surfaces(w)
    assert display_problems(w) == []
    p.header.buttons[4].click()
    _idle(w)
    p.go(3)                                                 # what the step bar click runs
    _idle(w)
    assert not p.show_box["cytoplasm"].isChecked()
    assert display_problems(w) == []


def test_factor_change_in_3d_recomputes_without_error(window):
    """A factor changed in 3D: the recomputed result is shown with show_compartments, same error."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    w.set_ndisplay(3)
    _wait_surfaces(w)
    before = w.state.result
    p.factor["nucleus"].setValue(0.9)
    w.compute(sync=True)
    assert w.state.result is not before
    _idle(w)
    assert display_problems(w) == []


def test_channel_visibility_survives_edit_of_another_channel(window):
    """A hidden channel stays hidden when the colour (or name, role) of another channel changes in step 2."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    _channel_box(w, "VIM").setChecked(False)
    assert not w.image_layers()[1].visible
    p.go(1)
    combo = p.channel_table.cellWidget(3, 3)                # colour of the nucleus channel
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    _pump()
    assert not w.image_layers()[1].visible, "hidden channel VIM shown again"


def test_leaving_a_name_field_without_change_keeps_everything(window):
    """Clicking into a channel name in step 2 and out again (no edit) emits editingFinished: the result is
    dropped, hidden channels are shown again and the target mask boxes are unticked."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    _channel_box(w, "VIM").setChecked(False)
    p.target_box["TTN"].setChecked(True)
    result = w.state.result
    p.go(1)
    p.channel_table.cellWidget(0, 2).editingFinished.emit()      # focus left the name field, text unchanged
    _pump()
    assert w.state.result is result, "result dropped without any change"
    assert not w.image_layers()[1].visible and p.target_box["TTN"].isChecked()


def test_target_mask_choice_survives_edit_of_another_channel(window):
    """Target mask boxes are rebuilt unticked by set_targets on every channel table edit."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.target_box["TTN"].setChecked(True)
    p.go(1)
    combo = p.channel_table.cellWidget(3, 3)
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    _pump()
    p.go(3)
    _idle(w)
    assert p.target_box["TTN"].isChecked(), "target mask TTN unticked although the user ticked it"
    assert display_problems(w) == []


def test_only_menu_matches_the_checkboxes(window):
    """'Only nucleus' stays in the menu after the user ticks cytoplasm again; 'Only target mask' stays in the
    menu after the target boxes were rebuilt unticked."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.only_menu.setCurrentIndex(1)
    p.only_menu.activated.emit(1)
    p.show_box["cytoplasm"].setChecked(True)
    shown = {k for k, cb in p.show_box.items() if cb.isChecked()}
    assert not (p.only_menu.currentText() == "Only nucleus" and shown != {"nucleus"}), \
        f"menu says '{p.only_menu.currentText()}' but {sorted(shown)} are shown"


def test_step2_display_boxes_follow_channel_visibility(window):
    """A channel hidden in step 4 is still ticked in the 'Display' list of step 2."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    _channel_box(w, "TTN").setChecked(False)
    p.go(1)
    assert display_problems(w) == []


def test_brightness_slider_survives_channel_edit(window):
    """The brightness slider of step 2 is rebuilt at 100 % after any channel edit, the contrast stays changed."""
    w, p = window, window.panel
    w.load_example()
    p.go(1)
    slider = p.display_box.itemAt(0).layout().itemAt(1).widget()
    slider.setValue(200)
    layer = w.image_layers()[0]
    limits = tuple(layer.contrast_limits)
    combo = p.channel_table.cellWidget(3, 3)
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
    _pump()
    slider = p.display_box.itemAt(0).layout().itemAt(1).widget()
    assert tuple(layer.contrast_limits) == pytest.approx(limits)
    assert slider.value() == 200, f"slider shows {slider.value()} %, the channel is still shown at 200 %"


def test_draw_after_3d_shows_shapes_and_cell_outline(window):
    """'Draw' in step 3 after viewing in 3D: the view goes to 2D, but the drawn shapes stay hidden and the cell
    edge stays a filled volume at 35 % (set_ndisplay is bypassed)."""
    w, p = window, window.panel
    w.load_example()
    p.go(2)
    _idle(w)
    w.set_ndisplay(3)
    w.start_exclusion()
    assert w.viewer.dims.ndisplay == 2
    assert w._layer(w.EXCLUDE_LAYER).visible, "the shapes being drawn are invisible"
    assert display_problems(w) == []
    w.apply_exclusions()


def test_cell_outline_after_3d_and_2d_in_another_step(window):
    """3D in step 3, 2D in step 2, back to step 3: the cell edge stays filled at 35 % in 2D."""
    w, p = window, window.panel
    w.load_example()
    p.go(2)
    _idle(w)
    w.set_ndisplay(3)
    p.go(1)
    w.set_ndisplay(2)
    p.go(2)
    _idle(w)
    assert display_problems(w) == []


def test_zoom_and_plane_survive_2d_3d_2d(window):
    w, p = window, window.panel
    w.load_example()
    p.go(2)
    _idle(w)
    w.viewer.camera.zoom = w.viewer.camera.zoom * 2
    w.viewer.dims.set_current_step(0, 10)
    zoom, center = w.viewer.camera.zoom, tuple(w.viewer.camera.center)
    w.set_ndisplay(3)
    w.set_ndisplay(2)
    assert w.viewer.dims.current_step[0] == 10
    assert w.viewer.camera.zoom == pytest.approx(zoom), "the 2D zoom is lost after a look in 3D"
    assert tuple(w.viewer.camera.center)[-2:] == pytest.approx(center[-2:])


def test_zoom_and_plane_survive_step_changes(window):
    w, p = window, window.panel
    _open_example_at_step4(w)
    w.viewer.camera.zoom = w.viewer.camera.zoom * 1.7
    w.viewer.dims.set_current_step(0, 12)
    zoom = w.viewer.camera.zoom
    for k in (4, 3, 2, 1, 0, 3):
        p.go(k)
        _idle(w)
        assert w.viewer.camera.zoom == pytest.approx(zoom) and w.viewer.dims.current_step[0] == 12, k


def test_settings_ok_keeps_the_factors_of_the_open_cell(window):
    """Settings > OK (even without a change) resets the factors the user set for the open cell."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.factor["nucleus"].setValue(0.8)
    _idle(w)
    w._set_params(w.params, ask=True)                        # what open_settings does after OK
    _idle(w)
    assert p.factor["nucleus"].value() == pytest.approx(0.8), "factor 0.8 of the open cell reset"


def test_use_these_factors_for_all_cells_keeps_the_shown_value(window):
    """'Use these factors for all cells' while an accepted cell with its own factor is open: the box jumps back
    to the factor stored with the cell, although the user just chose another value for all cells."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.factor["nucleus"].setValue(0.8)
    _idle(w)
    p.go(4)
    w.accept_cell()
    p.go(3)
    _idle(w)
    p.factor["nucleus"].setValue(0.9)
    _idle(w)
    w.factors_as_default()
    _idle(w)
    assert w.params.nucleus.factor == pytest.approx(0.9)
    assert p.factor["nucleus"].value() == pytest.approx(0.9), f"box shows {p.factor['nucleus'].value()}"


def test_accepted_cell_still_says_accepted_after_step_changes(window):
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.go(4)
    w.accept_cell()
    assert p.accept_btn.text() == "Accepted" and not p.accept_btn.isEnabled()
    p.go(3)
    _idle(w)
    p.go(4)
    assert w.state.accepted_current and not p.accept_btn.isEnabled()
    assert p.accept_btn.text() == "Accepted", f"a disabled button reads '{p.accept_btn.text()}'"


def test_settings_in_step5_leave_no_stale_numbers(window):
    """Settings > OK while step 5 is open: the result is dropped (state.result = None), the table still shows
    the old numbers and 'Accept cell' then says that no image is open."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.go(4)
    assert p.accept_btn.isEnabled()
    w._set_params(w.params, ask=True)
    _idle(w)
    if w.state.result is None:
        assert p.step != 4 or p.results.rowCount() == 0, "step 5 shows numbers of a result that was dropped"
    w.accept_cell()
    assert not any("No image" in m for m in w.messages), w.messages


def test_no_masks_of_the_previous_cell_while_the_next_one_is_computed(window):
    """'Next cell in this image', then step 4: until the new result is there, the compartment layer is shown
    with the masks of the previous cell."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    old = w._layer(w.LABEL_LAYER).data.copy()
    p.go(4)
    w.accept_cell()
    w.next_cell()
    _idle(w)
    w.start_exclusion()                                     # another cell: a different region
    w._layer(w.EXCLUDE_LAYER).add_polygons(np.array([[0, 0], [0, 80], [80, 0]], float))
    w.apply_exclusions()
    _idle(w)
    p.go(3)                                                 # the compartments are computed in a worker
    lab = w._layer(w.LABEL_LAYER)
    stale = w.state.computing and lab.visible and np.array_equal(lab.data, old)
    _idle(w)
    assert not stale, "step 4 showed the compartments of the previous cell while computing the new one"


def test_save_and_cell_viewer_change_nothing(window, tmp_path):
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.show_box["actin"].setChecked(False)
    p.front_box.setChecked(True)
    _channel_box(w, "VIM").setChecked(False)
    p.go(4)
    w.accept_cell()
    before = shown_choices(w)
    w.save_results(folder=tmp_path / "analysis", notify=False)
    _idle(w)
    w.open_cell_viewer()
    _pump(0.5)
    v = w._viewers[-1]
    v.cell_list.setCurrentRow(0)
    _pump(0.5)
    v.set_mode("3d")
    _pump(1.0)
    v.close()
    _pump(0.3)
    assert shown_choices(w) == before and p.step == 4
    p.go(3)
    _idle(w)
    assert shown_choices(w) == before and display_problems(w) == []


def test_next_cell_and_open_cell_keep_display_choices(window):
    """Display choices are global; factors are per cell (reset by Next cell, restored by Open cell)."""
    w, p = window, window.panel
    _open_example_at_step4(w)
    p.show_box["cytoplasm"].setChecked(False)
    p.target_box["TTN"].setChecked(True)
    p.front_box.setChecked(True)
    p.outline_box.setChecked(False)
    _channel_box(w, "VIM").setChecked(False)
    p.factor["actin"].setValue(0.85)
    _idle(w)
    p.go(4)
    w.accept_cell()
    display = {k: v for k, v in shown_choices(w).items() if k != "factors"}
    w.next_cell()
    _idle(w)
    p.go(3)
    _idle(w)
    assert {k: v for k, v in shown_choices(w).items() if k != "factors"} == display
    assert p.factor["actin"].value() == pytest.approx(1.0)
    assert display_problems(w) == []
    p.go(4)
    w.edit_cell(0)
    _idle(w)
    p.go(3)
    _idle(w)
    assert {k: v for k, v in shown_choices(w).items() if k != "factors"} == display
    assert p.factor["actin"].value() == pytest.approx(0.85)
    assert display_problems(w) == []


def _two_images(w, tmp_path):
    from localization_tool.core.channels import ChannelAssignment
    from localization_tool.core.reader import write_imagej_tiff
    from localization_tool.core.synthetic import make_golden_cell

    folder = tmp_path / "20260301 - WT"
    folder.mkdir()
    for name, shift in (("a.tif", -1.0), ("b.tif", -0.5)):
        stack, _ = make_golden_cell(voxel_um=(0.13, 0.08, 0.08), nucleus_shift_x=shift, margin_um=1.5)
        write_imagej_tiff(stack, folder / name)
    w.add_folder(folder, sync=True)
    roles = [ChannelAssignment("Target", "T", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
             ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]
    for im in w.session.images:
        im.assignments = [ChannelAssignment(**vars(a)) for a in roles]


def _open_image(w, index):
    w.open_image_index(index)
    t = time.time()
    while (w.state.index != index or w.state.full is None or w.panel._working) and time.time() - t < 60:
        _pump(0.1)
    _idle(w)


def test_next_image_and_apply_to_all_keep_display_choices(window, tmp_path):
    """Global display choices survive 'Next image' and 'Apply settings to all cells' (target mask boxes are
    rebuilt unticked on every image, same cause as the channel-edit case)."""
    from localization_tool.core.params import AnalysisParams

    w, p = window, window.panel
    _two_images(w, tmp_path)
    _open_image(w, 1)                                       # b first: accepted with the old settings
    p.go(2)
    _idle(w)
    p.go(3)
    _idle(w)
    p.go(4)
    w.accept_cell()
    d = AnalysisParams().to_dict()
    d["nucleus"]["factor"] = 0.9
    p.go(0)
    w._set_params(AnalysisParams.from_dict(d), ask=False)  # b is now computed with other settings
    _open_image(w, 0)
    p.go(2)
    _idle(w)
    p.go(3)
    _idle(w)
    p.show_box["actin"].setChecked(False)
    p.target_box["T"].setChecked(True)
    p.front_box.setChecked(True)
    _channel_box(w, "Actin").setChecked(False)
    p.go(4)
    w.accept_cell()
    before = shown_choices(w)
    w.run_all("all")                                        # 'Apply settings to all cells' in step 5
    t = time.time()
    while w._batch is not None and time.time() - t < 180:
        _pump(0.1)
    _idle(w)
    assert w.session.images[1].cells[0].results["params"]["nucleus"]["factor"] == pytest.approx(0.9)
    assert p.step == 4 and shown_choices(w) == before
    p.go(3)
    _idle(w)
    assert shown_choices(w) == before and display_problems(w) == []
    if w.session.images[1].cells[0].status == "automatic":
        w.next_image()                                      # b is 'to check': Next image opens it
    else:
        w.edit_cell(1)                                      # outlines unchanged: open it from the list
    t = time.time()
    while (w.state.index != 1 or w.state.result is None and p.step < 3) and time.time() - t < 120:
        _pump(0.1)
    _idle(w)
    p.go(3)
    _idle(w)
    got = shown_choices(w)
    assert got["show"] == before["show"] and got["front"] == before["front"]
    assert got["targets"] == before["targets"], "target mask T unticked on the next image"
    assert display_problems(w) == []


# ---------------------------------------------------------------------------- random walk
class Walk:
    """Random user: every action changes the remembered state the way a user expects, then the remembered
    state is compared with what the program shows."""

    def __init__(self, w, rng, use_3d):
        self.w, self.p, self.rng, self.use_3d = w, w.panel, rng, use_3d
        self.log: list[str] = []
        self.expect = shown_choices(w)
        self.saved_factors: dict[str, dict] = {}
        self.cell_name = None

    def fail(self, what):
        raise AssertionError(f"{what}\nafter actions:\n  " + "\n  ".join(self.log[-25:]))

    def check(self):
        _idle(self.w)
        got = shown_choices(self.w)
        for key, value in self.expect.items():
            if key == "targets":                            # only labels that exist
                value = {k: v for k, v in value.items() if k in got["targets"]}
            if got[key] != value:
                self.fail(f"remembered {key}={value}, shown {got[key]}")
        problems = [x for x in display_problems(self.w) if not any(i and i in x for i in IGNORE)]
        if problems:
            self.fail("display does not match: " + "; ".join(problems))

    def actions(self):
        p, w = self.p, self.w
        acts = [("step bar", self.step_bar), ("next", self.next), ("back", self.back)]
        if p.step == 3:
            acts += [("toggle mask", self.toggle_mask), ("toggle target", self.toggle_target),
                     ("toggle front", self.toggle_front), ("toggle outlines", self.toggle_outline),
                     ("only menu", self.only), ("factor", self.factor), ("channel box", self.channel_box)]
        if p.step == 2:
            acts += [("cell factor", self.cell_factor), ("channel box", self.channel_box)]
        if p.step == 4:
            acts += [("accept", self.accept), ("next cell", self.next_cell), ("open cell", self.open_cell)]
        if self.use_3d and p.step in (2, 3):
            acts += [("2D/3D", self.toggle_3d)]
        acts += [("plane", self.plane)]
        return acts

    def run(self, n):
        for _ in range(n):
            name, fn = self.rng.choice(self.actions())
            text = fn()
            self.log.append(f"{name}: {text} (now step {self.p.step + 1}, {self.w.viewer.dims.ndisplay}D)")
            self.check()

    # each action returns a short description
    def step_bar(self):
        k = self.rng.randrange(5)
        if self.p.header.buttons[k].isEnabled():
            self.p.header.buttons[k].click()
        return f"step bar {k + 1}"

    def next(self):
        if self.p.next_btn.isEnabled() and self.p.next_btn.isVisible():
            self.p.next_btn.click()
        return "Next"

    def back(self):
        if self.p.back_btn.isEnabled():
            self.p.back_btn.click()
        return "Back"

    def toggle_mask(self):
        key = self.rng.choice(list(self.p.show_box))
        cb = self.p.show_box[key]
        cb.setChecked(not cb.isChecked())
        self.expect["show"][key] = cb.isChecked()
        return f"{key} -> {cb.isChecked()}"

    def toggle_target(self):
        if not self.p.target_box:
            return "no targets"
        key = self.rng.choice(list(self.p.target_box))
        cb = self.p.target_box[key]
        cb.setChecked(not cb.isChecked())
        self.expect["targets"][key] = cb.isChecked()
        return f"target {key} -> {cb.isChecked()}"

    def toggle_front(self):
        if not self.p.front_box.isEnabled():
            return "front disabled (3D)"
        self.p.front_box.setChecked(not self.p.front_box.isChecked())
        self.expect["front"] = self.p.front_box.isChecked()
        return f"front -> {self.expect['front']}"

    def toggle_outline(self):
        if not self.p.outline_box.isEnabled():
            return "outlines disabled (3D)"
        self.p.outline_box.setChecked(not self.p.outline_box.isChecked())
        self.expect["outline"] = self.p.outline_box.isChecked()
        return f"outlines -> {self.expect['outline']}"

    def only(self):
        keys = [None, "nucleus", "actin", "actin_rich", "cytoplasm", "target"]
        i = self.rng.randrange(len(keys))
        self.p.only_menu.setCurrentIndex(i)
        self.p.only_menu.activated.emit(i)
        if self.p.only_menu.currentIndex() == i:
            key = keys[i]
            self.expect["show"] = {k: cb.isChecked() for k, cb in self.p.show_box.items()}
            self.expect["targets"] = {k: cb.isChecked() for k, cb in self.p.target_box.items()}
            want = {k: key is None or k == key for k in self.p.show_box}
            if self.expect["show"] != want:
                self.fail(f"'Only' menu {self.p.only_menu.currentText()} gave {self.expect['show']}")
        return self.p.only_menu.currentText()

    def factor(self):
        key = self.rng.choice(["nucleus", "actin", "actin_rich", "target"])
        value = self.rng.choice([0.8, 0.9, 1.0, 1.1])
        self.p.factor[key].setValue(value)
        self.expect["factors"][key] = value
        return f"factor {key} = {value}"

    def cell_factor(self):
        value = self.rng.choice([0.8, 1.0, 1.2])
        self.p.cell_factor.setValue(value)
        self.expect["factors"]["cell"] = value
        return f"edge factor = {value}"

    def channel_box(self):
        row = self.p.comp_channels if self.p.step == 3 else self.p.cell_channels
        boxes = _checkboxes(row)
        b = self.rng.choice(boxes)
        b.setChecked(not b.isChecked())
        self.expect["channels"][b.text()] = b.isChecked()
        return f"channel {b.text()} -> {b.isChecked()}"

    def toggle_3d(self):
        n = 5 - self.w.viewer.dims.ndisplay
        (self.p.view_3d if n == 3 else self.p.view_2d).click()
        self.expect["ndisplay"] = n
        if n == 3 and self.p.step == 3:
            _wait_surfaces(self.w)
        return f"view {n}D"

    def plane(self):
        if self.w.viewer.dims.ndisplay == 2:
            self.w.viewer.dims.set_current_step(0, self.rng.randrange(self.w.state.full.shape_zyx[0]))
        return f"plane {self.w.viewer.dims.current_step[0]}"

    def accept(self):
        if self.p.accept_btn.isEnabled():
            self.p.accept_btn.click()
            self.saved_factors[self.w.state.editing] = dict(self.expect["factors"])
        return "accept"

    def next_cell(self):
        self.p.next_cell_btn.click()
        self.expect["factors"] = {k: round(v, 4) for k, v in self.w.base_factors().items()}
        return "next cell"

    def open_cell(self):
        if not self.saved_factors:
            return "no cell yet"
        names = sorted(self.saved_factors)
        name = self.rng.choice(names)
        self.w.edit_cell(names.index(name))
        self.expect["factors"] = dict(self.saved_factors[name])
        return f"open {name}"


@pytest.mark.parametrize("seed,use_3d", [(87, False), (88, True)])
def test_random_walk(window, seed, use_3d):
    w = window
    if os.environ.get("LT_D87_EXPLORE"):
        from localization_tool.gui import app as A

        orig = A.MainWindow.show_compartments

        def safe(self, result):
            try:
                orig(self, result)
            except ValueError:
                self.update_compartment_display()

        w.show_compartments = safe.__get__(w)
    w.load_example()
    walk = Walk(w, random.Random(seed + int(os.environ.get("LT_D87_SEED_OFFSET", "0"))), use_3d)
    walk.check()
    walk.run(int(os.environ.get("LT_D87_STEPS", "60")))


# ---------------------------------------------------------------------------- real data
@pytest.mark.realdata
@pytest.mark.skipif(not REAL_CZI.exists(), reason="real CZI not available")
def test_real_czi_step_changes_keep_display(window):
    w, p = window, window.panel
    w.add_images([REAL_CZI], open_first=True)
    t = time.time()
    while w.state.full is None and time.time() - t < 120:
        _pump(0.1)
    assert w.state.full is not None, w.messages
    assert w.can_leave(1), [a.role for a in w.state.assignments]
    p.go(2)
    _idle(w, 300)
    p.go(3)
    _idle(w, 300)
    assert p.step == 3 and w.state.result is not None, w.messages
    p.show_box["cytoplasm"].setChecked(False)
    _channel_box(w, w.image_layers()[0].name).setChecked(False)
    choices = shown_choices(w)
    found = []                                  # every mismatch on the way, so the 3D part also runs
    for k in (4, 3, 1, 3, 0, 3):
        p.go(k)
        _idle(w, 300)
        if shown_choices(w) != choices:
            found.append(f"step {k + 1}: choices {shown_choices(w)}")
        found += [f"step {k + 1}: {x}" for x in display_problems(w)]
    w.set_ndisplay(3)
    _wait_surfaces(w, 300)
    choices["ndisplay"] = 3
    found += [f"3D step 4: {x}" for x in display_problems(w)]
    p.go(4)
    _idle(w)
    try:
        p.go(3)
    except Exception as exc:                    # noqa: BLE001 - reported below
        found.append(f"3D back to step 4 raised {exc!r}")
    _idle(w, 300)
    if shown_choices(w) != choices:
        found.append(f"3D back: choices {shown_choices(w)}")
    found += [f"3D back to step 4: {x}" for x in display_problems(w)]
    assert found == [], "\n".join(found)
