"""Stress and robustness audit of the GUI (crashes and hangs in daily use, audit D87).

Every test drives the real main window with the synthetic practice cell. Tests marked realdata use
one real CZI. The tests describe the wanted behaviour: a failing test is a confirmed defect.
Run:  xvfb-run -a -s "-screen 0 1600x1000x24" python -m pytest -q tests/test_d87_stress.py
Software OpenGL (llvmpipe) is used under xvfb, GPU driver crashes cannot reproduce here.
"""

import faulthandler
import gc
import os
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import weakref
from pathlib import Path

import pytest

faulthandler.enable(all_threads=True)

if not os.environ.get("LT_D87"):
    # long and memory hungry (about 20 min, each test best in its own process):
    #   LT_D87=1 xvfb-run -a pytest -q tests/test_d87_stress.py -k <name>
    pytest.skip("D87 audit tests: set LT_D87=1", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
_found = sorted(Path(os.environ["LT_TEST_DATA"]).rglob("*.czi")) if os.environ.get("LT_TEST_DATA") else []
REAL_CZI = Path(os.environ.get("LT_D87_CZI") or (str(_found[0]) if _found else "/no/real/file.czi"))
NEEDS_DISPLAY = pytest.mark.skipif(sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
                                   reason="no display (use xvfb-run)")


# ---------------------------------------------------------------------------- helpers
def _pump(seconds=0.3):
    from qtpy.QtWidgets import QApplication

    t = time.time()
    while time.time() - t < seconds:
        QApplication.processEvents()
        time.sleep(0.01)


def _live_workers() -> int:
    from superqt.utils._qthreading import WorkerBase

    return len(WorkerBase._worker_set)


def _idle(w) -> bool:
    p = w.panel
    return (not w.state.computing and not p._working and not p._debounce.isActive()
            and not p._cell_debounce.isActive() and _live_workers() == 0)


def _wait_idle(w, seconds=120):
    t = time.time()
    while time.time() - t < seconds:
        _pump(0.05)
        if _idle(w):
            _pump(0.1)
            return
    raise AssertionError("the program did not become idle (calculation hangs)")


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    import psutil  # pragma: no cover

    return psutil.Process().memory_info().rss / 1e6  # pragma: no cover


def _deferred_deletes() -> None:
    """Tests pump events without app.exec(), where Qt keeps deleteLater() until a loop runs."""
    from qtpy.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _alive(cls) -> int:
    gc.collect()
    return sum(1 for o in gc.get_objects() if type(o) is cls)


def _free_leaked(refs) -> None:
    """Free cell viewers the program left alive, so later tests keep their memory (test hygiene only)."""
    for ref in refs:
        v = ref()
        if v is None:
            continue
        try:
            if v._qt3d is not None:
                v._model3d.layers.clear()
                v._qt3d.close()
            v.deleteLater()
        except RuntimeError:
            pass
    _pump(0.3)
    gc.collect()


@pytest.fixture
def errors(monkeypatch):
    """Exceptions raised inside Qt slots and worker callbacks (napari shows them as pop-ups)."""
    found = []

    def hook(etype, value, tb):
        found.append("".join(traceback.format_exception(etype, value, tb)))

    monkeypatch.setattr(sys, "excepthook", hook)
    monkeypatch.setattr(threading, "excepthook", lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback))
    return found


@pytest.fixture
def window(tmp_path, monkeypatch, errors):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("no display (use xvfb-run)")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import napari

    from localization_tool.gui import app as A
    from localization_tool.gui import theme

    A._patch_napari()
    viewer = napari.Viewer(show=True)
    viewer.theme = theme.register_napari_theme()
    w = A.MainWindow(viewer)
    w.setup()
    msgs = []
    monkeypatch.setattr(w, "message", lambda text, *a, **k: msgs.append(str(text)))
    w._test_messages = msgs
    yield w
    try:
        _wait_idle(w, 60)
    except AssertionError:
        pass
    viewer.close()
    _pump(0.2)


def _to_step4(w):
    p = w.panel
    if w.state.full is None:
        w.load_example()
        _pump(0.2)
    p.go(2)
    _wait_idle(w)
    p.go(3)
    _wait_idle(w)
    assert p.step == 3 and w.state.result is not None


def _no_errors(errors):
    assert not errors, "exception(s) in Qt slots:\n" + "\n".join(e[-1500:] for e in errors)


# ---------------------------------------------------------------------------- 1 crash log
def test_crash_log_keeps_a_windows_native_crash(tmp_path):
    """faulthandler writes 'Windows fatal exception: access violation' on Windows (no 'Fatal Python
    error', no 'Traceback'). The next start must keep that log, not overwrite it."""
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    log = cfg / "crash.log"
    windows_dump = ("CellLocalizer 1.4 started 2026-10-13 09:00:00\n"
                    "Windows fatal exception: access violation\n\n"
                    "Current thread 0x00001a2c (most recent call first):\n"
                    '  File "vispy\\gloo\\glir.py", line 1311 in _create\n'
                    '  File "napari\\_vispy\\canvas.py", line 600 in on_draw\n')
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        os.environ["LT_CONFIG_DIR"] = {str(cfg)!r}
        from localization_tool.gui import app as A
        print("PREVIOUS=" + repr(A._install_crash_log() is not None))
    """)
    log.write_text(windows_dump, encoding="utf-8")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         env={**os.environ, "LT_CONFIG_DIR": str(cfg)})
    assert "PREVIOUS=" in out.stdout, out.stderr
    kept = list(cfg.glob("crash*.log"))
    still_there = any("access violation" in p.read_text(encoding="utf-8", errors="replace") for p in kept)
    assert "PREVIOUS=True" in out.stdout and still_there, (
        "a native Windows crash is not recognised and crash.log is overwritten at the next start "
        f"(stdout {out.stdout!r}, files {[p.name for p in kept]})")


# ---------------------------------------------------------------------------- 2 compartments shown in 3D
def test_compartments_shown_while_in_3d(window, errors):
    """Step 3 in 3D, then Next (or Back from step 5 while in 3D): show_compartments adds the 2D
    front arrow to a Shapes layer while the viewer is 3D and raises ValueError."""
    w, p = window, window.panel
    w.load_example()
    _pump(0.2)
    p.go(2)
    _wait_idle(w)
    w.set_ndisplay(3)
    _pump(0.2)
    p.go(3)                                  # Next while the view is 3D
    _wait_idle(w)
    assert w.state.result is not None
    _no_errors(errors)
    assert p.comp_status.text(), "done() stopped half-way: no status text"
    assert w._surfaces, "the 3D surfaces were never requested"


def test_back_from_results_while_in_3d(window, errors):
    w, p = window, window.panel
    _to_step4(w)
    p.go(4)
    w.set_ndisplay(3)
    _wait_idle(w)
    try:
        p.go(3)                              # Back while in 3D
    except ValueError:
        errors.append(traceback.format_exc())
    _wait_idle(w)
    _no_errors(errors)


# ---------------------------------------------------------------------------- 3 cell viewer
def _analysis(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_v13 import make_analysis

    return make_analysis(tmp_path / "analysis")


def test_cell_viewer_open_close_releases_memory(window, tmp_path, errors):
    """Opening and closing the cell viewer (with 3D and a turned camera) must free the window,
    its napari QtViewer (one OpenGL context each) and the cell data."""
    from napari.qt import QtViewer

    from localization_tool.gui import cellviewer

    path = _analysis(tmp_path)
    base_cv, base_qv = _alive(cellviewer.CellViewer), _alive(QtViewer)
    n = 4                                    # the first round warms up napari/vispy (about 350 MB here)
    rss, refs = [], []
    try:
        _open_close_rounds(window, path, n, rss, refs)
    finally:
        cv, qv = _alive(cellviewer.CellViewer) - base_cv, _alive(QtViewer) - base_qv
        _free_leaked(refs)
    per_round = (rss[-1] - rss[0]) / (n - 1)
    _no_errors(errors)
    assert cv == 0 and qv == 0 and per_round < 60, (
        f"after {n} open/close: {cv} CellViewer and {qv} QtViewer (GL canvas) still alive, "
        f"memory +{per_round:.0f} MB per open/close after the first")


def _open_close_rounds(window, path, n, rss, refs):
    from localization_tool.gui import cellviewer

    for _ in range(n):
        v = cellviewer.open_viewer(path, window.viewer.window._qt_window)
        refs.append(weakref.ref(v))
        v.cell_list.setCurrentRow(0)
        v.wait_idle()                            # the cell loads in the background (D87)
        v.set_mode("3d")
        _pump(0.3)
        v.wait_idle()
        for a in range(5):
            v._model3d.camera.angles = (0, 15 * a, 90)
            _pump(0.02)
        assert len(v._model3d.camera.events.angles.callbacks) < 20
        v.close()
        _pump(0.2)
        _deferred_deletes()                      # what the running event loop of the program does
        del v
        gc.collect()
        rss.append(_rss_mb())


# ---------------------------------------------------------------------------- 4 stale surface results
def test_surface_result_after_new_analysis_is_dropped(window, errors):
    """File > New analysis (menu not locked) while the 3D surfaces are computed: the old cell's
    surfaces must not appear in the empty viewer."""
    w = window
    _to_step4(w)
    w.set_ndisplay(3)
    assert _live_workers() >= 1
    w.new_analysis(confirm=False)
    _wait_idle(w)
    stale = [l.name for l in w.viewer.layers if l.name.startswith("3D ")]
    _no_errors(errors)
    assert not stale, f"surfaces of the closed cell added after New analysis: {stale}"


def test_surface_result_after_reopening_image_is_dropped(window, errors):
    w = window
    _to_step4(w)
    w.set_ndisplay(3)
    w.load_example()                         # Help > Practice ...: set_image does not wait for surfaces
    _wait_idle(w)
    stale = [l.name for l in w.viewer.layers if l.name.startswith("3D ")]
    assert w.state.result is None
    assert not stale and w._surface_key is None, f"stale surfaces on the new image: {stale}"


# ---------------------------------------------------------------------------- 5 overlapping workers
def test_overlapping_workers_keep_the_panel_locked(window, errors, monkeypatch):
    """A factor changed < 0.5 s before 3D: surfaces and compartments run at the same time. The first
    worker to finish calls set_working(False) and unlocks the panel while the other still runs."""
    from localization_tool.gui import app as A

    w, p = window, window.panel
    _to_step4(w)
    real = A.analyze_cell
    monkeypatch.setattr(A, "analyze_cell", lambda *a, **k: (time.sleep(3), real(*a, **k))[1])
    p.factor["nucleus"].setValue(0.8)        # debounce 0.5 s
    w.set_ndisplay(3)                        # surfaces start now, the compute 0.5 s later
    bad, t = 0, time.time()
    while time.time() - t < 120:
        _pump(0.02)
        if w.state.computing and (not p._working or p.view_row.isEnabled()):
            bad += 1
        if _idle(w):
            break
    _wait_idle(w)
    assert bad == 0, f"panel and 2D/3D buttons unlocked during a running calculation ({bad} samples)"


def test_close_during_single_cell_compute_asks(window, monkeypatch):
    """Closing while a cell is computed: the program must ask or wait for the worker; before the fix
    confirm_close only knew the batch and napari waited in aboutToQuit (window gone, process running)."""
    from localization_tool.gui import app as A

    w, p = window, window.panel
    w.load_example()
    _pump(0.2)
    real = A.cell_mask
    monkeypatch.setattr(A, "cell_mask", lambda *a, **k: (time.sleep(2), real(*a, **k))[1])
    asked = []
    monkeypatch.setattr(w, "ask", lambda text, *a, **k: asked.append(text) or True)
    p.go(2)
    assert w.state.computing and _live_workers() >= 1
    ok = w.confirm_close()
    # fix: the program waits for the running calculation (up to 30 s) before it closes
    assert asked or (ok and not w.state.computing and not p._working), \
        "the program may quit while a worker runs, without asking or waiting"
    _wait_idle(w)


# ---------------------------------------------------------------------------- 6 hangs: visibility churn
def test_unchanged_display_does_not_touch_layers(window):
    """napari 0.6.6 rebuilds all overlays of a layer at every `visible =` assignment, even unchanged.
    One click on a compartment checkbox re-assigns about a dozen layers (about 1 s here)."""
    w = window
    _to_step4(w)
    count = {"n": 0}
    for layer in w.viewer.layers:
        layer.events.visible.connect(lambda e: count.__setitem__("n", count["n"] + 1))
    t = time.perf_counter()
    w.update_compartment_display()           # nothing changed
    dt = time.perf_counter() - t
    assert count["n"] == 0, f"{count['n']} visible assignments without change, {dt:.2f} s blocked"


# ---------------------------------------------------------------------------- 7 Ctrl+Y bypasses set_ndisplay
def test_napari_shortcut_ctrl_y_goes_through_set_ndisplay(window, errors):
    """napari's own Ctrl+Y (toggle_ndisplay) is still bound: it switches to 3D without set_ndisplay,
    so contours stay on, the voxel labels stay visible and no surfaces are made."""
    from napari.utils.action_manager import action_manager

    w = window
    _to_step4(w)
    bound = [str(k) for k in action_manager._shortcuts.get("napari:toggle_ndisplay", [])]
    action_manager.trigger("napari:toggle_ndisplay")      # what Ctrl+Y on the image does
    _pump(0.3)
    lab = w._layer(w.LABEL_LAYER)
    assert not (bound and (lab.visible or lab.contour)), (
        f"{bound}: 3D shown with voxel labels (contour {lab.contour}) instead of surfaces")


# ---------------------------------------------------------------------------- 8 stress
def test_toggle_2d_3d_30_times_also_while_computing(window, errors):
    """The user clicks 2D/3D 30 times quickly, also while the compartments are computed. Only enabled
    buttons are clicked (like a user). No exception, no worker left, no growing connections."""
    w, p = window, window.panel
    _to_step4(w)
    cam = w.viewer.camera.events.angles
    cbs0, r0 = len(cam.callbacks), _rss_mb()
    clicks = 0
    for i in range(30):
        if i == 10:
            p.factor["actin"].setValue(0.85)     # a compute starts in 0.5 s
        b = p.view_3d if w.viewer.dims.ndisplay == 2 else p.view_2d
        if b.isEnabled() and p.view_row.isEnabled():
            b.click()
            clicks += 1
        _pump(0.05)
    _wait_idle(w, 300)
    w.set_ndisplay(2)
    _wait_idle(w)
    cbs1, grown = len(cam.callbacks), _rss_mb() - r0
    _no_errors(errors)
    assert _live_workers() == 0
    assert cbs1 <= cbs0 + len(w._surfaces) + 2, f"camera callbacks {cbs0} -> {cbs1}"
    assert grown < 600, f"memory +{grown:.0f} MB after {clicks} toggles"
    assert w.state.result_key == w._result_key()


def test_rapid_factor_changes_never_lose_the_last_value(window, errors):
    """Factors changed quickly (only while the boxes are enabled, like a user): the shown result
    belongs to the last factors and every worker ends."""
    w, p = window, window.panel
    _to_step4(w)
    values = [0.7, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05, 0.75, 1.0, 0.85]
    for i, v in enumerate(values):
        key = ("nucleus", "actin")[i % 2]
        if p.factor[key].isEnabled() and p.pages.isEnabled():
            p.factor[key].setValue(v)
        _pump((0.1, 0.4, 0.6)[i % 3])
    _wait_idle(w, 300)
    _no_errors(errors)
    assert _live_workers() == 0
    assert w.state.result_key == w._result_key(), "result does not belong to the shown factors"
    assert p.next_btn.isEnabled()


def test_switch_step_and_cell_while_workers_run(window, errors):
    """Back and forth between steps, Next cell and the practice image while workers run."""
    w, p = window, window.panel
    _to_step4(w)
    for i in range(6):
        p.factor["nucleus"].setValue(0.8 + 0.05 * i)
        _pump(0.55)                              # the compute has just started
        for k in (2, 1, 0):
            if p.header.buttons[k].isEnabled():
                p.header.buttons[k].click()
                _pump(0.05)
        _wait_idle(w)
        p.go(2)
        _wait_idle(w)
        p.go(3)
        _wait_idle(w)
        if i == 3:
            p.go(4)
            w.next_cell()
            _wait_idle(w)
            p.go(3)
            _wait_idle(w)
    _no_errors(errors)
    assert _live_workers() == 0 and p.step == 3 and w.state.result_key == w._result_key()


def test_random_clicks_monkey(window, errors):
    """100 random clicks on enabled controls with random pauses (seeded)."""
    import random

    w, p = window, window.panel
    rng = random.Random(1)
    w.load_example()
    _pump(0.2)
    done = []
    for _ in range(100):
        acts = [(f"step{k + 1}", b.click) for k, b in enumerate(p.header.buttons) if b.isEnabled()]
        for name, b in (("next", p.next_btn), ("back", p.back_btn)):
            if b.isEnabled() and b.isVisible():
                acts.append((name, b.click))
        if p.view_row.isEnabled():
            acts += [("2D", p.view_2d.click), ("3D", p.view_3d.click)]
        if p.pages.isEnabled() and p.step == 2:
            acts.append(("cell factor", lambda: p.cell_factor.setValue(rng.choice([0.8, 0.9, 1.0, 1.1]))))
        if p.pages.isEnabled() and p.step == 3:
            k = rng.choice(list(p.factor))
            acts.append(("factor", lambda k=k: p.factor[k].setValue(rng.choice([0.8, 1.0, 1.2]))))
            cb = rng.choice(list(p.show_box.values()) + [p.front_box, p.outline_box])
            if cb.isEnabled():
                acts.append(("checkbox", cb.toggle))
        name, fn = rng.choice(acts)
        done.append((p.step, name))
        try:
            fn()
        except Exception:
            errors.append(f"after {done[-3:]}:\n" + traceback.format_exc())
        _pump(rng.choice([0.0, 0.05, 0.2, 0.6]))
    _wait_idle(w, 300)
    _no_errors(errors)
    assert _live_workers() == 0


# ---------------------------------------------------------------------------- 9 real data
@pytest.mark.realdata
@pytest.mark.skipif(not REAL_CZI.exists(), reason="real CZI not available")
def test_real_czi_toggle_and_cell_viewer(window, errors, tmp_path):
    """One real Airyscan CZI: load, cell edge, compartments, 2D/3D toggles; then the cell viewer of
    this cell opened and closed three times (memory must not grow per round)."""
    from localization_tool.core import batch, project_file
    from localization_tool.core.channels import ChannelAssignment
    from localization_tool.core.params import AnalysisParams
    from localization_tool.core.reader import read_image
    from localization_tool.gui import cellviewer

    w, p = window, window.panel
    w.add_images([REAL_CZI], open_first=True)
    t = time.time()
    while (w.state.full is None or p._working) and time.time() - t < 120:
        _pump(0.05)
    assert w.state.full is not None
    p.go(1)
    p.go(2)
    _wait_idle(w, 300)
    p.go(3)
    _wait_idle(w, 300)
    for _ in range(4):
        w.set_ndisplay(3)
        _wait_idle(w, 300)
        w.set_ndisplay(2)
        _pump(0.1)
    _no_errors(errors)
    assert _live_workers() == 0

    # a saved analysis with this cell, opened in the cell viewer
    roles = [ChannelAssignment("Target", "TTN", "#FF2BD6"), ChannelAssignment("Actin", "Actin", "#2BD46B"),
             ChannelAssignment("Nucleus", "Nucleus", "#3D7DFF")]
    params = AnalysisParams(sensitivity=False, colocalization=False)
    im = project_file.ImageSpec(path=REAL_CZI, experiment="x", condition="TC", assignments=list(roles))
    af = project_file.AnalysisFile(images=[im], params=params)
    spec = project_file.CellSpec(name="cell1", status="accepted")
    _, box, r = batch.analyze_spec(read_image(REAL_CZI), im.assignments, params, spec)
    spec.box, spec.results = box, batch.cell_results(im, spec, box, r, params)
    im.cells.append(spec)
    folder = tmp_path / "analysis"
    folder.mkdir()
    path = project_file.save(af, folder / "analysis.lta.json")
    rss, blocked, refs = [], [], []
    for _ in range(3):
        v = cellviewer.open_viewer(path, None)
        refs.append(weakref.ref(v))
        v.cell_list.setCurrentRow(0)
        v.wait_idle()
        v.set_mode("3d")
        v._render_timer.stop()
        t = time.perf_counter()
        v._render()                          # the mask surfaces are computed in the background (D87)
        blocked.append(time.perf_counter() - t)
        v.wait_idle()
        _pump(0.2)
        v.close()
        _pump(0.2)
        _deferred_deletes()                      # what the running event loop of the program does
        del v
        gc.collect()
        rss.append(_rss_mb())
    _free_leaked(refs)
    per_round = (rss[-1] - rss[0]) / 2
    _no_errors(errors)
    # volume upload stays in the GUI thread (about 2 s with software OpenGL), surfaces run in the background;
    # Windows reports "not responding" after 5 s
    assert max(blocked[1:]) < 4.0, f"3D render blocked the GUI thread {[round(b, 1) for b in blocked]} s"
    assert per_round < 100, (f"cell viewer, real cell: memory +{per_round:.0f} MB per open/close after the first "
                             f"(3D render blocked the GUI thread {[round(b, 1) for b in blocked]} s)")
