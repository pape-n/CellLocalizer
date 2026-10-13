"""Main window: napari canvas reduced to what the workflow needs (D2, D17) with the guided
workflow panel on the right (D21). All calculations go through localization_tool.core.

Layers are created once per image and then only updated: one image layer per channel, one
shapes layer for excluded regions, one labels layer for the cell edge, one for the compartments
and one per target mask. Creating and removing layers repeatedly cost seconds and graphics memory
on the lab computers (0.2.0a1 feedback).

An analysis is a folder with analysis.lta.json, Results.xlsx, the control images and the
corrected masks. It can be opened again; stored results are shown without recalculation.
"""

from __future__ import annotations

import faulthandler
import itertools
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from localization_tool import APP_NAME, __version__
from localization_tool.core import batch, project_file
from localization_tool.core.analysis import SENSITIVITY_FACTORS, CellResult, analyze_cell, cell_mask
from localization_tool.core.conditions import suggest_condition, suggest_experiment
from localization_tool.core.channels import (
    DISPLAY_COLORS, ChannelAssignment, Panel, find_panel, load_panels, save_panels, suggest_assignments,
    upsert_panel, validate,
)
from localization_tool.core.crop import CropBox, crop
from localization_tool.core.image import ImageStack
from localization_tool.core.params import MASK_KEYS, AnalysisParams
from localization_tool.core.reader import SUPPORTED_SUFFIXES, ReaderError, read_image
from localization_tool.core.segmentation import exclusion_mask
from localization_tool.paths import panels_file, user_config_dir
from localization_tool import prefs
from localization_tool.gui import theme

# compartment labels shown in step 4: 1 nucleus, 2 actin, 4 actin-rich (part of actin), 3 cytoplasm
LABEL_COLORS = {1: (0.24, 0.49, 1.0, 1), 2: (0.17, 0.83, 0.42, 1), 3: (0.6, 0.64, 0.69, 1), 4: (1.0, 0.84, 0.04, 1)}
LABEL_KEYS = {1: "nucleus", 2: "actin", 3: "cytoplasm", 4: "actin_rich"}
PAINT_VALUES = (1, 2, 3, 0)          # Nucleus, Actin, Cytoplasm, Not cell
EXCLUDE_EDIT = {"edge_color": "#E0605A", "face_color": [0.88, 0.38, 0.35, 0.3]}
EXCLUDE_DONE = {"edge_color": "#3A414A", "face_color": [0, 0, 0, 1]}
ANALYSIS_NAME = "analysis.lta.json"
STATUS_TEXT = {"automatic": "to check"}      # the stored status "automatic" is shown as "to check"
MEMORY_FACTOR = 13                            # peak memory of an analysis / size of the image (phase 5 review)
MEMORY_WARN_GB = 6.0
EXAMPLE_PATH = Path("Synthetic example")


def _is_example(path: Path | None) -> bool:
    return path is not None and Path(path).name == EXAMPLE_PATH.name and not Path(path).exists()


@dataclass
class AppState:
    full: ImageStack | None = None
    shown: ImageStack | None = None          # kept for the performance test
    path: Path | None = None
    index: int = -1                          # position in the image list of the analysis
    assignments: list[ChannelAssignment] = field(default_factory=list)
    load_seconds: float = 0.0
    experiment: str = ""
    condition: str = ""
    box: CropBox | None = None
    exclusions: list = field(default_factory=list)
    crop: ImageStack | None = None
    crop_key: tuple | None = None
    cell: np.ndarray | None = None           # cell mask on the crop grid (step 3)
    cell_key: tuple | None = None
    cell_corrected: bool = False
    cell_thresholds: dict = field(default_factory=dict)
    cell_variants: dict = field(default_factory=dict)   # cell masks at the sensitivity factors (D56)
    result: CellResult | None = None
    result_key: tuple | None = None
    result_corrected: bool = False
    computing: bool = False
    accepted_current: bool = False
    editing: str | None = None               # name of the accepted cell being edited again


# ---------------------------------------------------------------------------- crash log
def crash_log_path() -> Path:
    return user_config_dir() / "crash.log"


def _install_crash_log() -> str | None:
    """Write native crashes and unhandled errors to crash.log. Returns the text of a log left by
    a previous crash, so the program can point to it once."""
    path = crash_log_path()
    previous = None
    try:
        if path.exists() and path.stat().st_size > 0:
            previous = path.read_text(encoding="utf-8", errors="replace")[-4000:]
        fh = open(path, "w", encoding="utf-8")
        fh.write(f"{APP_NAME} {__version__} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.flush()
        faulthandler.enable(file=fh, all_threads=True)
        old_hook = sys.excepthook

        def hook(etype, value, tb):
            fh.write("".join(traceback.format_exception(etype, value, tb)))
            fh.flush()
            old_hook(etype, value, tb)

        sys.excepthook = hook
        _install_crash_log.handle = fh
    except OSError:
        pass
    if previous and ("Fatal Python error" in previous or "Traceback" in previous):
        return previous
    return None


def _log_napari_errors() -> None:
    """napari shows errors of the running program as pop-ups and replaces sys.excepthook; write them
    to crash.log as well, so a report from the lab contains the traceback."""
    fh = getattr(_install_crash_log, "handle", None)
    if fh is None:
        return
    try:
        from napari.utils.notifications import NotificationSeverity, notification_manager
    except Exception:
        return

    def log(note):
        try:
            if note.severity < NotificationSeverity.ERROR:
                return
            exc = getattr(note, "exception", None)
            text = ("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc is not None
                    else str(getattr(note, "message", note)))
            fh.write(f"{time.strftime('%H:%M:%S')} error\n{text}\n")
            fh.flush()
        except Exception:
            pass

    notification_manager.notification_ready.connect(log)


def _patch_napari() -> None:
    """Work arounds for napari 0.6.6 (0.4.0a1 feedback).

    - Labels thumbnails: when the view switches between 2D and 3D, napari sometimes computes the
      thumbnail of a 3D slice with a 2D zoom factor and raises "sequence argument must have length
      equal to input rank". The layer list with the thumbnails is hidden (D17), so they are skipped.
    - The triangulation of shapes warns "'where' used without 'out'" for lines. Harmless, hidden.
    """
    import warnings

    from napari.layers import Labels

    Labels._update_thumbnail = lambda self: None
    warnings.filterwarnings("ignore", message=".*'where' used without 'out'.*")
    warnings.filterwarnings("ignore", message=".*Contours are not displayed during 3D rendering.*")


def run_gui(argv=None, smoke_report: str | None = None) -> int:
    import os

    os.environ.setdefault("QT_API", "pyside6")
    previous_crash = _install_crash_log()
    import napari
    from qtpy.QtGui import QFont

    from localization_tool.gui import theme

    _patch_napari()

    viewer = napari.Viewer(title=f"{APP_NAME} {__version__}", show=False)
    _log_napari_errors()
    app = napari.qt.get_qapp()
    from qtpy.QtGui import QIcon

    from localization_tool.paths import resource

    icon = QIcon(str(resource("icon.png")))
    app.setWindowIcon(icon)
    viewer.window._qt_window.setWindowIcon(icon)
    font = QFont(theme.FONT_FAMILY)
    font.setPointSizeF(theme.pt("body", prefs.font_scale()))
    app.setFont(font)
    viewer.theme = theme.register_napari_theme()
    window = MainWindow(viewer)
    window.setup()
    window.install_close_guard()
    args = [a for a in (argv or [])[1:] if not a.startswith("-")]
    if smoke_report or (args and Path(args[0]).exists()):
        window.show_analysis_window()
    else:
        window.show_start()
    if previous_crash and not smoke_report:
        try:
            crash_log_path().with_name("crash_previous.log").write_text(previous_crash, encoding="utf-8")
        except OSError:
            pass
        window.message("The program ended unexpectedly last time. The details are in\n"
                       f"{crash_log_path().with_name('crash_previous.log')}\nPlease send this file along.")

    if args and Path(args[0]).exists():
        window.add_images([Path(args[0])], open_first=True)
    result = {"code": 0}
    if smoke_report:
        from qtpy.QtCore import QTimer

        def smoke():
            from localization_tool.gui.perf import run_performance_test
            try:
                window.load_example()
                text = run_performance_test(window)
                Path(smoke_report).write_text(text + "\nRESULT: PASSED\n", encoding="utf-8")
            except Exception:
                Path(smoke_report).write_text(traceback.format_exc() + "\nRESULT: FAILED\n", encoding="utf-8")
                result["code"] = 1
            viewer.close()

        QTimer.singleShot(1500, smoke)
    napari.run()
    return result["code"]


class MainWindow:
    """Owns the analysis (session), the state of the current cell, the napari layers and the panel."""

    EXCLUDE_LAYER = "Excluded"
    CELL_LAYER = "Cell edge"
    LABEL_LAYER = "Compartments"
    FRONT_LAYER = "Front direction"

    def __init__(self, viewer):
        self.viewer = viewer
        self.state = AppState()
        self.params, blind = prefs.load_defaults()
        self.session = project_file.AnalysisFile(params=self.params, blind=blind)
        if prefs.load_vocabulary():
            self.session.vocabulary = prefs.load_vocabulary()
        self.session_dir: Path | None = None
        self.pending: dict[tuple[str, str], dict] = {}     # control image and masks not yet written
        self.panels = load_panels(panels_file())
        self._image_layers = []
        self._target_layers: dict[str, object] = {}
        self._worker = None
        self._batch = None
        self._saving = False
        self._save_again = False
        self._save_worker = None
        self._writing: dict = {}                           # pending items of a save still being written
        self._surfaces: dict[str, object] = {}            # 3D view: smoothed mask surfaces
        self._surface_key = None
        self._drawing = False
        self._after_load = None
        self._restore_masks = None
        self.start = None                                  # start window (D59), created in show_start

    # ------------------------------------------------------------------ setup
    def install_close_guard(self) -> None:
        """Ask before the window closes while cells are unsaved, a save or a batch is running (the
        File > Quit menu and the window's close button both end here)."""
        from qtpy.QtCore import QEvent, QObject

        try:
            from napari.settings import get_settings

            settings = get_settings()
            settings._save_on_change = False      # only for this session, other napari installs keep theirs
            settings.application.confirm_close_window = False             # our question replaces napari's
        except Exception:
            pass
        main = self

        class Guard(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close and not main.confirm_close():
                    event.ignore()
                    return True
                return False

        self._close_guard = Guard()
        self.viewer.window._qt_window.installEventFilter(self._close_guard)

    def confirm_close(self) -> bool:
        """True if the program may close now."""
        from qtpy.QtWidgets import QApplication, QMessageBox

        if self._batch is not None:
            if not self.ask("A calculation is still running. Stop it and quit?", "Stop and quit"):
                return False
            self._batch.quit()
        if self._unsaved():
            n = (len(self.pending) if self.session_dir is not None else
                 sum(1 for im in self.session.images for c in im.cells if c.status != batch.EXCLUDED))
            box = QMessageBox(self.viewer.window._qt_window)
            box.setWindowTitle(APP_NAME)
            box.setText(f"{n} cell(s) of this analysis are not saved yet.")
            save = box.addButton("Save ...", QMessageBox.AcceptRole)
            box.addButton("Quit without saving", QMessageBox.DestructiveRole)
            cancel = box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(save)
            box.exec_()
            if box.clickedButton() is cancel:
                return False
            if box.clickedButton() is save and self.save_results() is None:
                return False
        t0 = time.perf_counter()
        while self._saving and time.perf_counter() - t0 < 60:        # let the last automatic save finish
            QApplication.processEvents()
            time.sleep(0.05)
        return True

    def setup(self) -> None:
        from localization_tool.gui import theme

        _patch_napari()
        from localization_tool.gui.workflow import WorkflowPanel

        self.panel = WorkflowPanel(self)
        theme.style(self.panel)
        self.dock = self.viewer.window.add_dock_widget(self.panel, name="Workflow", area="right")
        self._reduce_ui()
        self._place_view_controls()
        self.panel.refresh()

    def _place_view_controls(self) -> None:
        """2D/3D, zoom and fit sit on the image in the lower left corner (D68): they change only the view."""
        from qtpy.QtCore import QEvent, QObject

        from localization_tool.gui import theme

        canvas = self.viewer.window._qt_viewer.canvas.native
        bar = self.panel.view_row
        bar.setParent(canvas)
        theme.style(bar)
        bar.adjustSize()

        def place():
            bar.adjustSize()
            bar.move(12, max(0, canvas.height() - bar.height() - 12))
            bar.raise_()

        class _Follow(QObject):
            def eventFilter(self, obj, event):  # noqa: N802
                if event.type() in (QEvent.Resize, QEvent.Show):
                    place()
                return False

        self._view_follow = _Follow(canvas)
        canvas.installEventFilter(self._view_follow)
        bar.show()
        place()

    def _reduce_ui(self) -> None:
        """Hide everything of napari the workflow does not need (D17)."""
        win = self.viewer.window
        qv = win._qt_viewer
        for name in ("dockLayerList", "dockLayerControls", "dockConsole"):
            dock = getattr(qv, name, None)
            if dock is not None:
                dock.setVisible(False)
                try:
                    dock.toggleViewAction().setVisible(False)
                except Exception:
                    pass
        from qtpy.QtWidgets import QDockWidget, QWidget
        try:
            # napari rebuilds its title bar whenever the dock becomes visible
            self.dock.visibilityChanged.disconnect(self.dock._on_visibility_changed)
        except Exception:
            pass
        self.dock.setTitleBarWidget(QWidget())
        try:
            self.dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        except Exception:
            pass
        try:
            qv.set_welcome_visible(False)
        except Exception:
            pass
        # dropped files go through our reader, not through napari's
        qv._open_from_list_of_urls_data = lambda urls, stack=False, choose_plugin=False: self._open_dropped(urls)
        self.viewer.text_overlay.text = ("Add a folder or images with the buttons on the right, or drop .czi files here.\n"
                                         "Help > How to use explains the steps.")
        self.viewer.text_overlay.position = "top_center"
        self.viewer.text_overlay.font_size = 14
        self.viewer.text_overlay.color = "#9AA4AF"
        self.viewer.text_overlay.visible = True
        bar = win._qt_window.menuBar()
        for action in bar.actions():
            action.setVisible(False)
        file_menu = bar.addMenu("File")
        file_menu.addAction("Home (start page)", self.show_start)
        file_menu.addSeparator()
        file_menu.addAction("New analysis", self.new_analysis)
        file_menu.addAction("Open analysis ...", self.panel._open_analysis)
        file_menu.addAction("Add folder ...", self.panel._add_folder)
        file_menu.addAction("Add images ...", self.panel._add_images)
        file_menu.addSeparator()
        file_menu.addAction("Analyze all images", lambda: self.run_all("new"))
        file_menu.addAction("Apply settings to all cells", lambda: self.run_all("all"))
        file_menu.addSeparator()
        file_menu.addAction("Save results", self.save_results)
        file_menu.addAction("View and export cells ...", self.open_cell_viewer)
        file_menu.addSeparator()
        file_menu.addAction("Quit", win._qt_window.close)
        settings_menu = bar.addMenu("Settings")
        settings_menu.addAction("Settings of this analysis ...", self.open_settings)
        settings_menu.addAction("Default settings for new analyses ...", self.open_program_settings)
        help_menu = bar.addMenu("Help")
        help_menu.addAction("How to use (English)", lambda: self.open_help("en"))
        help_menu.addAction("Bedienung (Deutsch)", lambda: self.open_help("de"))
        help_menu.addAction("Methods text ...", self.open_methods)
        help_menu.addAction("Practice with a synthetic cell", self.load_example)
        help_menu.addSeparator()
        help_menu.addAction("Report a problem ...", self.open_feedback)
        help_menu.addAction("Performance test ...", self.open_performance_test)
        help_menu.addAction("Show settings and crash log folder", self._show_log_folder)
        help_menu.addAction(f"About {APP_NAME}", self._about)
        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        """Keyboard (D62): Ctrl+Right / Ctrl+Left next and back, in step 5 Ctrl+Enter accepts and
        Ctrl+Shift+Right opens the next image. Single letters are not used: napari gives them to its
        layer tools while the image has the focus."""
        from qtpy.QtGui import QKeySequence
        from qtpy.QtWidgets import QShortcut

        win = self.viewer.window._qt_window
        p = self.panel

        def nav(delta):
            if delta > 0 and not p.next_btn.isEnabled():
                return
            if delta < 0 and not p.back_btn.isEnabled():
                return
            p.go(p.step + delta)

        def in_results(fn):
            if p.step == 4:
                fn()

        self._shortcuts = []
        for keys, fn in (("Ctrl+Right", lambda: nav(1)), ("Ctrl+Left", lambda: nav(-1)),
                         ("Ctrl+Return", lambda: in_results(lambda: p.accept_btn.isEnabled() and self.accept_cell())),
                         ("Ctrl+Enter", lambda: in_results(lambda: p.accept_btn.isEnabled() and self.accept_cell())),
                         ("Ctrl+Shift+Right", lambda: in_results(self.next_image))):
            sc = QShortcut(QKeySequence(keys), win)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    # ------------------------------------------------------------------ start window, about, methods
    def show_start(self) -> None:
        """The start page (D59). The image window is hidden, the analysis stays open."""
        from localization_tool.gui.start import StartWindow

        if self.start is None:
            self.start = StartWindow(
                new_analysis=self._start_new, open_analysis=self._start_open, open_guide=lambda: self.open_help("en"),
                open_methods=self.open_methods, open_settings=self.open_program_settings,
                open_example=self._start_example, open_about=self._about, dropped=self._start_dropped,
                recent=prefs.recent_analyses, continue_current=self.show_analysis_window,
                confirm_close=self.confirm_close, open_viewer=lambda: self.open_cell_viewer(ask=True))
            icon = self.viewer.window._qt_window.windowIcon()
            self.start.setWindowIcon(icon)
        name = self.session_dir.name if self.session_dir else (f"{len(self.session.images)} image(s), not saved yet"
                                                               if self.session.images else "")
        self.start.refresh(has_analysis=bool(self.session.images), name=name)
        self.viewer.window._qt_window.hide()
        self.start.show()
        self.start.raise_()
        self.start.activateWindow()

    def open_cell_viewer(self, ask: bool = False) -> None:
        """Cell viewer (D71) for the open analysis once it is saved, or for an analysis file chosen now."""
        from localization_tool.gui.cellviewer import open_viewer

        path = None
        if not ask and self.session_dir is not None and (self.session_dir / ANALYSIS_NAME).exists():
            self.save_results(notify=False)
            path = self.session_dir / ANALYSIS_NAME
        elif not ask:
            self.message("Save the results first (step 5, Save results ...). The cell viewer shows saved analyses.")
            return
        parent = self.start if ask and self.start is not None else self.viewer.window._qt_window
        v = open_viewer(path, parent)
        if v is not None:
            v.setWindowIcon(self.viewer.window._qt_window.windowIcon())
            self._viewers = [w for w in getattr(self, "_viewers", []) if w.isVisible()] + [v]

    def show_analysis_window(self) -> None:
        win = self.viewer.window._qt_window
        if self.start is not None:
            self.start.hide()
        win.showMaximized()
        win.raise_()
        win.activateWindow()

    def _start_new(self) -> None:
        if self.session.images and not self.ask("Start a new analysis? The open analysis is closed"
                                                + (" and accepted cells that are not saved are lost."
                                                   if self._unsaved() else "."), "Start new analysis"):
            return
        self.pending = {}                     # already confirmed above
        self.new_analysis(confirm=False)
        self.show_analysis_window()

    def _start_open(self, path: Path | None) -> None:
        from qtpy.QtWidgets import QFileDialog

        if path is None:
            name, _ = QFileDialog.getOpenFileName(self.start, "Open analysis", str(Path.home()), "Analysis (*.lta.json)")
            if not name:
                return
            path = Path(name)
        if self.open_analysis(path):
            self.show_analysis_window()

    def _start_example(self) -> None:
        self.show_analysis_window()
        self.load_example()

    def _start_dropped(self, paths: list[Path]) -> None:
        """Files dropped on the start page: with an analysis open, ask whether they belong to it."""
        from qtpy.QtWidgets import QMessageBox

        is_analysis = any(p.name.endswith(".lta.json") for p in paths)
        if self.session.images and not is_analysis:
            box = QMessageBox(self.start)
            box.setWindowTitle(APP_NAME)
            box.setText("An analysis is open. Add the dropped images to it, or start a new analysis with them?")
            add = box.addButton("Add to the open analysis", QMessageBox.AcceptRole)
            new = box.addButton("New analysis", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is new:
                if self._unsaved() and not self.ask("Accepted cells that are not saved are lost. Continue?",
                                                    "Discard and continue"):
                    return
                self.pending = {}
                self.new_analysis(confirm=False)
            elif box.clickedButton() is not add:
                return
        self.show_analysis_window()
        self._open_dropped_paths(paths)

    def open_methods(self) -> None:
        from localization_tool.core.methods_text import summary_from_analysis
        from localization_tool.gui.start import MethodsDialog

        if self.session.images and any(c.results for im in self.session.images for c in im.cells):
            summary, source = summary_from_analysis(self.session), "the open analysis"
            params = self.params
        else:
            summary, source = None, "the default settings (open or analyse cells to fill in the numbers)"
            params = self.params if self.session.images else prefs.load_defaults()[0]
        parent = self.start if (self.start is not None and self.start.isVisible()) else self.viewer.window._qt_window
        MethodsDialog(parent, params, summary, source).exec_()

    def open_program_settings(self) -> None:
        """Default settings for new analyses (start page). Saved in the user's settings folder."""
        from localization_tool.gui.settings import SettingsDialog

        params, blind = prefs.load_defaults()
        parent = self.start if (self.start is not None and self.start.isVisible()) else self.viewer.window._qt_window
        vocab = prefs.load_vocabulary() or list(self.session.vocabulary)
        dlg = SettingsDialog(parent, params, [], vocab, None, blind=blind, program_defaults=True)
        if dlg.exec_() and dlg.result_params is not None:
            prefs.save_defaults(dlg.result_params, dlg.result_blind, dlg.result_vocabulary)

    def _open_dropped(self, urls) -> None:
        self._open_dropped_paths([Path(u.toLocalFile()) for u in urls if u.isLocalFile()])

    def _open_dropped_paths(self, paths: list[Path]) -> None:
        folders = [p for p in paths if p.is_dir()]
        for f in folders:
            self.add_folder(f)
        if folders:
            return
        images = [p for p in paths if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if images:
            self.add_images(images, open_first=True)
        elif any(p.name.endswith(".lta.json") for p in paths):
            self.open_analysis(next(p for p in paths if p.name.endswith(".lta.json")))
        else:
            self.message("Please drop .czi or .tif files, or an analysis file (.lta.json).")

    def _show_log_folder(self) -> None:
        self.message(f"Settings, panels and the crash log are in:\n{user_config_dir()}")

    def _about(self) -> None:
        from localization_tool.gui.start import AboutDialog

        parent = self.start if (self.start is not None and self.start.isVisible()) else self.viewer.window._qt_window
        AboutDialog(parent).exec_()

    def open_help(self, language: str = "en") -> None:
        from localization_tool.gui.help import HelpDialog

        self._help = HelpDialog(self.viewer.window._qt_window, language)
        self._help.show()

    def open_feedback(self) -> None:
        from localization_tool.gui.feedback import FeedbackDialog

        FeedbackDialog(self.viewer.window._qt_window).exec_()

    def open_performance_test(self) -> None:
        from localization_tool.gui.perf import PerformanceDialog

        PerformanceDialog(self).exec_()

    def message(self, text: str, title: str = APP_NAME, error: bool = False) -> None:
        from qtpy.QtWidgets import QMessageBox

        box = QMessageBox.critical if error else QMessageBox.information
        box(self.viewer.window._qt_window, title, text)

    def ask(self, text: str, yes: str = "Yes", no: str = "Cancel") -> bool:
        """Question with buttons named after the action ("Start new analysis" instead of "Yes"). The
        safe answer is the default, so Enter never discards anything."""
        from qtpy.QtWidgets import QMessageBox

        box = QMessageBox(self.viewer.window._qt_window)
        theme.style(box)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Question)
        box.setText(text)
        ok = box.addButton(yes, QMessageBox.AcceptRole)
        ok.setProperty("role", "primary")
        cancel = box.addButton(no, QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is ok

    def status(self, text: str) -> None:
        self.viewer.status = text

    # ------------------------------------------------------------------ workflow rules
    def can_leave(self, step: int) -> bool:
        s = self.state
        if step == 0:
            return s.full is not None
        if step == 1:
            return s.full is not None and not [p for p in validate(s.assignments) if not p.startswith("No Nucleus")]
        if step == 2:
            return s.full is not None and s.cell is not None and not s.computing and s.cell_key == self._cell_key()
        if step == 3:
            return s.result is not None and not s.computing and s.result_key == self._result_key()
        return False

    def blocker(self, step: int) -> str:
        """Why the given step cannot be left forward (tooltip of a locked step, D62)."""
        s = self.state
        if s.computing:
            return "Wait until the calculation has finished."
        if step == 0 and s.full is None:
            return "Open an image first: add a folder or images, then double-click an image in the list."
        if step == 1:
            problems = [p for p in validate(s.assignments) if not p.startswith("No Nucleus")]
            if problems:
                return problems[0]
        if step == 2 and (s.cell is None or s.cell_key != self._cell_key()):
            return "Open step 3 first: the cell edge is computed there."
        if step == 3 and (s.result is None or s.result_key != self._result_key()):
            return "Open step 4 first: the compartments are computed there."
        return ""

    def enter_step(self, step: int, previous: int) -> None:
        if not hasattr(self, "panel"):          # the panel calls this once while it is built
            return
        self.refresh_lists()
        if self.state.full is None:
            return
        if previous == 2 and step != 2:
            if self._drawing:
                self.apply_exclusions()
        self._set_visible(self.LABEL_LAYER, step == 3)
        self._set_visible(self.FRONT_LAYER, False)
        for layer in self._target_layers.values():
            layer.visible = False
        self._hide_surfaces()
        self._set_visible(self.CELL_LAYER, step in (2, 3))
        if step == 2:
            self._update_box()
            self.panel.cell_error.setText("")
            if self.state.cell is None or self.state.cell_key != self._cell_key():
                self.compute_cell()
            else:
                self._show_cell()
        if step == 3:
            if self.state.result is None or self.state.result_key != self._result_key():
                self.compute()
            else:
                self.show_compartments(self.state.result)
        if step == 4 and self.state.result is not None:
            self.panel.show_results(self.state.result)
            self.panel.accept_btn.setEnabled(not self.state.accepted_current)
            im = self._current_image()
            spec = next((c for c in im.cells if c.name == self.state.editing), None) if im else None
            done = "Excluded" if spec is not None and spec.status == batch.EXCLUDED else "Accepted"
            self.panel.accept_btn.setText(done if self.state.accepted_current else "Accept cell")
            checked = spec is not None and str(spec.status).startswith("accepted")
            self.panel.accept_btn.setText("Save changes to cell" if checked else "Accept cell")

    def _crop_key(self) -> tuple:
        b = self.state.box
        ex = tuple(tuple(map(tuple, np.round(np.asarray(p), 3))) for p in self.state.exclusions)
        return (b.z0, b.z1, b.y0, b.y1, b.x0, b.x1, ex) if b else (None, ex)

    def _cell_key(self) -> tuple:
        return (self._crop_key(), round(self.panel.cell_factor.value(), 4),
                tuple((a.role, a.label) for a in self.state.assignments))

    def _result_key(self) -> tuple:
        f = self.panel.factors()
        return (self._cell_key(), id(self.state.cell), tuple(round(f[k], 4) for k in sorted(f)))

    # ------------------------------------------------------------------ analysis (session)
    def new_analysis(self, confirm: bool = True) -> None:
        if confirm and self._unsaved() and not self.ask("Start a new analysis? Accepted cells that are not saved "
                                                        "are lost.", "Start new analysis"):
            return
        self.params, blind = prefs.load_defaults()
        self.session = project_file.AnalysisFile(params=self.params, blind=blind,
                                                 vocabulary=prefs.load_vocabulary() or list(self.session.vocabulary))
        self.session_dir = None
        self.pending = {}
        self._clear_image()
        self.panel.show_analysis(None, "")
        self.panel.save_info.setText("")
        self.refresh_lists()

    def _clear_image(self) -> None:
        """No image open: steps 2 to 5 are locked until an image of the (new) list is opened."""
        self.state = AppState()
        self._load_seq = getattr(self, "_load_seq", 0) + 1      # an image still loading is dropped
        self._after_load = None
        self.panel.set_busy(False)
        self.viewer.layers.clear()
        self._image_layers, self._target_layers, self._surfaces = [], {}, {}
        self._surface_key = None
        self._drawing = False
        self.viewer.text_overlay.text = ("Add a folder or images with the buttons on the right, or drop .czi "
                                         "files here.\nHelp > How to use explains the steps.")
        self.viewer.text_overlay.visible = True
        self.panel.accepted_info.setText("")
        self.panel.show_info(None, 0.0, "", "")
        self.panel.go(0)

    def _unsaved(self) -> bool:
        return bool(self.pending) or (self.session_dir is None and any(im.cells for im in self.session.images))

    def open_analysis(self, path: Path) -> bool:
        try:
            af = project_file.load(path)
        except ValueError as exc:
            self.message(str(exc), error=True)
            return False
        if self._unsaved() and not self.ask("Open another analysis? Accepted cells that are not saved are lost.",
                                            "Open other analysis"):
            return False
        prefs.add_recent(path)
        self.session = af
        self.params = af.params
        self.session_dir = path.parent
        self.pending = {}
        self._clear_image()
        self.panel.show_analysis(self.session_dir, self._short_count())
        self.panel.save_info.setText("")
        self.refresh_lists()
        self.status(f"Opened {path}")
        old = [c for im in af.images for c in im.cells
               if c.status != batch.EXCLUDED and c.results and c.results.get("results_version") != batch.RESULTS_VERSION]
        if old and self.ask(f"{len(old)} cell(s) were computed by an older version of the program, partly with "
                            "other definitions (e.g. cytoplasm without actin). Compute them again now?\n\n"
                            "Removed regions are kept. Cells whose outlines stay exactly the same keep "
                            "their status, all others are marked 'to check'.", "Compute again", "Later"):
            self.run_all("all")
        return True

    def _short_count(self) -> str:
        """For the header line: '12 images · 8 cells' (empty without images)."""
        n_img = len(self.session.images)
        if not n_img:
            return ""
        n_cells = sum(1 for im in self.session.images for c in im.cells if c.status != batch.EXCLUDED)
        return f"{n_img} image{'s' if n_img != 1 else ''} · {n_cells} cell{'s' if n_cells != 1 else ''}"

    def _count_text(self) -> str:
        cells = [c for im in self.session.images for c in im.cells]
        n_ex = sum(1 for c in cells if c.status == batch.EXCLUDED)
        todo = sum(1 for c in cells if c.status == batch.AUTOMATIC)
        text = f"{len(self.session.images)} image(s), {len(cells) - n_ex} cell(s)"
        if todo:
            text += f" ({todo} to check)"
        if n_ex:
            text += f", {n_ex} excluded"
        return text + "."

    def add_images(self, paths: list[Path], open_first: bool = False) -> None:
        known = {str(im.path) for im in self.session.images}
        first = None
        for p in paths:
            if str(p) in known:
                idx = next(i for i, im in enumerate(self.session.images) if str(im.path) == str(p))
            else:
                self.session.images.append(project_file.ImageSpec(
                    path=p, experiment=suggest_experiment(p),
                    condition=suggest_condition(p, self.session.vocabulary)))
                idx = len(self.session.images) - 1
                known.add(str(p))
            first = idx if first is None else first
        self.session.assign_codes()
        self.refresh_lists()
        if open_first and first is not None:
            if self.session.blind:           # not the first of the folder: that would follow the folder order
                added = {str(p) for p in paths}
                first = next(i for i in self.session.blind_order() if str(self.session.images[i].path) in added)
            self.open_image_index(first)

    def add_folder(self, folder: Path, sync: bool = False) -> None:
        """All images of a folder and its subfolders (core.files.find_images). Runs in the background:
        on network or cloud folders reading every CZI header can take a while."""
        from localization_tool.core.files import find_images

        def done(res):
            self.panel.set_busy(False)
            self._folder_found(folder, *res)

        def failed(exc):
            self.panel.set_busy(False)
            self.message(f"The folder could not be searched:\n{exc}", error=True)

        self.panel.set_busy(True, f"Searching {folder.name} ...")
        self._run(find_images, (folder,), {}, done, failed, sync)

    def _folder_found(self, folder: Path, images: list, skipped: list) -> None:
        known = {str(im.path) for im in self.session.images}
        new = [p for p in images if str(p) not in known]
        if not images:
            self.message(f"No image files found in\n{folder}" +
                         (f"\n\n{len(skipped)} file(s) skipped." if skipped else ""))
            return
        self.add_images(new, open_first=bool(new) and self.state.full is None)
        text = f"{len(new)} image(s) added from {folder.name}."
        if len(images) > len(new):
            text += f" {len(images) - len(new)} were already in the list."
        if skipped:
            lines = [f"{p.relative_to(folder)}: {r}" for p, r in skipped[:15]]
            more = f"\n... and {len(skipped) - 15} more" if len(skipped) > 15 else ""
            text += f"\n\nSkipped:\n" + "\n".join(lines) + more
        self.message(text)

    def exclude_image(self, index: int, reason: str) -> None:
        """Leave a whole image out, with a reason (sheet Excluded)."""
        if not (0 <= index < len(self.session.images)):
            return
        im = self.session.images[index]
        if any(c.status != batch.EXCLUDED for c in im.cells) and \
                not self.ask(f"{self.session.shown_name(im)} has cells. Exclude all of them?", "Exclude all"):
            return
        if im.cells:
            for c in im.cells:
                c.status, c.note = batch.EXCLUDED, reason
        else:
            im.cells = [project_file.CellSpec(name="cell1", status=batch.EXCLUDED, note=reason)]
        self._autosave()
        self.refresh_lists()

    def exclude_cell(self, reason: str) -> None:
        """Step 5: leave the current cell out, with a reason."""
        s = self.state
        im = self._current_image()
        if im is None:
            return
        name = s.editing or self._next_cell_name(im)
        spec = next((c for c in im.cells if c.name == name), None)
        if spec is None:
            spec = project_file.CellSpec(name=name, box=s.box, exclusions=[p.tolist() for p in s.exclusions])
            im.cells = sorted(im.cells + [spec], key=lambda c: c.name)
        spec.status, spec.note = batch.EXCLUDED, reason
        s.editing = name
        s.accepted_current = True
        self.panel.accept_btn.setEnabled(False)
        self.panel.accept_btn.setText("Excluded")
        self.panel.set_accepted_text(f"{self.session.shown_name(im)} / {name} excluded: {reason}", excluded=True)
        self._autosave()
        self.refresh_lists()

    def remove_image(self, index: int) -> None:
        if not (0 <= index < len(self.session.images)):
            return
        im = self.session.images[index]
        if im.cells and not self.ask(f"{self.session.shown_name(im)} has {len(im.cells)} accepted cell(s). "
                                     "Remove them as well?", "Remove image and cells"):
            return
        for c in im.cells:
            self.pending.pop((str(im.path), c.name), None)
        del self.session.images[index]
        if self.state.index == index:
            self.state.index = -1
        elif self.state.index > index:
            self.state.index -= 1
        self._autosave()
        self.refresh_lists()

    def open_image_index(self, index: int, after=None) -> None:
        if not (0 <= index < len(self.session.images)):
            return
        if self.state.computing:
            self.message("Please wait until the current calculation has finished.")
            return
        im = self.session.images[index]
        if _is_example(im.path):
            self.load_example()
            if after:
                after()
            return
        self._after_load = after
        self.load_path(im.path, index)

    def set_meta(self, experiment: str, condition: str) -> None:
        self.state.experiment, self.state.condition = experiment, condition
        im = self._current_image()
        if im is not None:
            im.experiment, im.condition = experiment, condition

    def _current_image(self) -> project_file.ImageSpec | None:
        i = self.state.index
        return self.session.images[i] if 0 <= i < len(self.session.images) else None

    def refresh_lists(self) -> None:
        cur = self.state.index
        entries = []
        for i in self.session.blind_order():
            im = self.session.images[i]
            cells = [c for c in im.cells if c.status != batch.EXCLUDED]
            n = len(cells)
            todo = sum(1 for c in cells if c.status == batch.AUTOMATIC)
            text = ""
            if n:
                text = f"{n} cell{'s' if n != 1 else ''}" + (f", {todo} to check" if todo else "")
            elif im.cells:
                text = "excluded"
            entries.append((self.session.shown_name(im), text, i == cur))
        self.panel.show_images(entries)
        entries = []
        for i in self.session.blind_order():
            im = self.session.images[i]
            hidden = self._hidden(im)
            for c in im.cells:
                cond = f"  [{im.condition}]" if im.condition and not hidden else ""
                why = f" ({c.note})" if c.status == batch.EXCLUDED and c.note else ""
                status = STATUS_TEXT.get(c.status, c.status)
                name = self.session.shown_name(im) if hidden else im.path.stem
                entries.append(f"{status}{why}: {name} / {c.name}{cond}")
        self.panel.show_cells(entries)
        self.panel.show_analysis(self.session_dir, self._short_count())

    def _hidden(self, im) -> bool:
        """Blind review hides name and condition of this image (D57)."""
        return im is not None and self.session.blind and bool(im.code) and not im.revealed

    def reveal_current(self) -> None:
        """'Show name': the current image is shown with its name from now on, noted in the analysis file."""
        im = self._current_image()
        if im is None or not self._hidden(im):
            return
        im.revealed = True
        self._autosave()
        s = self.state
        self.panel.show_info(s.full, s.load_seconds, s.experiment, s.condition, shown=im.path.name, hidden=False)
        self.refresh_lists()

    def _cell_entries(self) -> list[tuple[int, int]]:
        """(image index, cell index) per row of the cell list, in the order of the image list."""
        return [(i, j) for i in self.session.blind_order() for j in range(len(self.session.images[i].cells))]

    def image_index(self, row: int) -> int:
        """Image index for a row of the image list (sorted by code during blind review, D57)."""
        order = self.session.blind_order()
        return order[row] if 0 <= row < len(order) else -1

    # ------------------------------------------------------------------ loading
    def load_path(self, path: Path, index: int = -1) -> None:
        from napari.qt.threading import create_worker

        ims = self.session.images
        shown = self.session.shown_name(ims[index]) if 0 <= index < len(ims) else path.name   # D57
        self.status(f"Loading {shown} ...")
        self.panel.set_busy(True, f"Loading {shown} ...")
        t0 = time.perf_counter()
        self._load_seq = getattr(self, "_load_seq", 0) + 1
        token = self._load_seq

        def done(stack):
            if token != self._load_seq:          # another image was chosen meanwhile: that one wins
                return
            self.panel.set_busy(False)
            self.set_image(stack, time.perf_counter() - t0, path, index)
            after, self._after_load = self._after_load, None
            if after:
                after()

        def failed(exc):
            if token != self._load_seq:
                return
            self.panel.set_busy(False)
            self._after_load = None
            msg = str(exc) if isinstance(exc, ReaderError) else f"{shown} could not be loaded:\n{exc}"
            if shown != path.name:
                msg = msg.replace(str(path), shown).replace(path.name, shown)
            self.message(msg, "Could not open file", error=True)
            self.status("")

        worker = create_worker(read_image, path, _start_thread=False)
        worker.returned.connect(done)
        worker.errored.connect(failed)
        self._worker = worker
        worker.start()

    def load_example(self) -> None:
        from localization_tool.core.synthetic import make_practice_cell

        if not any(_is_example(im.path) for im in self.session.images):
            self.session.images.append(project_file.ImageSpec(path=EXAMPLE_PATH, experiment="Example"))
        index = next(i for i, im in enumerate(self.session.images) if _is_example(im.path))
        t0 = time.perf_counter()
        stack = make_practice_cell()
        im = self.session.images[index]
        if im.assignments is None:
            roles = [("Target", "TTN", DISPLAY_COLORS["Magenta"]), ("Target", "VIM", DISPLAY_COLORS["Orange"]),
                     ("Actin", "Actin", DISPLAY_COLORS["Green"]), ("Nucleus", "Nucleus", DISPLAY_COLORS["Blue"])]
            im.assignments = [ChannelAssignment(*r) for r in roles]
        self.set_image(stack, time.perf_counter() - t0, None, index)

    def set_image(self, stack: ImageStack, load_seconds: float, path: Path | None, index: int = -1) -> None:
        if self.state.computing:
            self.message("Please wait until the current calculation has finished.")
            return
        im = self.session.images[index] if 0 <= index < len(self.session.images) else None
        self.state = AppState(full=stack, shown=stack, load_seconds=load_seconds, path=path, index=index)
        self.state.experiment = (im.experiment if im else "") or (path.parent.name if path else "Example")
        self.state.condition = im.condition if im else ""
        if im is not None and im.assignments and len(im.assignments) == stack.n_channels:
            self.state.assignments = [ChannelAssignment(**vars(a)) for a in im.assignments]
            note = "Roles as saved in this analysis."
        else:
            panel = find_panel(self.panels, stack.channel_signature())
            if panel is not None and len(panel.assignments) == stack.n_channels:
                self.state.assignments = [ChannelAssignment(**vars(a)) for a in panel.assignments]
                note = f"Panel '{panel.name}' applied."
            else:
                self.state.assignments = suggest_assignments(stack.channels, path.stem if path else "",
                                                             tuple(self.session.vocabulary or ()))
                note = ("No saved panel for these channels. Roles guessed from the file name, please check "
                        "them and save a panel.")
        if im is not None:
            im.experiment, im.condition = self.state.experiment, self.state.condition
        self.panel.set_factors({}, self.base_factors())
        self.viewer.text_overlay.visible = False
        self._build_layers(stack)
        self._update_box()
        self.panel.show_info(stack, load_seconds, self.state.experiment, self.state.condition,
                             shown=self.session.shown_name(im) if im is not None else None,
                             hidden=self._hidden(im))
        self.panel.populate_channels(stack, self.state.assignments, note)
        self.panel.set_targets([a.label for a in self.state.assignments if a.role == "Target"])
        self.panel.accepted_info.setText("")
        shown = self.session.shown_name(im) if im is not None else stack.display_name
        self.status(f"{shown} loaded in {load_seconds:.1f} s")
        self.refresh_lists()
        self.panel.go(0)                      # stay in step 1: the user checks the image and clicks Next (D62)

    # ------------------------------------------------------------------ layers
    def _build_layers(self, stack: ImageStack) -> None:
        """All layers of one image, created once."""
        self.viewer.layers.clear()
        self._image_layers = []
        self._target_layers = {}
        self._surfaces = {}
        self._surface_key = None
        for ch, a in zip(stack.channels, self.state.assignments):
            data = stack.data[ch.index]
            lo, hi = _contrast(data)
            layer = self.viewer.add_image(
                data, name=a.label, colormap=_colormap(a), blending="additive", contrast_limits=(lo, hi),
                scale=stack.voxel_um, interpolation2d="linear", interpolation3d="linear",   # display only (D19)
                metadata={"role": a.role, "channel": ch.index, "auto_hi": hi, "auto_lo": lo},
                visible=a.role != "Ignore")
            self._image_layers.append(layer)
        ex = self.viewer.add_shapes(name=self.EXCLUDE_LAYER, ndim=2, scale=stack.voxel_um[1:], edge_width=2,
                                    **EXCLUDE_DONE)
        ex.events.data.connect(self._exclusions_changed)
        cell = self.viewer.add_labels(np.zeros(stack.shape_zyx, np.uint8), name=self.CELL_LAYER, opacity=0.9,
                                      scale=stack.voxel_um, visible=False)
        cell.colormap = _label_cmap({1: (1, 1, 1, 1)})
        cell.contour = 1
        labels = self.viewer.add_labels(np.zeros(stack.shape_zyx, np.uint8), name=self.LABEL_LAYER,
                                        opacity=0.7, scale=stack.voxel_um, visible=False)
        labels.colormap = _label_cmap(LABEL_COLORS)
        labels.contour = 1
        self.viewer.add_shapes(name=self.FRONT_LAYER, ndim=2, scale=stack.voxel_um[1:], edge_width=3,
                               edge_color="#FF8C00", face_color=[0, 0, 0, 0], visible=False)
        self.viewer.dims.ndisplay = 2
        self.viewer.reset_view()
        if stack.shape_zyx[0] > 1:
            self.viewer.dims.set_current_step(0, stack.shape_zyx[0] // 2)
        self._align_layers()
        self._style_dims(stack)
        self._select_image_layer()

    def _style_dims(self, stack: ImageStack) -> None:
        """Plane slider under the image: readable plane number, axis named z, no play button (UI review)."""
        try:
            from qtpy.QtWidgets import QWidget

            from localization_tool.gui import theme

            if stack.shape_zyx[0] > 1:
                self.viewer.dims.axis_labels = ("z", "y", "x")
            dims = self.viewer.window._qt_viewer.dims
            dims.setStyleSheet(f"#slice_label, #axis_label {{ color: {theme.TOKENS['text_muted']}; }}")
            for w in dims.findChildren(QWidget):
                if type(w).__name__ == "QtPlayButton":
                    w.setVisible(False)
        except Exception:                  # napari internals: cosmetic only, never stop loading
            pass
        try:                               # napari's own status hints ("use <2> for transform", activity)
            bar = self.viewer.window._qt_window.statusBar()
            bar._activity_item.setVisible(False)
            bar.setHelpText("")
            bar.setHelpText = lambda text: None      # napari would set the layer tool hint again
        except Exception:
            pass

    def _align_layers(self) -> None:
        """Show each channel moved by the same whole voxels as in the analysis (D55), so the outlines
        drawn from the aligned channels fit the image on screen. Display only, the data stay unchanged."""
        from localization_tool.core.analysis import channel_offset

        full = self.state.full
        if full is None:
            return
        for layer, a in zip(self._image_layers, self.state.assignments):
            k = layer.metadata.get("channel", 0)
            ch = full.channels[k] if k < len(full.channels) else None
            off = channel_offset(self.params, a.label, ch.fluor if ch else None) or (0.0, 0.0, 0.0)
            steps = [int(round(-d / v)) if v > 0 else 0 for d, v in zip(off, full.voxel_um)]
            layer.translate = tuple(st * v for st, v in zip(steps, full.voxel_um))

    def _target_layer(self, label: str, color: str):
        name = f"Target mask {label}"
        layer = self._target_layers.get(label)
        if layer is None or layer not in self.viewer.layers:
            rgb = _hex_to_rgb(color)
            layer = self.viewer.add_labels(np.zeros(self.state.full.shape_zyx, np.uint8), name=name, opacity=0.9,
                                           scale=self.state.full.voxel_um, visible=False)
            layer.colormap = _label_cmap({1: (*rgb, 1)})
            layer.contour = 1
            self._target_layers[label] = layer
        return layer

    def _select_image_layer(self) -> None:
        if self._image_layers:
            self.viewer.layers.selection.active = self._image_layers[0]

    def image_layers(self):
        return [layer for layer in self._image_layers if layer in self.viewer.layers]

    def apply_assignments(self, assignments: list[ChannelAssignment]) -> None:
        """Update names, colours and visibility in place. No image layer is rebuilt."""
        self.state.assignments = assignments
        self.state.result = None
        self.state.cell_key = None
        for layer, a in zip(self._image_layers, assignments):
            if layer.name != a.label:
                layer.name = a.label
            layer.colormap = _colormap(a)
            layer.metadata["role"] = a.role
            layer.visible = a.role != "Ignore"
        self._align_layers()
        for layer in list(self._target_layers.values()) + list(self._surfaces.values()):
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._target_layers = {}
        self._surfaces = {}
        self._surface_key = None
        self.panel.set_targets([a.label for a in assignments if a.role == "Target"])
        self.panel.refresh()

    def set_brightness(self, layer, factor: float) -> None:
        lo, hi = layer.metadata.get("auto_lo", 0.0), layer.metadata.get("auto_hi", 1.0)
        layer.contrast_limits = (lo, max(lo + 1e-6, lo + (hi - lo) / factor))

    def set_ndisplay(self, n: int) -> None:
        if self._drawing and n == 3:
            self.apply_exclusions()
        if n == 3:                                   # contours exist only in 2D
            for layer in self.viewer.layers:
                if hasattr(layer, "contour") and layer.contour:
                    layer.contour = 0
        self.viewer.dims.ndisplay = n
        self._set_visible(self.EXCLUDE_LAYER, n == 2)        # the drawn shapes are 2D, they float in 3D
        if self.panel.step == 3:
            self.update_compartment_display()
        elif self.panel.step == 2:
            self._show_cell_3d(n == 3)
        self.panel.refresh()

    def reset_view(self) -> None:
        self.viewer.reset_view()

    def zoom(self, factor: float) -> None:
        self.viewer.camera.zoom = self.viewer.camera.zoom * factor

    def fit_cell(self) -> None:
        """Zoom to the cell (or the analysed region) with a small margin."""
        s = self.state
        if s.full is None or s.box is None:
            return
        b = s.box
        dz, dy, dx = s.full.voxel_um
        y0, y1, x0, x1 = b.y0, b.y1, b.x0, b.x1
        if s.cell is not None and s.cell.any():
            ys = np.flatnonzero(s.cell.any(axis=(0, 2)))
            xs = np.flatnonzero(s.cell.any(axis=(0, 1)))
            y0, y1, x0, x1 = b.y0 + ys[0], b.y0 + ys[-1] + 1, b.x0 + xs[0], b.x0 + xs[-1] + 1
        cy, cx = (y0 + y1) / 2 * dy, (x0 + x1) / 2 * dx
        center = list(self.viewer.camera.center)
        center[-2:] = [cy, cx]
        self.viewer.camera.center = tuple(center)
        size = self.viewer.window._qt_viewer.canvas.native.size()
        h_um, w_um = (y1 - y0) * dy * 1.15, (x1 - x0) * dx * 1.15
        if h_um > 0 and w_um > 0:
            self.viewer.camera.zoom = min(size.height() / h_um, size.width() / w_um)

    def save_panel(self, name: str) -> None:
        stack = self.state.full
        if stack is None:
            return
        panel = Panel(name=name, signature=stack.channel_signature(),
                      assignments=[ChannelAssignment(**vars(a)) for a in self.state.assignments])
        self.panels = upsert_panel(self.panels, panel)
        save_panels(panels_file(), self.panels)

    # ------------------------------------------------------------------ step 3: cell (D28, D29)
    def start_exclusion(self) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is None:
            return
        self.viewer.dims.ndisplay = 2
        layer.current_edge_color = EXCLUDE_EDIT["edge_color"]
        layer.current_face_color = EXCLUDE_EDIT["face_color"]
        if len(layer.data):
            layer.edge_color = EXCLUDE_EDIT["edge_color"]
            layer.face_color = EXCLUDE_EDIT["face_color"]
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon"
        self._drawing = True
        self.status("Click the corners around the part to remove, double-click to close the shape.")
        self.panel.refresh()

    def apply_exclusions(self) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is None:
            return
        layer.mode = "pan_zoom"
        if len(layer.data):
            layer.edge_color = EXCLUDE_DONE["edge_color"]
            layer.face_color = EXCLUDE_DONE["face_color"]
        layer.current_edge_color = EXCLUDE_DONE["edge_color"]
        layer.current_face_color = EXCLUDE_DONE["face_color"]
        self._drawing = False
        self._select_image_layer()
        self.status("")
        self._exclusions_changed()

    def undo_exclusion(self) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is not None and len(layer.data):
            layer.data = list(layer.data)[:-1]
            self._exclusions_changed()

    def clear_exclusions(self) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is not None:
            layer.data = []
        self._exclusions_changed()

    def set_exclusions(self, polygons: list) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is None:
            return
        layer.events.data.disconnect(self._exclusions_changed)
        layer.data = [np.asarray(p, float) for p in polygons]
        if len(layer.data):
            layer.edge_color = EXCLUDE_DONE["edge_color"]
            layer.face_color = EXCLUDE_DONE["face_color"]
        layer.events.data.connect(self._exclusions_changed)
        self.state.exclusions = [np.asarray(p, float) for p in polygons]
        self._update_box()

    def _exclusions_changed(self, event=None) -> None:
        layer = self._layer(self.EXCLUDE_LAYER)
        if layer is None:
            return
        self.state.exclusions = [np.asarray(p)[:, -2:].astype(float) for p in layer.data if len(p) >= 3]
        self.state.accepted_current = False
        self._update_box()
        if not self._drawing and self.panel.step == 2:
            self.compute_cell()

    def _update_box(self) -> None:
        """The analysed region: all planes, the box around what is not excluded."""
        s = self.state.full
        if s is None:
            return
        nz, ny, nx = s.shape_zyx
        keep = ~exclusion_mask(self.state.exclusions, (0, 0, 0), (1, ny, nx))[0]
        if not keep.any():
            self.state.box = None
            self.panel.cell_error.setText("Everything is excluded. Undo the last shape.")
            self.panel.refresh()
            return
        ys, xs = np.nonzero(keep)
        box = CropBox(0, nz, int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        if box != self.state.box or self.state.crop is None:
            self.state.box = box
            self.state.crop = crop(s, box)
        self.panel.refresh()

    def compute_cell(self, sync: bool = False) -> None:
        """Cell mask of the current crop with the edge factor of step 3."""
        s = self.state
        if s.full is None or s.box is None or s.computing:
            return
        self.panel._cell_debounce.stop()         # computed now with the current factor
        params = self.current_params()
        key = self._cell_key()
        args = (s.crop, [ChannelAssignment(**vars(a)) for a in s.assignments], params,
                [p.copy() for p in s.exclusions], SENSITIVITY_FACTORS if params.sensitivity else ())

        def done(res):
            mask, thr = res[0], res[1]
            s.computing = False
            s.cell, s.cell_key, s.cell_thresholds = mask, key, thr
            s.cell_variants = res[2] if len(res) > 2 else {}
            s.cell_corrected = False
            s.result = None
            self.panel.set_cell_computing(False)
            self._show_cell()
            self.panel.refresh()
            self.panel.continue_after_compute()

        def failed(exc):
            s.computing = False
            self.panel.cancel_pending_step()
            self.panel.set_cell_computing(False)
            self.message(f"Finding the cell failed:\n{exc}", error=True)
            self.panel.refresh()

        s.computing = True
        self.panel.set_cell_computing(True)
        self.panel.refresh()
        self._run(cell_mask, args, {}, done, failed, sync)

    def _show_cell(self) -> None:
        layer = self._layer(self.CELL_LAYER)
        s = self.state
        if layer is None or s.cell is None or s.box is None:
            return
        b = s.box
        data = layer.data
        data[...] = 0
        data[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = s.cell.astype(np.uint8)
        layer.data = data
        layer.refresh()
        layer.visible = True
        dz, dy, dx = s.full.voxel_um
        vol = float(s.cell.sum()) * dz * dy * dx
        planes = int(s.cell.reshape(s.cell.shape[0], -1).any(axis=1).sum())
        n = len(s.exclusions)
        ex = f" {n} region(s) removed." if n else ""
        corr = " Painted in an earlier version." if s.cell_corrected else ""
        self.panel.cell_info.setText(f"Cell {vol:.0f} µm³ in {planes} plane(s).{ex}{corr}")

    # ------------------------------------------------------------------ step 4: compartments
    def base_factors(self) -> dict[str, float]:
        """The factors of the analysis settings: starting values of every cell."""
        return {k: getattr(self.params, k).factor for k in MASK_KEYS}

    def current_params(self) -> AnalysisParams:
        p = AnalysisParams.from_dict(self.params.to_dict())
        for key, value in self.panel.factors().items():
            getattr(p, key).factor = float(value)
        return p

    def _overrides(self) -> dict:
        base = self.base_factors()
        return {k: {"factor": v} for k, v in self.panel.factors().items() if abs(v - base.get(k, 1.0)) > 1e-9}

    def factors_as_default(self) -> None:
        """Step 4: the factors of this cell become the starting values of the analysis."""
        d = self.params.to_dict()
        for key, value in self.panel.factors().items():
            d[key]["factor"] = float(value)
        self._set_params(AnalysisParams.from_dict(d), ask=True)

    def compute(self, sync: bool = False, masks: dict | None = None) -> None:
        """Run the analysis of the current crop with the cell from step 3."""
        s = self.state
        if s.crop is None or s.cell is None or s.computing:
            return
        if masks is None and self._restore_masks is not None:
            masks, self._restore_masks = self._restore_masks, None
        if masks is None:
            masks = {"cell": s.cell}
        corrected = set(masks) - {"cell"} != set()
        self.panel._debounce.stop()              # computed now with the current factors
        params = self.current_params()
        args = (s.crop, [ChannelAssignment(**vars(a)) for a in s.assignments], params)
        kwargs = {"exclusions_yx": [p.copy() for p in s.exclusions], "masks": masks,
                  "cell_variants": getattr(s, "cell_variants", None) if set(masks) == {"cell"} else None}
        t0 = time.perf_counter()
        key = self._result_key()

        def done(result):
            s.computing = False
            s.result = result
            s.result_key = key
            s.result_corrected = bool(corrected)
            s.accepted_current = False
            self.panel.set_computing(False)
            self.show_compartments(result)
            how = "masks painted in an earlier version" if corrected else "automatic masks"
            self.panel.comp_status.setText(f"Computed in {time.perf_counter() - t0:.1f} s with {how}.")
            self.panel.comp_notes.setText("\n".join(result.notes))
            self.panel.refresh()
            if self.panel.step == 4:
                self.panel.show_results(result)
            self.panel.continue_after_compute()

        def failed(exc):
            s.computing = False
            self.panel.cancel_pending_step()
            self.panel.set_computing(False)
            self.panel.comp_status.setText("")
            self.message(f"The analysis failed:\n{exc}", error=True)
            self.panel.refresh()

        s.computing = True
        self.panel.set_computing(True, "Computing the compartments ...")
        self.panel.refresh()
        self._run(analyze_cell, args, kwargs, done, failed, sync)

    def _run(self, func, args, kwargs, done, failed, sync: bool) -> None:
        """Run in a worker thread (or directly for tests). Finished workers stay referenced inside
        Qt and napari, so their arguments are dropped to free the crops (about 45 MB each)."""
        if sync:
            try:
                res = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover
                failed(exc)
                return
            done(res)
            return
        from napari.qt.threading import create_worker

        holder = {}

        def release():
            w = holder.pop("worker", None)
            if w is not None:
                w._args, w._kwargs = (), {}

        worker = create_worker(func, *args, **kwargs, _start_thread=False)
        holder["worker"] = worker
        worker.returned.connect(lambda r: (release(), done(r)))
        worker.errored.connect(lambda e: (release(), failed(e)))
        self._worker = worker
        worker.start()

    def show_compartments(self, result: CellResult) -> None:
        layer = self._layer(self.LABEL_LAYER)
        b = self.state.box
        if layer is None or b is None:
            return
        lab = result.masks["labels"].astype(np.uint8)
        lab[(lab == 2) & result.masks["actin_rich"]] = 4
        data = layer.data
        data[...] = 0
        data[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = lab
        layer.data = data
        layer.refresh()
        colors = {a.label: a.color for a in self.state.assignments}
        for label in [a.label for a in self.state.assignments if a.role == "Target"]:
            t = result.masks.get(f"target_{label}")
            if t is None:
                continue
            tl = self._target_layer(label, colors.get(label, "#FFFFFF"))
            d = tl.data
            d[...] = 0
            d[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1] = t.astype(np.uint8)
            tl.data = d
            tl.refresh()
        self._select_image_layer()
        front = self._layer(self.FRONT_LAYER)
        pol = result.overlays.get("polarity")
        if front is not None:
            if pol:
                p0 = np.array(pol["start"]) + (b.y0, b.x0)
                p1 = np.array(pol["end"]) + (b.y0, b.x0)
                v = p1 - p0
                n = float(np.hypot(*v)) or 1.0
                u, w = v / n, np.array([-v[1], v[0]]) / n
                head = 0.12 * n
                front.data = []
                front.add_paths([np.array([p0, p1 - u * head])], edge_width=3, edge_color="#FF8C00")
                front.add_polygons([np.array([p1, p1 - u * head + w * head * 0.5, p1 - u * head - w * head * 0.5])],
                                   edge_width=1, edge_color="#FF8C00", face_color="#FF8C00")
            else:
                front.data = []
        self.update_compartment_display()

    def update_compartment_display(self) -> None:
        """Show or hide single compartments, the cell edge and the target masks (step 4).
        Cytoplasm is everything outside the nucleus (D39): with actin hidden, the actin voxels are
        shown in the cytoplasm colour. In 3D the masks are shown as smoothed surfaces."""
        if self.panel.step != 3:
            return
        vis = self.panel.visible_compartments()
        three_d = self.viewer.dims.ndisplay == 3
        outlines = self.panel.outline_box.isChecked()
        layer = self._layer(self.LABEL_LAYER)
        if layer is not None:
            show = {"nucleus": vis.get("nucleus", True), "actin": vis.get("actin", True),
                    "actin_rich": vis.get("actin_rich", True), "cytoplasm": vis.get("cytoplasm", True)}
            off = (0, 0, 0, 0)
            cyto = LABEL_COLORS[3] if show["cytoplasm"] else off
            colors = {1: LABEL_COLORS[1] if show["nucleus"] else off,
                      3: cyto,
                      2: LABEL_COLORS[2] if show["actin"] else cyto,
                      4: (LABEL_COLORS[4] if show["actin_rich"] else
                          LABEL_COLORS[2] if show["actin"] else cyto)}
            layer.colormap = _label_cmap(colors)
            _set_if(layer, "contour", 1 if outlines and not three_d else 0)
            layer.visible = not three_d
        cell = self._layer(self.CELL_LAYER)
        if cell is not None:
            cell.visible = vis.get("cell", True) and not three_d
            _set_if(cell, "contour", 0 if three_d else 1)
        for label, tl in self._target_layers.items():
            tl.visible = vis.get(f"target_{label}", False) and not three_d
            _set_if(tl, "contour", 1 if outlines and not three_d else 0)
        front = self._layer(self.FRONT_LAYER)
        if front is not None:
            front.visible = vis.get("front", False) and len(front.data) > 0 and not three_d
        self.panel.outline_box.setEnabled(not three_d)        # outlines and the arrow exist only in 2D
        self.panel.front_box.setEnabled(not three_d)
        if three_d:
            self._show_surfaces(vis)
        else:
            self._hide_surfaces()

    # ------------------------------------------------------------------ 3D surfaces (display only)
    def _surface_masks(self) -> dict[str, np.ndarray]:
        r = self.state.result
        if r is None:
            return {}
        lab = r.masks["labels"]
        out = {"cell": r.masks["cell"], "nucleus": lab == 1, "actin": np.isin(lab, (2,)),
               "actin_rich": r.masks["actin_rich"], "cytoplasm": np.isin(lab, (2, 3))}
        for a in self.state.assignments:
            if a.role == "Target" and f"target_{a.label}" in r.masks:
                out[f"target_{a.label}"] = r.masks[f"target_{a.label}"]
        return out

    def _show_surfaces(self, vis: dict) -> None:
        """Compute the smoothed surfaces once per result (in the background), then toggle them."""
        r = self.state.result
        if r is None or self.state.box is None:
            return
        key = (id(r), self.state.box)
        if self._surface_key != key:
            self._compute_surfaces(key)
            return
        for name, layer in self._surfaces.items():
            if layer in self.viewer.layers:
                layer.visible = vis.get(name, False) if not name.startswith("target_") else vis.get(name, False)

    def _compute_surfaces(self, key) -> None:
        from localization_tool.gui.surfaces import mask_surfaces

        masks = self._surface_masks()
        voxel = self.state.full.voxel_um
        b = self.state.box
        offset = (b.z0 * voxel[0], b.y0 * voxel[1], b.x0 * voxel[2])

        def done(meshes):
            self.panel.set_working(False)
            for layer in self._surfaces.values():
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
            self._surfaces = {}
            colors = {a.label: a.color for a in self.state.assignments}
            for name, mesh in meshes.items():
                if mesh is None:
                    continue
                color = SURFACE_COLORS.get(name) or _hex_to_rgb(colors.get(name[7:], "#FFFFFF"))
                opacity = 0.25 if name in ("cell", "cytoplasm") else 0.6
                layer = self.viewer.add_surface(mesh, name=f"3D {name}", translate=offset, opacity=opacity,
                                                colormap=_single_cmap(color), shading="smooth", visible=False,
                                                blending="translucent")
                self._surfaces[name] = layer
            self._surface_key = key
            self._select_image_layer()
            self.update_compartment_display()

        def failed(exc):
            self.panel.set_working(False)
            self.message(f"The 3D surfaces could not be computed:\n{exc}", error=True)

        self.panel.set_working(True, "Computing the 3D surfaces ...")
        self._run(mask_surfaces, (masks, voxel), {}, done, failed, False)

    def _hide_surfaces(self) -> None:
        for layer in self._surfaces.values():
            if layer in self.viewer.layers:
                layer.visible = False

    def _show_cell_3d(self, on: bool) -> None:
        """Step 3 in 3D: the cell edge as labels without contour (contours exist only in 2D)."""
        cell = self._layer(self.CELL_LAYER)
        if cell is not None:
            _set_if(cell, "contour", 0 if on else 1)
            cell.opacity = 0.35 if on else 0.9

    def show_working(self, busy: bool, text: str) -> None:
        """A note in the image while the program computes, so nobody clicks into a half-done view."""
        ov = self.viewer.text_overlay
        if busy:
            ov.text = text
            ov.font_size = 16
            ov.color = "#E6E9ED"
            ov.visible = True
        elif self.state.full is not None:
            ov.visible = False

    def stop_painting(self) -> None:
        """Painting was removed in 0.5.0a1; kept so older calls stay harmless."""
        return

    # ------------------------------------------------------------------ step 5: results
    def _next_cell_name(self, im: project_file.ImageSpec) -> str:
        names = {c.name for c in im.cells}
        k = 1
        while f"cell{k}" in names:
            k += 1
        return f"cell{k}"

    def accept_cell(self) -> None:
        s = self.state
        r = s.result
        im = self._current_image()
        if s.accepted_current:
            return
        if im is None or r is None:
            self.message("No image of the list is open. Open an image in step 1 first.", error=True)
            return
        name = s.editing or self._next_cell_name(im)
        corrected = s.result_corrected or s.cell_corrected
        status = "accepted, corrected" if corrected else "accepted"
        im.experiment, im.condition = s.experiment, s.condition
        png = batch.control_png(s.crop, s.assignments, r, im.path.name, s.experiment, s.condition, name)
        png_name = self._control_name(im, name)
        masks = None
        mask_file = ""
        if s.result_corrected:
            lab = r.masks["labels"].astype(np.uint8)
            masks = {"labels": lab}
        elif s.cell_corrected:
            masks = {"cell": s.cell}
        if masks is not None:
            mask_file = f"masks/{png_name[:-4]}.npz"
        spec = project_file.CellSpec(name=name, box=s.box, exclusions=[p.tolist() for p in s.exclusions],
                                     status=status, overrides=self._overrides(), mask_file=mask_file)
        spec.results = batch.cell_results(im, spec, s.box, r, self.params)
        im.cells = [c for c in im.cells if c.name != name] + [spec]
        im.cells.sort(key=lambda c: c.name)
        im.assignments = [ChannelAssignment(**vars(a)) for a in s.assignments]
        im.experiment, im.condition = s.experiment, s.condition
        self.pending[(str(im.path), name)] = {"png": png, "png_name": png_name, "masks": masks,
                                              "mask_file": mask_file}
        s.accepted_current = True
        s.editing = name
        self.panel.accept_btn.setEnabled(False)
        self.panel.accept_btn.setText("Accepted")
        saved = self._autosave()
        where = " Saved." if saved else " Save the results when you are done."
        self.panel.set_accepted_text(f"{self.session.shown_name(im)} / {name} accepted. {self._count_text()}{where}")
        self.status(f"{self.session.shown_name(im)} / {name} accepted.")
        self.refresh_lists()

    def _control_name(self, im: project_file.ImageSpec, cell: str) -> str:
        return batch.control_name(im.path, cell, [i.path for i in self.session.images])

    def next_cell(self) -> None:
        """Another cell in the same image: start again in step 3 with nothing removed."""
        s = self.state
        s.editing = None
        s.accepted_current = False
        s.cell = None
        s.cell_variants = {}
        s.result = None
        self.clear_exclusions()
        self.panel.set_factors({}, self.base_factors())
        self.panel.go(2)

    def next_image(self) -> None:
        """The next image without a checked cell: a new image, or a cell from 'Analyze all images'."""
        cur = self.state.index
        order = self.session.blind_order()     # by code during blind review: not folder or condition (D57)
        pos = order.index(cur) if cur in order else -1
        for i in order[pos + 1:] + order[:max(pos, 0) + 1]:
            im = self.session.images[i]
            if _is_example(im.path) or (i == cur and self.state.editing is None):
                continue
            todo = [j for j, c in enumerate(im.cells) if c.status == batch.AUTOMATIC]
            if todo:
                self.edit_cell(self._cell_entries().index((i, todo[0])))
                return
            if not im.cells:
                self.open_image_index(i)
                return
        self.panel.go(0)
        self.message("Every image in the list has a checked cell. Add more images or save the results.")

    def edit_cell(self, row: int) -> None:
        entries = self._cell_entries()
        if not (0 <= row < len(entries)):
            return
        i, j = entries[row]
        im = self.session.images[i]
        spec = im.cells[j]

        def restore():
            self._restore_cell(i, spec)

        if self.state.index == i and self.state.full is not None:
            restore()
        else:
            self.open_image_index(i, after=restore)

    def _restore_cell(self, index: int, spec: project_file.CellSpec) -> None:
        s = self.state
        im = self.session.images[index]
        self.set_exclusions(spec.exclusions)
        self.panel.set_factors(spec.overrides, self.base_factors())
        masks = None
        pend = self.all_pending().get((str(im.path), spec.name))
        if pend and pend.get("masks") is not None:
            m = pend["masks"]
            masks = ({"cell": m["labels"] > 0, "nucleus": m["labels"] == 1, "actin": np.isin(m["labels"], (2, 4))}
                     if "labels" in m else {"cell": m["cell"]})
        elif spec.mask_file and self.session_dir is not None and (self.session_dir / spec.mask_file).exists():
            masks = project_file.load_masks(self.session_dir / spec.mask_file)
        s.editing = spec.name
        if masks is not None and s.crop is not None and masks["cell"].shape == s.crop.shape_zyx:
            s.cell = masks["cell"]
            s.cell_variants = {}                      # the variants belong to a computed cell, not to this one
            s.cell_key = self._cell_key()
            s.cell_corrected = True
            s.result = None
            if set(masks) != {"cell"}:
                self._restore_masks = masks           # used by the next compute in step 4
            self._show_cell()
            self.panel.go(2)
            self.panel.go(3)
        else:
            s.cell = None
            s.cell_variants = {}
            self.panel.go(2)
        self.status(f"Editing {self.session.shown_name(im)} / {spec.name}")

    def remove_cell(self, row: int) -> None:
        entries = self._cell_entries()
        if not (0 <= row < len(entries)):
            return
        i, j = entries[row]
        im = self.session.images[i]
        spec = im.cells[j]
        if not self.ask(f"Remove {self.session.shown_name(im)} / {spec.name} from the analysis?", "Remove cell"):
            return
        del im.cells[j]
        self.pending.pop((str(im.path), spec.name), None)
        if self.state.index == i and self.state.editing == spec.name:
            self.state.editing = None
            self.state.accepted_current = False
        self._autosave()
        self.refresh_lists()

    # ------------------------------------------------------------------ settings and all cells (phase 2)
    def open_settings(self) -> None:
        from localization_tool.gui.settings import SettingsDialog

        labels = list(dict.fromkeys(
            [a.label for a in self.state.assignments if a.role != "Ignore"]
            + [a.label for im in self.session.images for a in (im.assignments or []) if a.role != "Ignore"]))
        estimate = self._estimate_crosstalk if self.state.cell is not None else None
        chans = None
        if self.state.full is not None and self.state.assignments:
            chans = [(a.label, ch.fluor or ch.name, a.role)
                     for a, ch in zip(self.state.assignments, self.state.full.channels)]
        dark = None
        if self.state.full is not None and self.state.assignments:
            from localization_tool.core.qc import dark_level

            dark = {a.label: dark_level(self.state.full.data[i]) for i, a in enumerate(self.state.assignments)
                    if a.role != "Ignore" and i < self.state.full.data.shape[0]}
        dlg = SettingsDialog(self.viewer.window._qt_window, self.params, labels, self.session.vocabulary, estimate,
                             blind=self.session.blind, channels=chans, dark_levels=dark)
        if dlg.exec_() and dlg.result_params is not None:
            self.session.vocabulary = dlg.result_vocabulary
            if dlg.result_blind != self.session.blind:
                self.session.blind = dlg.result_blind
                self.session.assign_codes()
                self.refresh_lists()
                s = self.state
                im = self._current_image()
                if s.full is not None and im is not None:
                    self.panel.show_info(s.full, s.load_seconds, s.experiment, s.condition,
                                         shown=self.session.shown_name(im), hidden=self._hidden(im))
            self._set_params(dlg.result_params, ask=True)

    def _set_params(self, params: AnalysisParams, ask: bool) -> None:
        """New settings for the analysis. Cells computed with other settings can be recomputed."""
        self.params = params
        self.session.params = params
        self._align_layers()
        s = self.state
        if s.full is not None:
            self.panel.set_factors(self._overrides_from_state(), self.base_factors())
            s.cell_key = None
            s.result = None
        stale = [(i, c.name) for i, im in enumerate(self.session.images) for c in im.cells
                 if c.status != "excluded" and batch.needs_update(c, params)]
        self._autosave()
        if stale and ask and self.ask(f"{len(stale)} cell(s) were computed with other settings. "
                                      "Compute them again now with the new settings?\n\n"
                                      "Painted corrections and removed regions are kept. Cells whose "
                                      "outlines change are marked 'to check' and should be checked again.",
                                      "Compute again", "Later"):
            self.run_all("all")
        elif s.full is not None and self.panel.step in (2, 3):
            self.panel.go(self.panel.step)

    def _overrides_from_state(self) -> dict:
        spec = None
        im = self._current_image()
        if im is not None and self.state.editing:
            spec = next((c for c in im.cells if c.name == self.state.editing), None)
        return spec.overrides if spec else {}

    def _estimate_crosstalk(self, into: str, src: str) -> dict:
        from localization_tool.core import crosstalk, metrics

        s = self.state
        labels = [a.label for a in s.assignments]
        if s.crop is None or s.cell is None or into not in labels or src not in labels:
            raise ValueError("Open the control image and find the cell in step 3 first.")
        from localization_tool.core.analysis import _shift, channel_offset

        i, j = labels.index(into), labels.index(src)

        def aligned(k: int):                 # the same channel offset as in the analysis (D55)
            ch = s.crop.channels[k] if k < len(s.crop.channels) else None
            off = channel_offset(self.params, labels[k], ch.fluor if ch else None)
            return _shift(s.crop.data[k], off, s.crop.voxel_um) if off else s.crop.data[k]

        a = metrics.background_subtract(aligned(i), self.params.background_for(into))
        b = metrics.background_subtract(aligned(j), self.params.background_for(src))
        return crosstalk.estimate_k(b, a, s.cell)

    def run_all(self, mode: str) -> None:
        """mode 'new': analyse every image without a cell automatically (status 'automatic').
        mode 'all': compute every cell again whose settings changed."""
        from napari.qt.threading import create_worker

        if self._batch is not None or self.state.computing:
            self.message("Please wait until the current calculation has finished.")
            return
        if mode == "new":
            jobs = [(i, None) for i, im in enumerate(self.session.images) if not im.cells and not _is_example(im.path)]
            if not self.session.images:
                self.message("The list has no images yet. Add a folder or images first.")
                return
            if not jobs:
                self.message("Every image in the list already has a cell. Use 'Apply settings to all cells' "
                             "(step 5) to compute them again.")
                return
            sizes = [self.session.images[i].path.stat().st_size for i, _ in jobs
                     if self.session.images[i].path.exists()]
            need = MEMORY_FACTOR * max(sizes, default=0) / 1e9
            if need > MEMORY_WARN_GB and not self.ask(
                    f"The largest image needs about {need:.0f} GB of memory for the analysis. On a computer with "
                    "less memory the program can become very slow or stop. Continue?", "Continue"):
                return
        else:
            jobs = [(i, c.name) for i, im in enumerate(self.session.images) for c in im.cells
                    if c.status != "excluded" and not _is_example(im.path)]
            if not jobs:
                self.message("There are no cells in this analysis yet.")
                return
        panels = self.panels

        def find(stack):
            panel = find_panel(panels, stack.channel_signature())
            return [ChannelAssignment(**vars(a)) for a in panel.assignments] if panel else None

        af = project_file.AnalysisFile(images=self.session.images, panel=self.session.panel, params=self.params)
        gen = batch.iter_cells(af, jobs, base_dir=self.session_dir, find_assignments=find,
                               shown_name=self.session.shown_name,
                               pending_masks=self.all_pending())
        def work():
            yield from gen

        worker = create_worker(work, _start_thread=False)
        summary = {"cells": 0, "unchanged": 0, "skipped": []}
        worker.yielded.connect(lambda ev: self._batch_event(ev, summary))
        worker.finished.connect(lambda: self._batch_done(summary))
        self._batch = worker
        self.panel.set_batch(True, f"Starting ({len(jobs)} cell(s)) ...")
        worker.start()

    def stop_batch(self) -> None:
        if self._batch is not None:
            self._batch.quit()
            self.panel.set_batch(True, "Stopping after the current cell ...")

    def _batch_event(self, ev: dict, summary: dict) -> None:
        kind = ev["kind"]
        if kind == "progress":
            self.panel.set_batch(True, ev["text"], ev["done"], ev["total"])
        elif kind == "skipped":
            if ev.get("unchanged"):
                summary["unchanged"] += 1
            else:
                summary["skipped"].append(ev["reason"])
        elif kind == "cell":
            idx = next((k for k, x in enumerate(self.session.images) if str(x.path) == ev["path"]), None)
            if idx is None:                      # the image was removed meanwhile
                return
            im = self.session.images[idx]
            ev = {**ev, "image": idx}
            spec = ev["spec"]
            if im.assignments is None:
                im.assignments = [ChannelAssignment(**vars(a)) for a in ev["assignments"]]
            im.cells = sorted([c for c in im.cells if c.name != spec.name] + [spec], key=lambda c: c.name)
            old = self.pending.get((str(im.path), spec.name), {})
            self.pending[(str(im.path), spec.name)] = {"png": ev["png"], "png_name": self._control_name(im, spec.name),
                                                        "masks": old.get("masks"), "mask_file": spec.mask_file}
            summary["cells"] += 1
            if self.state.index == ev["image"] and self.state.editing == spec.name:
                self.state.accepted_current = False
            self.refresh_lists()

    def _batch_done(self, summary: dict) -> None:
        self._batch = None
        self.panel.set_batch(False)
        saved = self._autosave()
        self.refresh_lists()
        text = [f"{summary['cells']} cell(s) computed."]
        if summary["unchanged"]:
            text.append(f"{summary['unchanged']} cell(s) were already up to date.")
        if summary["skipped"]:
            text.append("Not computed:\n" + "\n".join(summary["skipped"][:12]))
        if summary["cells"]:
            text.append("Saved." if saved else "Save the results to keep them.")
            text.append("These cells are marked 'to check'. Open the first one now? 'Next image' in step 5 then "
                        "opens the others one after the other.")
            if self.ask("\n\n".join(text), "Open first cell", "Not now"):
                self.state.index = -1
                self.next_image()
            return
        self.message("\n\n".join(text))

    # ------------------------------------------------------------------ saving
    def all_rows(self) -> tuple[list, list]:
        """Stored rows of every cell, with the current experiment, condition and status."""
        rows, prof = [], []
        practice_only = all(_is_example(im.path) for im in self.session.images)
        for im in self.session.images:
            if _is_example(im.path) and not practice_only:   # practice cell: never next to real cells
                continue
            meta = {"experiment": im.experiment, "condition": im.condition, "blind_code": im.code,
                    "name_revealed": im.revealed}
            for c in im.cells:
                if c.results and c.status != "excluded":
                    rows += [{**r, **meta, "status": c.status} for r in c.results.get("rows", [])]
                    prof += [{**r, **meta} for r in c.results.get("profile_rows", [])]
        return rows, prof

    def _autosave(self) -> bool:
        if self.session_dir is None:
            return False
        self.save_results(folder=self.session_dir, notify=False, background=True)
        return True

    def save_results(self, folder: Path | None = None, notify: bool = True, background: bool = False) -> Path | None:
        """Write the analysis folder: analysis.lta.json, Results.xlsx, control images, masks.

        The content is taken as a snapshot here. With background=True (automatic saving after each
        cell) the files are written in a worker thread, so the window does not wait for Excel."""
        if folder is None:
            folder = self.session_dir or (self.panel.ask_folder() if notify else None)
        if folder is None:
            return None
        if not any(im.cells for im in self.session.images):
            if notify:
                self.message("No accepted cells yet. Accept at least one cell in step 5.")
            return None
        if background and self._saving:
            self._save_again = True
            return None
        folder = Path(folder)
        if self.session_dir is None or folder != self.session_dir:
            try:                                  # a read-only folder, e.g. on the image server
                folder.mkdir(parents=True, exist_ok=True)
                probe = folder / ".localization_tool_write_test"
                probe.write_text("x", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                self.message(f"The folder cannot be written:\n{folder}\n\n{exc}\n\nPlease choose another folder.",
                             error=True)
                return None
        snap = self._snapshot(folder)
        first = self.session_dir is None
        if first:
            prefs.add_recent(folder / ANALYSIS_NAME)
        self.session_dir = folder
        n = sum(1 for im in self.session.images for c in im.cells if c.status != batch.EXCLUDED)
        self.panel.show_analysis(folder, self._short_count())

        def done(path):
            self._saving = False
            self._writing = {}
            self.panel.save_info.setText(f"Saved {time.strftime('%H:%M')} to {folder}. Results.xlsx holds all "
                                         "numbers. From now on every accepted cell is saved automatically.")
            self.status(f"Saved {n} cell(s) to {folder}")
            if self._save_again:
                self._save_again = False
                self.save_results(folder=folder, notify=False, background=True)

        def failed(exc):
            self._saving = False
            self._writing = {}
            for key, item in snap["pending"].items():
                self.pending.setdefault(key, item)
            locked = isinstance(exc, PermissionError)
            text = (f"Results.xlsx or the analysis file could not be written. Either the file is open in another "
                    f"program (e.g. Excel), or you have no write permission for the folder. Close the file or "
                    f"choose another folder; the results are saved again with the next cell or with "
                    f"'Save results'.\n\n{exc}" if locked else f"Saving failed:\n{exc.__class__.__name__}: {exc}")
            self.panel.save_info.setText("Not saved: " + ("file open or folder not writable" if locked else str(exc)))
            if notify or locked:
                self.message(text, error=True)

        if background:
            from napari.qt.threading import create_worker

            self._saving = True
            worker = create_worker(_write_snapshot, snap, _start_thread=False)
            worker.returned.connect(done)
            worker.errored.connect(failed)
            self._save_worker = worker
            worker.start()
            path = folder / ANALYSIS_NAME
        else:
            try:
                path = _write_snapshot(snap)
            except Exception as exc:             # e.g. openpyxl errors, not only OSError
                failed(exc)
                return None
            done(path)
        if notify and first:
            self.message(f"Saved {n} cell(s) to\n{folder}\n\nResults.xlsx: all numbers (start with the sheet "
                         f"'Read me')\n{batch.CONTROL_DIR}: one picture per cell\n{ANALYSIS_NAME}: open it again "
                         "with File > Open analysis\n\nFrom now on every accepted cell is saved automatically.")
        return path

    def all_pending(self) -> dict:
        """Control images and masks not yet on disk, including those of a save still running."""
        return {**self._writing, **self.pending}

    def _snapshot(self, folder: Path) -> dict:
        """Everything to write, taken now: later changes of the analysis do not mix into this save."""
        self.session.params = self.params
        pending, self.pending = self.pending, {}
        self._writing = dict(pending)
        rows, prof = self.all_rows()
        images = [{"image": im.path.name, "experiment": im.experiment, "condition": im.condition,
                   "cells": len(im.cells), "channel roles": batch.roles_text(im.assignments), "path": str(im.path)}
                  for im in self.session.images]
        info = batch.run_info(self.params, None, images)
        info["analysis file"] = str(folder / ANALYSIS_NAME)
        return {"folder": folder, "seq": next(_SEQ), "pending": pending, "json": project_file.dumps(self.session, folder / ANALYSIS_NAME),
                "rows": rows, "prof": prof, "excluded": batch.excluded_rows(self.session), "info": info}

    # ------------------------------------------------------------------ helpers
    def _layer(self, name: str):
        return self.viewer.layers[name] if name in self.viewer.layers else None

    def _set_visible(self, name: str, visible: bool) -> None:
        layer = self._layer(name)
        if layer is not None:
            layer.visible = visible


_SEQ = itertools.count()
_WRITE_LOCK = threading.Lock()
_WRITTEN: dict[str, int] = {}          # folder -> number of the newest snapshot written


def _write_snapshot(snap: dict) -> Path:
    """Write a snapshot taken by MainWindow._snapshot (runs in a worker thread for automatic saving).
    Saves run one after the other, and an older snapshot never overwrites a newer one."""
    folder = snap["folder"]
    with _WRITE_LOCK:
        if _WRITTEN.get(str(folder), -1) > snap["seq"]:
            _write_files(snap, tables=False)          # a newer save is on disk: only missing pictures
            return folder / ANALYSIS_NAME
        _write_files(snap)
        _WRITTEN[str(folder)] = snap["seq"]
    return folder / ANALYSIS_NAME


def _write_files(snap: dict, tables: bool = True) -> Path:
    folder = snap["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for (img, name), p in snap["pending"].items():
        if p.get("png"):
            out = folder / batch.CONTROL_DIR / (p.get("png_name") or batch.control_name(img, name))
            if tables or not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(p["png"])
        if p.get("masks") is not None and p.get("mask_file") and (tables or not (folder / p["mask_file"]).exists()):
            project_file.save_masks(folder / p["mask_file"], **p["masks"])
    if not tables:
        return folder / ANALYSIS_NAME
    path = project_file.write_text(folder / ANALYSIS_NAME, snap["json"])
    batch.write_workbook(folder, snap["rows"], snap["prof"], snap["info"], snap["excluded"])
    return path


SURFACE_COLORS = {"cell": (1.0, 1.0, 1.0), "cytoplasm": (0.6, 0.64, 0.69), "nucleus": (0.24, 0.49, 1.0),
                  "actin": (0.17, 0.83, 0.42), "actin_rich": (1.0, 0.84, 0.04)}


def _set_if(layer, name: str, value) -> None:
    """Set a layer property only when it changes: every set refreshes the layer."""
    if getattr(layer, name) != value:
        setattr(layer, name, value)


def _single_cmap(rgb):
    from napari.utils.colormaps import Colormap

    return Colormap(colors=[[*rgb, 1], [*rgb, 1]], name=f"lt_surface_{'_'.join(f'{c:.2f}' for c in rgb)}")


def _label_cmap(colors: dict):
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), **colors})


def _colormap(a: ChannelAssignment):
    from napari.utils.colormaps import Colormap

    rgb = _hex_to_rgb(a.color)
    return Colormap(colors=[[0, 0, 0, 1], [*rgb, 1]], name=f"lt_{a.color}")


def _contrast(data: np.ndarray) -> tuple[float, float]:
    sample = data[:, ::2, ::2] if data.ndim == 3 else data
    lo, hi = np.percentile(sample, (0.1, 99.9))
    if hi <= lo:
        lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        hi = lo + 1
    return float(lo), float(hi)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = (color or "#FFFFFF").lstrip("#")
    if c in DISPLAY_COLORS:
        c = DISPLAY_COLORS[c].lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


if __name__ == "__main__":
    sys.exit(run_gui(sys.argv))
