"""Guided workflow panel (D17, D21): five steps, one at a time, each with one sentence of instruction.

A step can only be left forward when its task is done. While the program computes, the panel shows
a busy bar and is locked, so nothing is clicked half way. The panel only talks to MainWindow;
all calculations happen in localization_tool.core.

Masks are set only through factors (0.4.0a1 feedback: painting by hand is neither reproducible nor
precise enough), parts that do not belong to the cell are removed with polygons in step 3.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QGridLayout, QHeaderView, QInputDialog, QLineEdit, QListWidget, QProgressBar, QScrollArea, QSizePolicy, QSlider,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from localization_tool.core.channels import validate
from localization_tool.core.reader import SUPPORTED_SUFFIXES
from localization_tool.gui import theme
from localization_tool.gui.widgets import ChannelTable, ElidedLabel, StepBar, button, label

LOADED = "Image loaded. Check the information below, then click Next."
INSTRUCTIONS = (
    "Open images: single files or a whole folder of an experiment day.",
    "Tell the program what each channel shows. Target is the protein you want to localize.",
    "Check the cell outline. Remove what does not belong to the cell, adjust the edge factor.",
    "Check the compartments. Show them one by one and adjust the edges with the factors.",
    "Check the numbers. Accept the cell or exclude it with a reason, then continue.",
)

COMPARTMENT_TOGGLES = (("nucleus", "Nucleus", "#3D7DFF"), ("actin", "Actin", "#2BD46B"),
                       ("actin_rich", "Actin-rich", "#FFD60A"), ("cytoplasm", "Cytoplasm", "#9AA4AF"),
                       ("cell", "Cell edge", "#FFFFFF"))

ONLY_ITEMS = {"Show all": None, "Only nucleus": "nucleus", "Only actin": "actin", "Only actin-rich": "actin_rich",
              "Only cytoplasm": "cytoplasm", "Only target mask": "target"}

CHOOSE_REASON = "Choose a reason ..."
EXCLUDE_REASONS = ("Cell cut off (stack too short)", "Several cells, cannot be separated", "Out of focus",
                   "Dying or dividing cell", "Debris or artefact", "Weak or uneven staining", "Other")


def panel_width() -> int:
    """Minimum panel width: narrower on small screens (more room for the image), wider with large text."""
    from qtpy.QtWidgets import QApplication

    from localization_tool.prefs import font_scale

    screen = QApplication.primaryScreen()
    small = screen is not None and screen.availableGeometry().width() < 1400
    return int((460 if small else 540) * max(1.0, font_scale()))


def _page() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 4, 0)
    lay.setSpacing(8)
    return w, lay


def _row(*widgets, stretch: bool = True) -> QHBoxLayout:
    row = QHBoxLayout()
    for w in widgets:
        row.addWidget(w)
    if stretch:
        row.addStretch(1)
    return row


def _factor_box(tip: str) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(0.2, 3.0)
    sb.setSingleStep(0.05)
    sb.setDecimals(2)
    sb.setValue(1.0)
    sb.setToolTip(tip)
    return sb


def _fmt(value, pattern: str = "{:.0f}", empty: str = "n/a") -> str:
    if value is None or value != value:
        return empty
    text = pattern.format(value)
    return text[1:] if text.startswith("-") and not text.strip("-0.,") else text      # no "-0.00"


def _set_role(widget, role: str) -> None:
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


class WorkflowPanel(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.step = 0
        self._base: dict[str, float] = {}
        self._working = False
        self._jobs: dict[str, str] = {}          # running calculations by kind (D87): one finishing does not unlock the others
        self._go_after: int | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top = QHBoxLayout()
        self.home_btn = button("← Home", flat=True, tip="Back to the start page. The analysis stays open.")
        self.home_btn.clicked.connect(self.main.show_start)
        top.addWidget(self.home_btn)
        top.addStretch(1)
        # which analysis is open; New and Open live on the start page and in the File menu (D68)
        self.analysis_info = ElidedLabel("", "hint")
        self.analysis_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.analysis_info, 1)
        self.show_analysis(None, "")
        root.addLayout(top)
        self.header = StepBar()
        self.header.stepClicked.connect(self.go)
        root.addWidget(self.header)
        self.instruction = label("", "instruction")
        root.addWidget(self.instruction)
        self.alert = label("", "warn")           # notes of steps 4 and 5, always visible (outside the scroll area)
        self.alert.setVisible(False)
        root.addWidget(self.alert)

        # view controls: they change only the display, so they sit on the image, not in the panel (D68).
        # MainWindow places this frame in the lower left corner of the image area.
        self.view_2d = button("2D", tip="Show single planes")
        self.view_3d = button("3D", tip="Show the stack in 3D (drag to rotate)")
        for b in (self.view_2d, self.view_3d):
            b.setCheckable(True)
        self.zoom_in = button("+", tip="Zoom in (or use the mouse wheel)")
        self.zoom_out = button("−", tip="Zoom out")
        self.fit_btn = button("Fit image", tip="Show the whole image")
        self.fit_cell_btn = button("Fit cell", tip="Zoom to the cell")
        self.view_row = QFrame()
        self.view_row.setObjectName("viewbar")
        vl = QVBoxLayout(self.view_row)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(4)
        pair = QHBoxLayout()
        pair.setSpacing(4)
        pair.addWidget(self.view_2d)
        pair.addWidget(self.view_3d)
        vl.addLayout(pair)
        zoom = QHBoxLayout()
        zoom.setSpacing(4)
        zoom.addWidget(self.zoom_in)
        zoom.addWidget(self.zoom_out)
        vl.addLayout(zoom)
        vl.addWidget(self.fit_btn)
        vl.addWidget(self.fit_cell_btn)
        for b in (self.view_2d, self.view_3d, self.zoom_in, self.zoom_out, self.fit_btn, self.fit_cell_btn):
            b.setProperty("size", "compact")
            b.setMinimumWidth(0)
        self.view_2d.clicked.connect(lambda: self.main.set_ndisplay(2))
        self.view_3d.clicked.connect(lambda: self.main.set_ndisplay(3))
        self.zoom_in.clicked.connect(lambda: self.main.zoom(1.3))
        self.zoom_out.clicked.connect(lambda: self.main.zoom(1 / 1.3))
        self.fit_btn.clicked.connect(self.main.reset_view)
        self.fit_cell_btn.clicked.connect(self.main.fit_cell)

        # busy indicator: shown while the program computes, the panel is locked meanwhile
        self.work_box = QWidget()
        wl = QVBoxLayout(self.work_box)
        wl.setContentsMargins(0, 0, 0, 0)
        self.work_text = label("", "title")
        self.work_bar = QProgressBar()
        self.work_bar.setRange(0, 0)
        self.work_bar.setTextVisible(False)
        wl.addWidget(self.work_text)
        wl.addWidget(self.work_bar)
        self.work_box.setVisible(False)
        root.addWidget(self.work_box)

        self.pages = QStackedWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.pages)
        root.addWidget(self.scroll, 1)
        self._build_open()
        self._build_channels()
        self._build_cell()
        self._build_compartments()
        self._build_results()

        # step 5: the decision is always visible, also on small screens (UI review of 1.2)
        self.decide_box = QWidget()
        dl = QVBoxLayout(self.decide_box)
        dl.setContentsMargins(0, 4, 0, 0)
        dl.setSpacing(4)
        dl.addLayout(_row(self.accept_btn, self.exclude_cell_btn, self.next_image_btn))
        dl.addWidget(self.accepted_info)
        self.decide_box.setVisible(False)
        root.addWidget(self.decide_box)
        self.back_btn = button("‹ Back", large=True, tip="Previous step (Ctrl+Left)")
        self.next_btn = button("Next ›", primary=True, large=True, tip="Next step (Ctrl+Right)")
        nav = QHBoxLayout()
        nav.addWidget(self.back_btn)
        nav.addStretch(1)
        nav.addWidget(self.save_btn)            # step 5: always visible, also on small screens
        nav.addWidget(self.next_btn)
        root.addLayout(nav)
        self.back_btn.clicked.connect(lambda: self.go(self.step - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.step + 1))
        self.setMinimumWidth(panel_width())
        self._last_dir = str(Path.home())
        self.go(0)

    def show_analysis(self, folder, counts: str) -> None:
        """Header line: the open analysis (folder name and counts), or that it is not saved yet."""
        if folder is None:
            self.analysis_info.set_full_text("Not saved" + (f" · {counts}" if counts else ""),
                                             "The analysis is kept until you save it (step 5, Save results ...).")
        else:
            self.analysis_info.set_full_text(f"{Path(folder).name} · {counts}", f"{folder}\n{counts}")

    # ------------------------------------------------------------------ navigation
    def reachable(self, target: int) -> bool:
        """A step can be opened when it is not later than the current one, or when every step from the
        current one up to it is complete (the same rule as Next, D62)."""
        if self._working:
            return False
        if target <= self.step:
            return True
        return all(self.main.can_leave(k) for k in range(self.step, target))

    def go(self, step: int) -> None:
        step = max(0, min(4, step))
        if self._flush_pending():
            if step > self.step:
                self._go_after = step            # continue as soon as the new value is computed
                return
        if step > self.step and not all(self.main.can_leave(k) for k in range(self.step, step)):
            return
        previous = self.step
        self.step = step
        for k in range(self.pages.count()):     # the stack takes the size of the shown page, not of the largest
            self.pages.widget(k).setSizePolicy(QSizePolicy.Preferred,
                                               QSizePolicy.Preferred if k == step else QSizePolicy.Ignored)
        self.pages.setCurrentIndex(step)
        self.pages.adjustSize()
        if step != previous:
            self.scroll.verticalScrollBar().setValue(0)
        self.instruction.setText(LOADED if step == 0 and self.main.state.full is not None else INSTRUCTIONS[step])
        if step == 1:
            self.rebuild_display()               # visibility may have changed in steps 3 and 4 (D87)
        if step in (2, 3):
            self.rebuild_channel_toggles()
        self.main.enter_step(step, previous)
        self.refresh()

    def _flush_pending(self) -> bool:
        """A factor changed less than 0.5 s ago: compute it now instead of losing it (phase 5 review)."""
        if self.step == 2 and self._cell_debounce.isActive():
            self._cell_debounce.stop()
            if not self.main.can_leave(2):
                self.main.compute_cell()
                return True
        if self.step == 3 and self._debounce.isActive():
            self._debounce.stop()
            if not self.main.can_leave(3):
                self.main.compute()
                return True
        return False

    def continue_after_compute(self) -> None:
        step, self._go_after = self._go_after, None
        if step is not None:
            self.go(step)

    def cancel_pending_step(self) -> None:
        self._go_after = None

    def refresh(self) -> None:
        busy = self._working
        batch = getattr(self.main, "_batch", None) is not None
        self.back_btn.setEnabled(self.step > 0 and not busy)
        self.next_btn.setVisible(self.step < 4)
        self.next_btn.setEnabled(self.main.can_leave(self.step) and not busy and not batch)
        self.next_btn.setToolTip("Next step (Ctrl+Right)" if self.next_btn.isEnabled() else
                                 (self.main.blocker(self.step) or "Next step (Ctrl+Right)"))
        self.save_btn.setVisible(self.step == 4)
        self.decide_box.setVisible(self.step == 4)
        self.decide_box.setEnabled(not busy)
        notes = {3: self.comp_notes.text(), 4: self.result_notes.text()}.get(self.step, "")
        self.alert.setText(notes)
        self.alert.setVisible(bool(notes))
        reach = [self.reachable(k) and not batch for k in range(5)]
        reasons = []
        for k in range(5):
            why = ""
            if not reach[k]:
                if busy or batch:
                    why = "Wait until the calculation has finished."
                else:
                    first = next((j for j in range(self.step, k) if not self.main.can_leave(j)), None)
                    why = self.main.blocker(first) if first is not None else ""
            reasons.append(why)
        self.header.set_state(self.step, reach, reasons)
        nd = self.main.viewer.dims.ndisplay
        self.view_2d.setChecked(nd == 2)
        self.view_3d.setChecked(nd == 3)

    def set_working(self, busy: bool, text: str = "", job: str = "work") -> None:
        """Lock the panel and show a busy bar while the program computes. Each kind of calculation is
        counted on its own, so the panel unlocks only when the last one has finished (D87)."""
        if busy:
            self._jobs[job] = text
        else:
            self._jobs.pop(job, None)
        busy = bool(self._jobs)
        text = next(reversed(self._jobs.values()), "") if busy else ""
        self._working = busy
        self.work_box.setVisible(busy)
        self.work_text.setText(text)
        self.pages.setEnabled(not busy)
        self.view_row.setEnabled(not busy)
        self.main.show_working(busy, text)
        self.refresh()

    # ------------------------------------------------------------------ 1 open
    def _build_open(self):
        w, lay = _page()
        lay.addWidget(label("Images", "title"))
        self.add_folder_btn = button("Add folder ...", primary=True,
                                     tip="All images in a folder and its subfolders, e.g. one experiment day")
        self.add_btn = button("Add images ...", tip="Choose one or several image files")
        lay.addLayout(_row(self.add_folder_btn, self.add_btn))
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(130)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self.image_list)
        self.empty_hint = label("No images yet. 'Add folder ...' takes a whole experiment day with its subfolders.",
                                "hint")
        lay.addWidget(self.empty_hint)
        self.open_sel_btn = button("Open selected")
        self.remove_img_btn = button("Remove from list")
        self.exclude_img_btn = button("Exclude ...", tip="Leave this image out of the analysis, with a reason")
        lay.addLayout(_row(self.open_sel_btn, self.remove_img_btn, self.exclude_img_btn))
        self.analyze_all_btn = button("Analyze all images", tip="Find the cell and the compartments in every image "
                                      "without a cell, with the current settings. Check them afterwards.")
        lay.addLayout(_row(self.analyze_all_btn))
        self.batch_box = QWidget()
        bl = QVBoxLayout(self.batch_box)
        bl.setContentsMargins(0, 0, 0, 0)
        self.batch_bar = QProgressBar()
        self.batch_text = label("", "hint")
        self.batch_stop = button("Stop")
        bl.addWidget(self.batch_bar)
        bl.addLayout(_row(self.batch_text, self.batch_stop))
        self.batch_box.setVisible(False)
        lay.addWidget(self.batch_box)
        self.busy = label("", "hint")
        lay.addWidget(self.busy)
        self.info = QFormLayout()
        self.info.setLabelAlignment(Qt.AlignRight)
        lay.addLayout(self.info)
        self.reveal_btn = button("Show name", tip="Blind review: show file name and condition of this image. "
                                 "This is noted in the analysis file.")
        self.reveal_btn.clicked.connect(self.main.reveal_current)
        self.reveal_btn.setVisible(False)
        lay.addLayout(_row(self.reveal_btn))
        self.open_warn = label("", "warn")
        lay.addWidget(self.open_warn)
        form = QFormLayout()
        self.experiment = QLineEdit()
        self.experiment.setPlaceholderText("e.g. 2026-06-25")
        self.condition = QLineEdit()
        self.condition.setPlaceholderText("e.g. WT, treated")
        self.meta_box = QWidget()
        self.meta_box.setLayout(form)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(label("Experiment", "hint", wrap=False), self.experiment)
        form.addRow(label("Condition", "hint", wrap=False), self.condition)
        lay.addWidget(self.meta_box)
        lay.addStretch(1)
        self.add_btn.clicked.connect(self._add_images)
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.open_sel_btn.clicked.connect(self._open_selected)
        self.image_list.itemDoubleClicked.connect(lambda *_: self._open_selected())
        self.remove_img_btn.clicked.connect(self._remove_image)
        self.exclude_img_btn.clicked.connect(self._exclude_image)
        self.analyze_all_btn.clicked.connect(lambda: self.main.run_all("new"))
        self.batch_stop.clicked.connect(self.main.stop_batch)
        self.experiment.editingFinished.connect(self._meta)
        self.condition.editingFinished.connect(self._meta)
        self.pages.addWidget(w)

    def _add_images(self):
        pattern = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", self._last_dir, f"Images ({pattern})")
        if paths:
            self._last_dir = str(Path(paths[0]).parent)
            self.main.add_images([Path(p) for p in paths], open_first=True)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add all images of a folder", self._last_dir)
        if folder:
            self._last_dir = folder
            self.main.add_folder(Path(folder))

    def _open_selected(self):
        row = self.image_list.currentRow()
        if row >= 0:
            self.main.open_image_index(self.main.image_index(row))

    def _remove_image(self):
        row = self.image_list.currentRow()
        if row >= 0:
            self.main.remove_image(self.main.image_index(row))

    def ask_reason(self, what: str) -> str | None:
        reason, ok = QInputDialog.getItem(self, f"Exclude {what}", f"Why is this {what} left out? The reason is "
                                          "listed in Results.xlsx (sheet Excluded).",
                                          [CHOOSE_REASON] + list(EXCLUDE_REASONS), 0, True)
        reason = reason.strip()
        return reason if ok and reason and reason != CHOOSE_REASON else None

    def _exclude_image(self):
        row = self.image_list.currentRow()
        if row < 0:
            return
        reason = self.ask_reason("image")
        if reason:
            self.main.exclude_image(self.main.image_index(row), reason)

    def _open_analysis(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open analysis", self._last_dir, "Analysis (*.lta.json)")
        if path:
            self._last_dir = str(Path(path).parent)
            self.main.open_analysis(Path(path))

    def _meta(self):
        e, c = self.experiment.text().strip(), self.condition.text().strip()
        self.main.set_meta(e, c)
        if self.main.state.full is not None:
            self.meta_hint.setText(f"Experiment: {e or '-'}, condition: {c or '-'} (change them in step 1).")

    def set_busy(self, busy: bool, text: str = ""):
        """Loading an image."""
        for b in (self.add_btn, self.add_folder_btn, self.open_sel_btn):
            b.setEnabled(not busy)
        self.busy.setText(text if busy else "")
        self.set_working(busy, text, job="load")

    def set_batch(self, busy: bool, text: str = "", done: int = 0, total: int = 0):
        """Progress of 'Analyze all images' and 'Apply settings to all cells'."""
        self.batch_box.setVisible(busy)
        for b in (self.analyze_all_btn, self.apply_all_btn, self.remove_img_btn, self.open_sel_btn,
                  self.exclude_img_btn, self.remove_cell_btn,
                  self.edit_cell_btn, self.add_btn, self.add_folder_btn):
            b.setEnabled(not busy)
        if busy:
            self.batch_text.setText(text)
            self.batch_bar.setRange(0, max(total, 1))
            self.batch_bar.setValue(done)
            if self.step not in (0, 4):
                self.go(0)

    def show_images(self, entries: list[tuple[str, str, bool]]):
        """entries: (file name, cells text, is current)."""
        self.image_list.clear()
        self.empty_hint.setVisible(not entries)
        for i, (name, text, current) in enumerate(entries):
            mark = "▶ " if current else "   "
            status = f"[{text}] " if text else ""
            self.image_list.addItem(f"{mark}{status}{name}")
            self.image_list.item(i).setToolTip(f"{name}  {text}")
            if current:
                self.image_list.setCurrentRow(i)

    def show_info(self, stack, load_seconds: float, experiment: str, condition: str, shown: str | None = None,
                  hidden: bool = False):
        """Information on the open image. hidden: blind review, name and condition are not shown (D57)."""
        while self.info.rowCount():
            self.info.removeRow(0)
        self.experiment.setText(experiment)
        self.condition.setText(condition)
        self.meta_box.setVisible(not hidden)
        self.meta_box.setEnabled(stack is not None)          # nothing to describe without an open image
        self.reveal_btn.setVisible(hidden and stack is not None)
        if hidden:
            self.meta_hint.setText("Blind review: name and condition are hidden.")
        else:
            self.meta_hint.setText(f"Experiment: {experiment or '-'}, condition: {condition or '-'} "
                                   "(change them in step 1)." if stack is not None else "")
        if stack is None:
            self.open_warn.setText("")
            return
        nz, ny, nx = stack.shape_zyx
        dz, dy, dx = stack.voxel_um
        sz, sy, sx = stack.size_um
        rows = [("File", f"{shown} (name hidden)" if hidden else (shown or stack.display_name)),
                ("Voxels (x, y, z)", f"{nx} x {ny} x {nz}"),
                ("Voxel size", f"{dx * 1000:.1f} x {dy * 1000:.1f} x {dz * 1000:.0f} nm"),
                ("Size", f"{sx:.1f} x {sy:.1f} x {sz:.2f} µm"),
                ("Channels", ", ".join(c.short for c in stack.channels)),
                ("Loaded in", f"{load_seconds:.1f} s")]
        if "coverslip_first" in stack.info:
            rows.append(("Coverslip", "first plane" if stack.info["coverslip_first"] else "last plane"))
        for k, v in rows:
            val = label(v)
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.info.addRow(label(k, "hint", wrap=False), val)
        warnings = []
        if not stack.calibrated:
            warnings.append("The file has no complete voxel calibration. Distances are not in µm.")
        self.open_warn.setText("\n".join(warnings))

    # ------------------------------------------------------------------ 2 channels
    def _build_channels(self):
        w, lay = _page()
        self.channel_table = ChannelTable()
        lay.addWidget(self.channel_table)
        self.channel_note = label("", "hint")
        lay.addWidget(self.channel_note)
        self.channel_problems = label("", "warn")
        lay.addWidget(self.channel_problems)
        self.meta_hint = label("", "hint")
        lay.addWidget(self.meta_hint)
        lay.addWidget(label("Roles: Nucleus (DAPI), Actin (phalloidin), Target (the protein to localize, several "
                            "possible), Membrane (a membrane stain, then only it defines the cell edge), Other "
                            "(shown, helps to find the cell edge, not measured), Ignore (not used, not shown).",
                            "hint"))
        self.save_panel_btn = button("Save as panel ...", tip="Reuse these roles for images with the same channels")
        lay.addLayout(_row(self.save_panel_btn))
        lay.addWidget(label("Display", "title"))
        lay.addWidget(label("Only changes what you see, never the measured values.", "hint"))
        self.display_box = QVBoxLayout()
        lay.addLayout(self.display_box)
        lay.addStretch(1)
        self.channel_table.changed.connect(self._channels_changed)
        self.save_panel_btn.clicked.connect(self._save_panel)
        self.pages.addWidget(w)

    def populate_channels(self, stack, assignments, note: str):
        self.channel_table.populate(stack.channels, assignments)
        self.channel_note.setText(note)
        self._show_problems(assignments)
        self.rebuild_display()

    def rebuild_display(self):
        _clear(self.display_box)
        for layer in self.main.image_layers():
            if layer.metadata.get("role") == "Ignore":
                continue
            row = QHBoxLayout()
            vis = QCheckBox(layer.name)
            vis.setChecked(layer.visible)
            vis.toggled.connect(lambda on, l=layer: setattr(l, "visible", on))
            bright = QSlider(Qt.Horizontal)
            bright.setRange(10, 300)
            bright.setValue(int(round(100 * layer.metadata.get("brightness", 1.0))))
            bright.setToolTip("Brightness of this channel on screen")
            bright.valueChanged.connect(lambda v, l=layer: self.main.set_brightness(l, v / 100.0))
            row.addWidget(vis, 1)
            row.addWidget(bright, 2)
            self.display_box.addLayout(row)

    def rebuild_channel_toggles(self):
        """Steps 3 and 4: one checkbox per channel to hide the image and look at the masks alone."""
        for row in (self.cell_channels, self.comp_channels):
            _clear(row)
            row.addWidget(label("Image:", "hint", wrap=False))
            for layer in self.main.image_layers():
                if layer.metadata.get("role") == "Ignore":
                    continue
                cb = QCheckBox(layer.name)
                cb.setChecked(layer.visible)
                cb.toggled.connect(lambda on, l=layer: setattr(l, "visible", on))
                row.addWidget(cb)
            row.addStretch(1)

    def _show_problems(self, assignments):
        self.channel_problems.setText("\n".join(validate(assignments)))

    def _channels_changed(self):
        a = self.channel_table.assignments()
        self._show_problems(a)
        if [vars(x) for x in a] == [vars(x) for x in (self.main.state.assignments or [])]:
            return                               # editing finished without a change (D87)
        self.main.apply_assignments(a)
        self.rebuild_display()
        self.refresh()

    def _save_panel(self):
        if self.main.state.full is None:
            return
        default = " / ".join(a.label for a in self.channel_table.assignments() if a.role != "Ignore")
        name, ok = QInputDialog.getText(self, "Save panel", "Name of the panel:", text=default)
        if ok and name.strip():
            self.main.save_panel(name.strip())
            self.channel_note.setText(f"Panel '{name.strip()}' saved. It is applied automatically to images "
                                      "with the same channels.")

    # ------------------------------------------------------------------ 3 cell
    def _build_cell(self):
        w, lay = _page()
        lay.addWidget(label("The white line is the cell edge in the current plane. Go through the planes with "
                            "the slider below the image.", "hint"))
        self.cell_channels = QHBoxLayout()
        lay.addLayout(self.cell_channels)
        lay.addWidget(label("Remove other cells or debris", "title"))
        lay.addWidget(label("'Draw', click the corners around the part, double-click to close, then 'Apply'. "
                            "Removed parts turn black in all planes.", "hint"))
        self.exclude_btn = button("Draw", tip="Click the corners of a shape, double-click to close it")
        self.apply_ex_btn = button("Apply", tip="Finish drawing, the shapes turn black")
        self.undo_ex_btn = button("Undo last")
        self.clear_ex_btn = button("Clear all")
        lay.addLayout(_row(self.exclude_btn, self.apply_ex_btn, self.undo_ex_btn, self.clear_ex_btn))
        lay.addWidget(label("Cell edge", "title"))
        self.cell_factor = _factor_box("Below 1 the cell gets larger, above 1 smaller. 1 is the automatic value.")
        self.cell_defaults = button("Default", tip="Back to the value of the analysis settings")
        lay.addLayout(_row(label("Edge factor", "hint", wrap=False), self.cell_factor, self.cell_defaults))
        lay.addWidget(label("Below 1 the cell gets larger, above 1 smaller. Signal that fades over a few planes "
                            "above thin protrusions is mostly blur of the bright planes below; a lower factor "
                            "counts that blur as cell.", "hint"))
        self.cell_info = label("", "hint")
        lay.addWidget(self.cell_info)
        self.cell_error = label("", "warn")
        lay.addWidget(self.cell_error)
        lay.addStretch(1)
        self._cell_debounce = QTimer(self)
        self._cell_debounce.setSingleShot(True)
        self._cell_debounce.setInterval(500)
        self._cell_debounce.timeout.connect(lambda: self.main.compute_cell())
        self.exclude_btn.clicked.connect(self.main.start_exclusion)
        self.apply_ex_btn.clicked.connect(self.main.apply_exclusions)
        self.undo_ex_btn.clicked.connect(self.main.undo_exclusion)
        self.clear_ex_btn.clicked.connect(self.main.clear_exclusions)
        self.cell_factor.valueChanged.connect(lambda *_: self._cell_debounce.start())
        self.cell_defaults.clicked.connect(lambda: self.cell_factor.setValue(self._base.get("cell", 1.0)))
        self.pages.addWidget(w)

    def set_cell_computing(self, busy: bool, text: str = ""):
        self.set_working(busy, text or "Finding the cell ...", job="cell")

    # ------------------------------------------------------------------ 4 compartments
    def _build_compartments(self):
        w, lay = _page()
        lay.addWidget(label("Show", "title"))
        self.show_box = {}
        grid = QGridLayout()                      # three columns: fits the panel also with large text
        grid.setContentsMargins(0, 0, 0, 0)
        for k, (key, name, color) in enumerate(COMPARTMENT_TOGGLES):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {theme.readable(color)}; }}")
            cb.toggled.connect(self._toggled)
            self.show_box[key] = cb
            grid.addWidget(cb, k // 3, k % 3)
        lay.addLayout(grid)
        self.target_row = QHBoxLayout()
        lay.addLayout(self.target_row)
        self.target_box: dict[str, QCheckBox] = {}
        self.front_box = QCheckBox("Front direction")
        self.front_box.setStyleSheet(f"QCheckBox {{ color: {theme.readable('#FF8C00')}; }}")
        self.front_box.setToolTip("Arrow from the cell centre towards the actin-rich part (polarity). Long and "
                                  "bold for a clear front, short and thin when the cell is nearly even (D84)")
        self.front_box.toggled.connect(lambda *_: self.main.update_compartment_display())
        self.outline_box = QCheckBox("Outlines only")
        self.outline_box.setChecked(True)
        self.only_menu = QComboBox()
        self.only_menu.addItems(list(ONLY_ITEMS) + ["Own selection"])
        lay.addLayout(_row(self.outline_box, self.only_menu))
        self.comp_channels = QHBoxLayout()
        lay.addLayout(self.comp_channels)
        lay.addWidget(label("Cytoplasm is everything outside the nucleus, actin included. Untick the image "
                            "channels to see the masks alone. In 3D the masks are drawn as smoothed surfaces.",
                            "hint"))
        lay.addWidget(label("Edges", "title"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.factor = {}
        for key, text in (("nucleus", "Nucleus edge"), ("actin", "Actin"), ("actin_rich", "Actin-rich"),
                          ("target", "Target mask"), ("comparison", "Comparison mask")):
            sb = _factor_box("Threshold factor: below 1 makes the mask larger, above 1 smaller")
            sb.valueChanged.connect(self._factor_changed)
            self.factor[key] = sb
            form.addRow(label(text, "hint", wrap=False), sb)
        lay.addLayout(form)
        self.defaults_btn = button("Default", tip="Back to the values of the analysis settings")
        self.factors_all_btn = button("Use these factors for all cells",
                                      tip="Make these factors the starting values of the analysis")
        lay.addLayout(_row(self.defaults_btn, self.factors_all_btn))
        self.target_hint = label("", "hint")
        lay.addWidget(self.target_hint)
        self.comp_status = label("", "hint")
        lay.addWidget(self.comp_status)
        self.comp_notes = label("", "warn")       # shown in the alert line under the instruction
        self.comp_notes.setVisible(False)
        lay.addStretch(1)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(lambda: self.main.compute())
        self.defaults_btn.clicked.connect(self._defaults)
        self.factors_all_btn.clicked.connect(self.main.factors_as_default)
        self.outline_box.toggled.connect(lambda *_: self.main.update_compartment_display())
        self.only_menu.activated.connect(self._only)     # also when the shown entry is chosen again (D87)
        self.pages.addWidget(w)

    def set_targets(self, labels: list[str], comparisons: list[str] = ()):
        """One checkbox per target and comparison mask, and the front direction. A mask that was shown stays
        shown (D87)."""
        comparisons = list(comparisons)
        mode = getattr(self.main, "params", None)
        self.target_hint.setText(
            "The target mask marks the stained voxels. The shares in step 5 count the target signal inside it "
            "(Settings > Masks), so correcting it changes the numbers." if mode is None or mode.target_signal == "mask"
            else "The target mask marks the stained voxels for checking. The shares in step 5 use all of the "
                 "target signal and do not depend on it (Settings > Masks).")
        self.factor["comparison"].setEnabled(bool(comparisons))
        was = {k: cb.isChecked() for k, cb in self.target_box.items()}
        for cb in self.target_box.values():
            cb.hide()
            cb.setParent(None)
            cb.deleteLater()
        self.target_box = {}
        while self.target_row.count():
            self.target_row.takeAt(0)
        for lab in list(labels) + comparisons:
            cb = QCheckBox(f"{'Comparison' if lab in comparisons else 'Target'} mask {lab}")
            cb.setChecked(was.get(lab, False))
            cb.toggled.connect(self._toggled)
            self.target_box[lab] = cb
            self.target_row.addWidget(cb)
        self.target_row.addWidget(self.front_box)
        self.target_row.addStretch(1)

    def _only_state(self, index: int) -> dict[str, bool]:
        key = list(ONLY_ITEMS.values())[index]
        out = {k: key is None or k == key for k in self.show_box}
        out.update({f"target_{k}": key == "target" for k in self.target_box})
        return out

    def _only(self, index: int):
        if index >= len(ONLY_ITEMS):             # "Own selection" changes nothing
            return
        state = self._only_state(index)
        for k, cb in self.show_box.items():
            cb.blockSignals(True)
            cb.setChecked(state[k])
            cb.blockSignals(False)
        for k, cb in self.target_box.items():
            cb.blockSignals(True)
            cb.setChecked(state[f"target_{k}"])
            cb.blockSignals(False)
        self.main.update_compartment_display()

    def _toggled(self, *_):
        """A box ticked by hand: the menu names the matching entry or "Own selection" (D87)."""
        now = {k: cb.isChecked() for k, cb in self.show_box.items()}
        now.update({f"target_{k}": cb.isChecked() for k, cb in self.target_box.items()})
        index = next((i for i in range(len(ONLY_ITEMS)) if self._only_state(i) == now), len(ONLY_ITEMS))
        self.only_menu.blockSignals(True)
        self.only_menu.setCurrentIndex(index)
        self.only_menu.blockSignals(False)
        self.main.update_compartment_display()

    def visible_compartments(self) -> dict[str, bool]:
        out = {k: cb.isChecked() for k, cb in self.show_box.items()}
        out.update({f"target_{k}": cb.isChecked() for k, cb in self.target_box.items()})
        out["front"] = self.front_box.isChecked()
        return out

    def factors(self) -> dict[str, float]:
        f = {k: sb.value() for k, sb in self.factor.items()}
        f["cell"] = self.cell_factor.value()
        return f

    def set_factors(self, overrides: dict, base: dict | None = None):
        """Factors of one cell: its own values, otherwise the values of the analysis settings."""
        if base is not None:
            self._base = dict(base)
        boxes = dict(self.factor)
        boxes["cell"] = self.cell_factor
        for key, sb in boxes.items():
            sb.blockSignals(True)
            sb.setValue(float((overrides or {}).get(key, {}).get("factor", self._base.get(key, 1.0))))
            sb.blockSignals(False)

    def _factor_changed(self, *_):
        self._debounce.start()

    def _defaults(self):
        for key, sb in self.factor.items():
            sb.blockSignals(True)
            sb.setValue(self._base.get(key, 1.0))
            sb.blockSignals(False)
        self.main.compute()

    def set_computing(self, busy: bool, text: str = ""):
        self.set_working(busy, text or "Computing the compartments ...", job="compartments")

    # ------------------------------------------------------------------ 5 results
    def _build_results(self):
        w, lay = _page()
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(["", "Signal %", "Volume %", "Enrichment"])
        self.results.verticalHeader().setVisible(False)
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # full height: the panel scrolls, not the table
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        explain = ("Signal %: share of the target's total signal in the cell. Volume %: share of the cell volume. "
                   "Enrichment = Signal % / Volume %: 1 means as dense as the cell average, above 1 enriched. "
                   "Nucleus and cytoplasm add up to 100 %. Actin is the actin part of the cytoplasm. Zones overlap "
                   "the compartments and each other. Empty: no value (see the notes).")
        self.results.setToolTip(explain)
        lay.addWidget(self.results)
        lay.addWidget(label("Enrichment 1 = as dense as the cell average. Hover over the table for details.", "hint"))
        self.result_facts = label("", "hint")
        lay.addWidget(self.result_facts)
        self.result_notes = label("", "warn")     # shown in the alert line under the instruction
        self.result_notes.setVisible(False)
        self.accept_btn = button("Accept cell", primary=True, large=True, tip="Accept (Ctrl+Enter)")
        self.exclude_cell_btn = button("Exclude cell ...", large=True, tip="Leave this cell out, with a reason")
        self.next_image_btn = button("Next image", large=True,
                                     tip="The next image without a checked cell (Ctrl+Shift+Right)")
        self.accepted_info = label("", "ok")           # these four sit in the fixed area below the page
        self.next_cell_btn = button("Next cell in this image", tip="Another cell in the same image: start "
                                    "again in step 3")
        lay.addLayout(_row(self.next_cell_btn))
        lay.addWidget(label("Cells in this analysis", "title"))
        self.cell_list = QListWidget()
        self.cell_list.setMinimumHeight(120)
        lay.addWidget(self.cell_list)
        self.edit_cell_btn = button("Open cell", tip="Load the image and the settings of this cell again")
        self.remove_cell_btn = button("Remove cell")
        lay.addLayout(_row(self.edit_cell_btn, self.remove_cell_btn))
        self.apply_all_btn = button("Apply settings to all cells", tip="Compute every cell again whose settings "
                                    "differ from the current ones.")
        lay.addLayout(_row(self.apply_all_btn))
        self.save_btn = button("Save results ...", primary=True,
                               tip="Writes Results.xlsx, the control images and the analysis file into one folder")
        self.save_btn.setVisible(False)
        self.save_info = label("", "hint")
        lay.addWidget(self.save_info)
        lay.addStretch(1)
        self.accept_btn.clicked.connect(self.main.accept_cell)
        self.exclude_cell_btn.clicked.connect(self._exclude_cell)
        self.next_cell_btn.clicked.connect(self.main.next_cell)
        self.next_image_btn.clicked.connect(self.main.next_image)
        self.edit_cell_btn.clicked.connect(lambda: self.main.edit_cell(self.cell_list.currentRow()))
        self.cell_list.itemDoubleClicked.connect(lambda *_: self.main.edit_cell(self.cell_list.currentRow()))
        self.remove_cell_btn.clicked.connect(lambda: self.main.remove_cell(self.cell_list.currentRow()))
        self.save_btn.clicked.connect(self._save)
        self.apply_all_btn.clicked.connect(lambda: self.main.run_all("all"))
        self.pages.addWidget(w)

    def _exclude_cell(self):
        reason = self.ask_reason("cell")
        if reason:
            self.main.exclude_cell(reason)

    def show_cells(self, entries: list[str]):
        self.cell_list.clear()
        for i, e in enumerate(entries):
            self.cell_list.addItem(e)
            self.cell_list.item(i).setToolTip(e)

    def set_accepted_text(self, text: str, excluded: bool = False):
        self.accepted_info.setText(text)
        _set_role(self.accepted_info, "warn" if excluded else "ok")

    def show_results(self, result):
        rows = []
        for target, v in result.target_values.items():
            rows.append((f"{target}", None, None, None, True))
            for key, name in (("nucleus", "Nucleus"), ("cytoplasm", "Cytoplasm"), ("actin", "  Actin"),
                              ("actin_rich", "    Actin-rich"), ("cytoplasm_without_actin", "  Without actin")):
                rows.append((name, v[f"signal_{key}_pct"], v[f"volume_{key}_pct"], v[f"enrichment_{key}"], False))
            rows.append(("Zones (overlap)", None, None, None, True))
            for key, name in (("envelope_zone", "  Nuclear envelope zone"), ("envelope_inner", "    inner half"),
                              ("envelope_outer", "    outer half"), ("nucleus_interior", "  Nucleus interior"),
                              ("cortex_shell", "  Cortex shell"), ("basal_layer", "  Basal layer")):
                if f"signal_{key}_pct" in v:
                    rows.append((name, v[f"signal_{key}_pct"], v[f"volume_{key}_pct"], v[f"enrichment_{key}"], False))
            if "signal_front_pct" in v:
                vf = result.cell_values.get("volume_front_pct")
                rows.append(("  Front half", v.get("signal_front_pct"), vf, v.get("enrichment_front"), False))
                if "enrichment_rear" in v:
                    sf = v.get("signal_front_pct")
                    rows.append(("  Rear half", 100 - sf if sf == sf else None, 100 - vf if vf == vf else None,
                                 v.get("enrichment_rear"), False))
            for key in v:                             # D83: in the area of a comparison channel
                if key.startswith("signal_in_") and key.endswith("_pct"):
                    name = key[len("signal_in_"):-len("_pct")]
                    rows.append((f"  In {name} area", v[key], v.get(f"volume_{name}_area_pct"),
                                 v.get(f"enrichment_in_{name}"), False))
            rows.append(("  Target mask", v.get("target_mask_signal_pct"), v.get("target_mask_volume_pct"), None, False))
        if not self.main.state.accepted_current:      # a new result: can be accepted (again)
            self.accept_btn.setEnabled(True)
            self.accept_btn.setText("Accept cell")
        self.results.setRowCount(len(rows))
        for r, (name, s, vol, e, head) in enumerate(rows):
            item = QTableWidgetItem(name)
            if head:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.results.setItem(r, 0, item)
            for c, val, fmt in ((1, s, "{:.1f}"), (2, vol, "{:.1f}"), (3, e, "{:.2f}")):
                txt = "" if val is None or val != val else fmt.format(val)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.results.setItem(r, c, it)
        self.results.resizeRowsToContents()
        self.results.setFixedHeight(self.results.horizontalHeader().height()
                                    + sum(self.results.rowHeight(r) for r in range(self.results.rowCount())) + 6)
        cv = result.cell_values
        facts = [f"Cell {_fmt(cv.get('cell_volume_um3'))} µm³, nucleus {_fmt(cv.get('nucleus_volume_um3'))} µm³, "
                 f"height {_fmt(cv.get('cell_height_um'), '{:.1f}')} µm"]
        p = self.main.params
        facts.append("Shares count the target signal inside the target mask (Settings > Masks)."
                     if p.target_signal == "mask" else "Shares count all target signal in the cell (Settings > Masks).")
        pi = cv.get("polarity_index")
        if pi is not None and pi == pi:
            clear = "clear front" if pi >= p.polarity_clear_index else \
                f"no clear front (below {p.polarity_clear_index:g}), the front values mean little"
            facts.append(f"Polarity index {pi:.2f}: {clear}. Actin in the front half {_fmt(cv.get('actin_front_pct'))} %")
        for target, v in result.target_values.items():
            r = v.get("rear_to_front_ratio")
            if r is not None and r == r:
                facts.append(f"{target}: rear to front {r:.2f} (density, 1 = even, above 1 = more in the rear)")
            pk = v.get("envelope_peak_um")
            if pk is not None and pk == pk:
                side = "outside" if pk > 0 else "inside"
                facts.append(f"{target}: highest enrichment at {pk:+.2f} µm from the nucleus edge ({side}, "
                             f"enrichment {_fmt(v.get('envelope_peak_enrichment'), '{:.2f}')})")
        for target, v in result.target_values.items():
            if "pcc_actin" in v:
                m1 = v.get("m1_actin")
                man = f", M1 {m1:.2f}, M2 {_fmt(v.get('m2_actin'), '{:.2f}')}" if m1 is not None and m1 == m1 else \
                    " (no Costes threshold, the channels are not positively correlated)"
                facts.append(f"{target}: PCC with actin {_fmt(v['pcc_actin'], '{:.2f}')}{man}")
        for target, v in result.target_values.items():
            lo, hi = v.get("enrichment_nucleus_min"), v.get("enrichment_nucleus_max")
            if lo is not None and lo == lo and hi is not None and hi == hi:
                edges = "the cell, nucleus and target edges" if p.target_signal == "mask" else "the cell and nucleus edges"
                facts.append(f"{target}: enrichment nucleus {lo:.2f} to {hi:.2f}, signal nucleus "
                             f"{_fmt(v.get('signal_nucleus_pct_min'), '{:.1f}')} to "
                             f"{_fmt(v.get('signal_nucleus_pct_max'), '{:.1f}')} % when {edges} "
                             "move (factor 0.8 to 1.2)")
        if not result.info.get("channel_offsets"):
            facts.append("Zones and heights without channel offset correction (Settings > Channel offset).")
        self.result_facts.setText("\n".join(facts))
        self.result_notes.setText("\n".join(result.notes))
        self.refresh()

    def _save(self):
        self.main.save_results()

    def ask_folder(self) -> Path | None:
        folder = QFileDialog.getExistingDirectory(self, "Folder for this analysis", self._last_dir)
        return Path(folder) if folder else None
