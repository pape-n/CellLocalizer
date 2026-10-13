"""Cell viewer (D71): look at the cells of a saved analysis and export pictures of them.

Left the cells of the analysis, in the middle the picture, right what it shows. Plane and section
pictures are drawn by core.figure, so what is seen is exactly what is saved. The 3D view uses a napari
canvas with the channels as volumes and the masks as smoothed surfaces (gui.surfaces), saved as a
screenshot. The control images of the analysis stay as they are.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressDialog,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from localization_tool import APP_NAME
from localization_tool.core import cellview, figure, project_file
from localization_tool.core.channels import DISPLAY_COLORS

MASK_COLOR_CHOICES = {"White": "#FFFFFF", "Light blue": "#00C8FF", "Slate": "#8A929C", "Lime": "#3CFF5A",
                      "Amber": "#FF9F1C", "Violet": "#C77DFF", "Pink": "#FF5C8A", **DISPLAY_COLORS}
SETTINGS_FILE = "viewer_settings.json"        # last settings of the viewer, in the folder of the kept masks
from localization_tool.gui import theme
from localization_tool.gui.widgets import button, label, separator

MODE_TITLES = {"plane": "Plane", "section": "Section", "3d": "3D"}


def _qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class _Preview(QLabel):
    """The picture, scaled to the available space without changing its aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background: black;")
        self._image: QImage | None = None

    def set_rgb(self, rgb: np.ndarray | None) -> None:
        self._image = _qimage(rgb) if rgb is not None else None
        self._fit()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._image is None:
            self.clear()
            return
        pm = QPixmap.fromImage(self._image)
        # fill the space; enlarged without smoothing (voxels stay voxels), reduced with smoothing
        grow = pm.width() < self.width() and pm.height() < self.height()
        pm = pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation if grow else Qt.SmoothTransformation)
        self.setPixmap(pm)


class CellViewer(QWidget):
    def __init__(self, analysis_path: Path):
        super().__init__()
        self.setObjectName("page")
        self.setAttribute(Qt.WA_StyledBackground, True)
        theme.style(self)
        theme.fit_to_screen(self, 1400, 860)
        self.path = Path(analysis_path)
        self.base_dir = self.path.parent
        self.af = project_file.load(self.path)
        self.setWindowTitle(f"Cell viewer: {self.base_dir.name} ({APP_NAME})")
        self.cell: cellview.CellData | None = None
        self.settings = figure.FigureSettings()
        self.mode = "plane"
        self._cells: list = []
        self._qt3d = None
        self._model3d = None
        self._surfaces: dict = {}
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(60)
        self._render_timer.timeout.connect(self._render)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self._build_list(), 0)
        self.stack = QStackedWidget()
        self.preview = _Preview()
        self.stack.addWidget(self.preview)
        root.addWidget(self.stack, 1)
        side = QScrollArea()
        side.setWidgetResizable(True)
        side.setFrameShape(QFrame.NoFrame)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side.setWidget(self._build_controls())
        side.setMinimumWidth(460)
        side.viewport().setStyleSheet(f"background: {theme.TOKENS['bg']};")
        root.addWidget(side, 0)
        self._remembered = self.base_dir / cellview.CACHE_DIR / SETTINGS_FILE
        try:
            if self._remembered.exists():
                self._apply_settings(figure.FigureSettings.load(self._remembered))
        except (OSError, ValueError, TypeError, KeyError):   # a damaged file: start with the defaults
            pass
        self._fill_list()

    # ------------------------------------------------------------------ left: cells
    def _build_list(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(300)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(label("Cells", "title"))
        lay.addWidget(label(str(self.path), "hint"))
        self.only_accepted = QCheckBox("Accepted cells only")
        self.only_accepted.setChecked(True)
        self.only_accepted.toggled.connect(self._fill_list)
        lay.addWidget(self.only_accepted)
        self.cell_list = QListWidget()
        self.cell_list.currentRowChanged.connect(self._open_row)
        lay.addWidget(self.cell_list, 1)
        self.cell_info = label("", "hint")
        lay.addWidget(self.cell_info)
        return w

    def _fill_list(self, *_):
        self._cells = cellview.cells_of(self.af, self.only_accepted.isChecked())
        self.cell_list.blockSignals(True)
        self.cell_list.clear()
        for im, c in self._cells:
            cond = f"{im.condition}  " if im.condition else ""
            item = QListWidgetItem(f"{cond}{im.path.stem} / {c.name}")
            item.setToolTip(f"{im.path}\nstatus: {c.status}")
            self.cell_list.addItem(item)
        self.cell_list.blockSignals(False)
        if not self._cells:
            self.cell_info.setText("No cells. Untick 'Accepted cells only' to see cells that are still to check.")

    def _open_row(self, row: int) -> None:
        if not (0 <= row < len(self._cells)):
            return
        im, spec = self._cells[row]
        if not im.path.exists():
            self.cell_info.setText(f"The image is not at its stored place:\n{im.path}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.cell_info.setText("Loading the image and the masks ...")
        QApplication.processEvents()
        try:
            self.cell = cellview.load_cell(self.af, im, spec, self.base_dir)
        except Exception as exc:              # a damaged file, a missing channel: say it, keep the window
            self.cell = None
            self.cell_info.setText(f"Could not open this cell:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        from localization_tool.core.batch import needs_update

        stale = needs_update(spec, self.af.params)
        self.cell_info.setText(f"{self.cell.name}\n{im.experiment} {im.condition}  ·  {spec.status}"
                               + ("\nComputed with other settings than the analysis now has." if stale else ""))
        self._surfaces = {}
        self._sync_settings_to_cell()
        self._rebuild_channel_rows()
        self._rebuild_mask_rows()
        nz = self.cell.stack.shape_zyx[0]
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(1, nz)
        self.z_slider.setValue(self.cell.default_z + 1)
        self.z_slider.blockSignals(False)
        self.settings.z = self.cell.default_z
        self._update_mode_widgets()
        self.schedule()

    def _sync_settings_to_cell(self) -> None:
        """Keep colours and contrast per channel name across cells (comparable pictures), new channels
        get the colour of the analysis and an automatic contrast."""
        c = self.cell
        old = {v.label: v for v in self.settings.channels}
        auto = figure.default_channel_views(c.stack.data, c.assignments, c.view_masks["cell"])
        self.settings.channels = [old.get(v.label, v) for v in auto]
        known = {m.name: m for m in self.settings.masks}
        self.settings.masks = [known.get(m.name, m) for m in figure.default_mask_views(self._mask_names())]

    def _mask_names(self) -> list[str]:
        """Panel order: target masks, compartments, zones."""
        vm = self.cell.view_masks
        return (sorted(n for n in vm if n.startswith("target_")) + [n for n in figure.PANEL_MASKS if n in vm]
                + [n for n in figure.PANEL_ZONES if n in vm])

    # ------------------------------------------------------------------ right: controls
    def _build_controls(self) -> QWidget:
        w = QWidget()
        w.setObjectName("page")                     # dark background also inside the scroll area
        w.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)
        lay.addWidget(label("View", "title"))
        modes = QHBoxLayout()
        self.mode_btns = {}
        for key, title in MODE_TITLES.items():
            b = button(title, tip={"plane": "One plane, as recorded",
                                   "section": "Vertical section through the cell at any angle, coverslip at the bottom",
                                   "3d": "The whole cell in 3D, drag to turn"}[key])
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            modes.addWidget(b)
            self.mode_btns[key] = b
        modes.addStretch(1)
        lay.addLayout(modes)
        self.view_form = QFormLayout()
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._z_changed)
        self.z_text = label("", "hint", wrap=False)
        zrow = QHBoxLayout()
        zrow.addWidget(self.z_slider, 1)
        zrow.addWidget(self.z_text)
        self.view_form.addRow("Plane", zrow)
        self.angle = QSlider(Qt.Horizontal)
        self.angle.setRange(0, 179)
        self.angle.valueChanged.connect(self._angle_changed)
        self.angle_text = label("0°", "hint", wrap=False)
        arow = QHBoxLayout()
        arow.addWidget(self.angle, 1)
        arow.addWidget(self.angle_text)
        self.view_form.addRow("Direction", arow)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-50, 50)
        self.offset.setSingleStep(0.5)
        self.offset.setDecimals(1)
        self.offset.setSuffix(" µm")
        self.offset.valueChanged.connect(self._offset_changed)
        self.view_form.addRow("Shift from centre", self.offset)
        lay.addLayout(self.view_form)

        lay.addWidget(separator())
        head = QHBoxLayout()
        head.addWidget(label("Channels", "title"))
        head.addStretch(1)
        auto = button("Auto contrast", flat=True, tip="0 to the 99.7th percentile inside the cell, for this cell")
        auto.clicked.connect(self._auto_contrast)
        head.addWidget(auto)
        lay.addLayout(head)
        self.chan_grid = QGridLayout()
        self.chan_grid.setHorizontalSpacing(6)
        lay.addLayout(self.chan_grid)

        lay.addWidget(separator())
        lay.addWidget(label("Masks and zones", "title"))
        self.mask_grid = QGridLayout()
        self.mask_grid.setHorizontalSpacing(6)
        lay.addLayout(self.mask_grid)

        lay.addWidget(separator())
        lay.addWidget(label("Picture", "title"))
        form = QFormLayout()
        self.scale = QSpinBox()
        self.scale.setRange(1, 8)
        self.scale.setValue(self.settings.scale)
        self.scale.setToolTip("Size of the saved picture: each voxel becomes this many pixels in each direction "
                              "(no smoothing, no invented detail). The preview always fills the window, so it does "
                              "not change; the size of the saved picture is shown below. In 3D: resolution of the "
                              "saved picture.")
        self.scale.valueChanged.connect(self._picture_changed)
        form.addRow("Pixels per voxel", self.scale)
        self.crop = QCheckBox("Only the cell (1 µm around it)")
        self.crop.setChecked(self.settings.crop_to_cell)
        self.crop.setToolTip("Off: the whole analysed region. The saved picture is always what the preview shows.")
        self.crop.toggled.connect(self._picture_changed)
        form.addRow("Frame", self.crop)
        self.size_info = label("", "hint")
        form.addRow("Saved picture", self.size_info)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        form.addRow("Resolution (dpi)", self.dpi)
        self.bar = QDoubleSpinBox()
        self.bar.setRange(0, 200)
        self.bar.setDecimals(1)
        self.bar.setSpecialValueText("automatic")
        self.bar.setSuffix(" µm")
        self.bar.valueChanged.connect(self._picture_changed)
        form.addRow("Scale bar", self.bar)
        self.bar_on = QCheckBox("Scale bar")
        self.bar_on.setChecked(True)
        self.bar_text = QCheckBox("Length as text")
        self.bar_text.setChecked(True)
        self.legend = QCheckBox("Legend")
        for cb in (self.bar_on, self.bar_text, self.legend):
            cb.toggled.connect(self._picture_changed)
        checks = QHBoxLayout()
        for cb in (self.bar_on, self.bar_text, self.legend):
            checks.addWidget(cb)
        checks.addStretch(1)
        form.addRow("", checks)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("e.g. WT, treated (optional)")
        self.title_edit.textChanged.connect(self._picture_changed)
        form.addRow("Title", self.title_edit)
        lay.addLayout(form)

        lay.addWidget(separator())
        lay.addWidget(label("Export", "title"))
        self.png_btn = button("Save picture (PNG) ...", primary=True)
        self.png_btn.clicked.connect(self.export_png)
        self.tif_btn = button("Save values (TIFF) ...",
                              tip="The recorded values of the shown plane or section, one channel per page, in µm")
        self.tif_btn.clicked.connect(self.export_tiff)
        lay.addWidget(self.png_btn)
        lay.addWidget(self.tif_btn)
        self.batch_btn = button("Export all cells ...", tip="This view and these settings for every cell, "
                                "sorted by condition (plane and section)")
        self.batch_btn.clicked.connect(self.export_all)
        lay.addWidget(self.batch_btn)
        trow = QHBoxLayout()
        save_t = button("Save settings ...", flat=True, tip="Colours, contrast, masks and picture settings as a file, "
                        "e.g. for the same look in another analysis. The viewer also remembers its last settings "
                        "for this analysis by itself.")
        save_t.clicked.connect(self.save_template)
        load_t = button("Load settings ...", flat=True)
        load_t.clicked.connect(self.load_template)
        trow.addWidget(save_t)
        trow.addWidget(load_t)
        trow.addStretch(1)
        lay.addLayout(trow)
        self.export_info = label("", "hint")
        lay.addWidget(self.export_info)
        lay.addStretch(1)
        self._update_mode_widgets()
        return w

    def _rebuild_channel_rows(self) -> None:
        _clear(self.chan_grid)
        for r, cv in enumerate(self.settings.channels):
            cb = QCheckBox(cv.label)
            cb.setChecked(cv.visible)
            cb.toggled.connect(lambda on, v=cv: (setattr(v, "visible", on), self.schedule()))
            color = QComboBox()
            color.setMaximumWidth(120)
            for name, hexval in DISPLAY_COLORS.items():
                color.addItem(name, hexval)
            if color.findData(cv.color) < 0:
                color.addItem(cv.color, cv.color)
            color.setCurrentIndex(color.findData(cv.color))
            color.currentIndexChanged.connect(lambda _i, v=cv, c=color: (setattr(v, "color", c.currentData()),
                                                                         self.schedule()))
            lo, hi = _num_box(cv.low), _num_box(cv.high)
            lo.setToolTip("Black below this value")
            hi.setToolTip("Full colour from this value")
            lo.valueChanged.connect(lambda x, v=cv: (setattr(v, "low", x), self.schedule()))
            hi.valueChanged.connect(lambda x, v=cv: (setattr(v, "high", x), self.schedule()))
            self.chan_grid.addWidget(cb, r, 0)
            self.chan_grid.addWidget(color, r, 1)
            self.chan_grid.addWidget(lo, r, 2)
            self.chan_grid.addWidget(hi, r, 3)

    def _rebuild_mask_rows(self) -> None:
        _clear(self.mask_grid)
        r = 0
        zone_start = next((i for i, mv in enumerate(self.settings.masks) if mv.name in figure.PANEL_ZONES), None)
        for i, mv in enumerate(self.settings.masks):
            if i == zone_start and i > 0:
                line = separator()
                self.mask_grid.addWidget(line, r, 0, 1, 3)
                r += 1
            cb = QCheckBox(figure.mask_title(mv.name))
            cb.setChecked(mv.visible)
            cb.toggled.connect(lambda on, v=mv: (setattr(v, "visible", on), self._masks_changed()))
            color = QComboBox()
            color.setMaximumWidth(120)
            for name, hexval in MASK_COLOR_CHOICES.items():
                color.addItem(name, hexval)
            if color.findData(mv.color.upper()) < 0 and color.findData(mv.color) < 0:
                color.addItem(mv.color, mv.color)
            color.setCurrentIndex(max(color.findData(mv.color.upper()), color.findData(mv.color)))
            color.currentIndexChanged.connect(lambda _i, v=mv, c=color: self._mask_color(v, c.currentData()))
            style = QComboBox()
            style.addItem("Outline", "outline")
            style.addItem("Area", "fill")
            style.setCurrentIndex(0 if mv.style == "outline" else 1)
            style.currentIndexChanged.connect(lambda _i, v=mv, s=style: self._mask_style(v, s.currentData()))
            self.mask_grid.addWidget(cb, r, 0)
            self.mask_grid.addWidget(color, r, 1)
            self.mask_grid.addWidget(style, r, 2)
            r += 1

    def _mask_color(self, mv, color: str) -> None:
        mv.color = color
        layer = getattr(self, "_surface_layers", {}).get(mv.name)
        if layer is not None:
            layer.colormap = _surface_cmap(color)
        self.schedule()

    def _mask_style(self, mv, style: str) -> None:
        mv.style = style
        layer = getattr(self, "_surface_layers", {}).get(mv.name)
        if layer is not None:
            layer.opacity = _surface_opacity(style)
        self.schedule()

    # ------------------------------------------------------------------ changes
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != "3d":
            self.settings.mode = mode
        self._update_mode_widgets()
        self.schedule()

    def _update_mode_widgets(self) -> None:
        for k, b in self.mode_btns.items():
            b.setChecked(k == self.mode)
        has = self.cell is not None
        _row_visible(self.view_form, 0, self.mode == "plane")
        _row_visible(self.view_form, 1, self.mode == "section")
        _row_visible(self.view_form, 2, self.mode == "section")
        for b in (self.png_btn, self.batch_btn, self.tif_btn):
            b.setEnabled(has)
        self.tif_btn.setEnabled(has and self.mode != "3d")
        self.batch_btn.setEnabled(has and self.mode != "3d")
        self.batch_btn.setToolTip("This view for every cell, sorted by condition" if self.mode != "3d" else
                                  "3D pictures are saved one cell at a time")
        self.stack.setCurrentIndex(1 if self.mode == "3d" and self.stack.count() > 1 else 0)

    def _z_changed(self, v: int) -> None:
        self.settings.z = v - 1
        if self.cell is not None:
            self.z_text.setText(f"{v} of {self.cell.stack.shape_zyx[0]}")
        self.schedule()

    def _angle_changed(self, v: int) -> None:
        self.settings.angle_deg = float(v)
        self.angle_text.setText(f"{v}°")
        self.schedule()

    def _offset_changed(self, v: float) -> None:
        self.settings.offset_um = float(v)
        self.schedule()

    def _picture_changed(self, *_):
        s = self.settings
        s.scale = self.scale.value()
        s.scale_bar_um = self.bar.value()
        s.show_scale_bar = self.bar_on.isChecked()
        s.show_scale_text = self.bar_text.isChecked()
        s.show_legend = self.legend.isChecked()
        s.crop_to_cell = self.crop.isChecked()
        s.title = self.title_edit.text().strip()
        self.schedule()

    def _masks_changed(self) -> None:
        self.schedule()

    def _auto_contrast(self) -> None:
        if self.cell is None:
            return
        self.settings.channels = figure.default_channel_views(self.cell.stack.data, self.cell.assignments,
                                                              self.cell.view_masks["cell"])
        self._rebuild_channel_rows()
        self.schedule()

    def schedule(self) -> None:
        self._render_timer.start()

    # ------------------------------------------------------------------ drawing
    def current_view(self) -> figure.View | None:
        c = self.cell
        if c is None or self.mode == "3d":
            return None
        return figure.make_view(c.stack.data, c.view_masks, c.voxel_um, self.settings, c.coverslip_first, c.default_z)

    def _render(self) -> None:
        if self.cell is None:
            self.preview.set_rgb(None)
            return
        if self.mode == "3d":
            self._render_3d()
            return
        try:
            view = self.current_view()
            rgb = figure.render(view, self.settings)
            self.preview.set_rgb(rgb)
            self.export_info.setText(view.caption + " (from the image metadata)")
            self._show_size(rgb.shape[1], rgb.shape[0])
        except ValueError as exc:                # e.g. the section line misses the image
            self.preview.set_rgb(None)
            self.export_info.setText(str(exc))
        self._remember()

    def _show_size(self, w: int, h: int) -> None:
        dpi = max(1, self.dpi.value())
        self.size_info.setText(f"{w} × {h} px, {w / dpi * 2.54:.1f} × {h / dpi * 2.54:.1f} cm at {dpi} dpi")

    def _remember(self) -> None:
        """The last settings of this analysis, used again when the viewer opens next time."""
        try:
            self._remembered.parent.mkdir(parents=True, exist_ok=True)
            self.settings.save(self._remembered)
        except (OSError, ValueError):            # read-only folder: the viewer works without
            pass

    def _ensure_3d(self):
        if self._qt3d is not None:
            return
        from napari.components import ViewerModel
        from napari.qt import QtViewer

        self._model3d = ViewerModel(title="Cell viewer 3D", ndisplay=3)
        try:
            self._model3d.theme = theme.register_napari_theme()
        except Exception:
            pass
        self._qt3d = QtViewer(self._model3d)
        self.stack.addWidget(self._qt3d)
        # light from the camera, so every side looks the same when the cell is turned (feedback 13.10.)
        self._model3d.camera.events.angles.connect(self._headlight, position="last")

    def _render_3d(self) -> None:
        from localization_tool.gui.surfaces import mask_surface

        self._ensure_3d()
        self.stack.setCurrentIndex(1)
        c = self.cell
        m = self._model3d
        key = id(c)
        if getattr(self, "_layers_key", None) != key:
            m.layers.clear()
            for i, cv in enumerate(self.settings.channels):
                m.add_image(np.asarray(c.stack.data[i]), name=cv.label, scale=c.voxel_um, blending="additive",
                            rendering="mip", colormap=_cmap(cv.color, cv.label))
            self._layers_key = key
            self._surface_layers = {}
            m.reset_view()
        for i, cv in enumerate(self.settings.channels):
            if cv.label in m.layers:
                lay = m.layers[cv.label]
                lay.visible = cv.visible
                lay.contrast_limits = (cv.low, max(cv.high, cv.low + 1e-6))
                lay.colormap = _cmap(cv.color, cv.label)
        for mv in self.settings.masks:
            layer = self._surface_layers.get(mv.name)
            if mv.visible and layer is None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    mesh = self._surfaces.get(mv.name) or mask_surface(c.view_masks[mv.name], c.voxel_um)
                finally:
                    QApplication.restoreOverrideCursor()
                self._surfaces[mv.name] = mesh
                if mesh is None:
                    continue
                layer = m.add_surface(mesh, name=f"3D {mv.name}", opacity=_surface_opacity(mv.style),
                                      colormap=_surface_cmap(mv.color), shading="smooth", blending="translucent")
                self._surface_layers[mv.name] = layer
                self._headlight()
            if layer is not None:
                layer.visible = mv.visible
        self.export_info.setText("3D: drag to turn, scroll to zoom. The picture is saved as shown.")
        self._show_size(self._qt3d.width() * self.settings.scale, self._qt3d.height() * self.settings.scale)

    def _headlight(self, event=None) -> None:
        """Even lighting of the mask surfaces: the light comes from the camera, mostly ambient."""
        if self._qt3d is None:
            return
        view = -np.array(self._model3d.camera.view_direction, float)[::-1]
        for layer in getattr(self, "_surface_layers", {}).values():
            try:
                sf = self._qt3d.canvas.layer_to_visual[layer].node.shading_filter
            except (AttributeError, KeyError):
                continue
            if sf is None:
                continue
            sf.light_dir = tuple(view)
            sf.ambient_light = (1, 1, 1, 0.45)
            sf.diffuse_light = (1, 1, 1, 0.55)
            sf.specular_light = (1, 1, 1, 0.05)

    # ------------------------------------------------------------------ export
    def _default_name(self, suffix: str) -> str:
        c = self.cell
        return figure.file_stem(f"{c.condition}_{c.name.replace(' / ', '_')}_{self.mode}") + suffix

    def export_png(self) -> None:
        if self.cell is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save picture", str(self.base_dir / self._default_name(".png")),
                                              "PNG (*.png)")
        if not path:
            return
        if self.mode == "3d":
            figure.save_png(self.screenshot_3d(), path, self.dpi.value())
        else:
            figure.save_png(figure.render(self.current_view(), self.settings), path, self.dpi.value())
        self.export_info.setText(f"Saved {Path(path).name}")

    def screenshot_3d(self) -> np.ndarray:
        """The 3D view as shown, at 'Pixels per voxel' times the window resolution. napari enlarges the
        canvas for this and would show more of the scene around the cell; the zoom is raised by the same
        factor while the picture is taken, so the frame is the one on screen."""
        k = float(self.settings.scale)
        cam = self._model3d.camera
        zoom = cam.zoom
        try:
            cam.zoom = zoom * k
            img = self._qt3d.screenshot(flash=False, scale=k)
        finally:
            cam.zoom = zoom
        return np.asarray(img)[..., :3]

    def export_tiff(self) -> None:
        view = self.current_view()
        if view is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save values", str(self.base_dir / self._default_name(".tif")),
                                              "TIFF (*.tif)")
        if path:
            figure.save_tiff(view, [v.label for v in self.settings.channels], path)
            self.export_info.setText(f"Saved {Path(path).name} (recorded values, not the colours)")

    def export_all(self) -> None:
        if self.cell is None or self.mode == "3d":
            return
        opts = _BatchDialog(self)
        if not opts.exec_():
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the pictures", str(self.base_dir))
        if not folder:
            return
        folder = Path(folder)
        cells = cellview.cells_of(self.af, opts.accepted.isChecked())
        prog = QProgressDialog("Exporting ...", "Stop", 0, len(cells), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        settings = figure.FigureSettings.from_dict(self.settings.to_dict())
        listing, failed = [], []
        for k, (im, spec) in enumerate(cells):
            prog.setValue(k)
            prog.setLabelText(f"{im.path.stem} / {spec.name} ({k + 1} of {len(cells)})")
            QApplication.processEvents()
            if prog.wasCanceled():
                break
            try:
                cell = cellview.load_cell(self.af, im, spec, self.base_dir)
                s = figure.FigureSettings.from_dict(settings.to_dict())
                if not opts.same_contrast.isChecked():
                    auto = figure.default_channel_views(cell.stack.data, cell.assignments, cell.view_masks["cell"])
                    by = {v.label: v for v in auto}
                    for v in s.channels:
                        if v.label in by:
                            v.low, v.high = by[v.label].low, by[v.label].high
                if s.mode == "plane":
                    s.z = cell.default_z if opts.own_plane.isChecked() else min(settings.z, cell.stack.shape_zyx[0] - 1)
                if opts.titles.isChecked():
                    s.title = f"{im.condition}  {im.path.stem} / {spec.name}".strip()
                view = figure.make_view(cell.stack.data, cell.view_masks, cell.voxel_um, s, cell.coverslip_first,
                                        cell.default_z)
                sub = folder / figure.file_stem(im.condition or "no condition")
                sub.mkdir(parents=True, exist_ok=True)
                stem = figure.file_stem(f"{im.path.stem}_{spec.name}_{s.mode}")
                if opts.png.isChecked():
                    figure.save_png(figure.render(view, s), sub / f"{stem}.png", self.dpi.value())
                if opts.tif.isChecked():
                    figure.save_tiff(view, [v.label for v in s.channels], sub / f"{stem}.tif")
                listing.append({"condition": im.condition, "experiment": im.experiment, "image": im.path.name,
                                "cell": spec.name, "status": spec.status, "view": view.caption, "file": f"{sub.name}/{stem}",
                                **{f"contrast_{v.label}": f"{v.low:g}-{v.high:g}" for v in s.channels}})
            except Exception as exc:
                failed.append(f"{im.path.name} / {spec.name}: {exc}")
        prog.setValue(len(cells))
        settings.save(folder / "figure_settings.json")
        if listing:
            with open(folder / "exported_cells.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(listing[0]))
                w.writeheader()
                w.writerows(listing)
        text = f"{len(listing)} cell(s) exported to {folder}."
        if failed:
            text += "\n\nNot exported:\n" + "\n".join(failed[:10])
        QMessageBox.information(self, APP_NAME, text)
        self.export_info.setText(f"{len(listing)} cell(s) exported")

    def save_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", str(self.base_dir / "figure_settings.json"),
                                              "Settings (*.json)")
        if path:
            self.settings.save(path)

    def load_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", str(self.base_dir), "Settings (*.json)")
        if not path:
            return
        try:
            loaded = figure.FigureSettings.load(path)
        except (ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, APP_NAME, f"These settings cannot be read:\n{exc}")
            return
        self._apply_settings(loaded)

    def _apply_settings(self, loaded: figure.FigureSettings) -> None:
        self.settings = loaded
        self.mode = loaded.mode
        for box, value in ((self.scale, loaded.scale), (self.bar, loaded.scale_bar_um)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        for cb, value in ((self.bar_on, loaded.show_scale_bar), (self.bar_text, loaded.show_scale_text),
                          (self.legend, loaded.show_legend), (self.crop, loaded.crop_to_cell)):
            cb.blockSignals(True)
            cb.setChecked(value)
            cb.blockSignals(False)
        self.title_edit.blockSignals(True)
        self.title_edit.setText(loaded.title)
        self.title_edit.blockSignals(False)
        self.angle.setValue(int(loaded.angle_deg))
        self.offset.setValue(loaded.offset_um)
        if self.cell is not None:
            self._sync_settings_to_cell()
            self._rebuild_channel_rows()
            self._rebuild_mask_rows()
        self._update_mode_widgets()
        self.schedule()


class _BatchDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        theme.style(self)
        self.setWindowTitle("Export all cells")
        lay = QVBoxLayout(self)
        lay.addWidget(label("Every cell is shown with the current view, colours and masks.", "hint"))
        self.accepted = QCheckBox("Accepted cells only")
        self.accepted.setChecked(True)
        self.same_contrast = QCheckBox("Same contrast for all cells (needed to compare brightness between cells)")
        self.same_contrast.setChecked(True)
        self.own_plane = QCheckBox("Plane view: each cell at its own middle plane (as in the control images)")
        self.own_plane.setChecked(True)
        self.titles = QCheckBox("Condition and cell name as title")
        self.png = QCheckBox("Pictures (PNG)")
        self.png.setChecked(True)
        self.tif = QCheckBox("Values (TIFF)")
        for cb in (self.accepted, self.same_contrast, self.own_plane, self.titles, self.png, self.tif):
            lay.addWidget(cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Choose folder ...")
        buttons.button(QDialogButtonBox.Ok).setProperty("role", "primary")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


# ---------------------------------------------------------------------------------- helpers
def _num_box(value: float) -> QDoubleSpinBox:
    b = QDoubleSpinBox()
    b.setRange(-1e9, 1e9)
    b.setDecimals(0 if abs(value) >= 100 else 2)
    b.setValue(value)
    b.setMinimumWidth(84)
    return b


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()                             # gone at once, deleted later
            w.setParent(None)
            w.deleteLater()


def _row_visible(form: QFormLayout, row: int, visible: bool) -> None:
    try:
        form.setRowVisible(row, visible)
    except AttributeError:                    # Qt before 6.4
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole):
            item = form.itemAt(row, role)
            if item is not None and item.widget():
                item.widget().setVisible(visible)


def _cmap(color: str, name: str):
    from napari.utils.colormaps import Colormap

    rgb = np.array(figure._hex(color)) / 255.0
    return Colormap([[0, 0, 0, 1], [*rgb, 1]], name=f"lt {name} {color}")


def _surface_opacity(style: str) -> float:
    return 0.5 if style == "outline" else 0.85


def _surface_cmap(color: str):
    """One colour for a whole surface (the vertex values are all 1)."""
    from napari.utils.colormaps import Colormap

    rgb = list(np.array(figure._hex(color)) / 255.0)
    return Colormap([[*rgb, 1], [*rgb, 1]], name=f"lt surface {color}")


def open_viewer(path: Path | None, parent=None) -> CellViewer | None:
    """Ask for the analysis file if none is given, then open the cell viewer."""
    if path is None:
        found, _ = QFileDialog.getOpenFileName(parent, "Open a saved analysis", str(Path.home()),
                                               "Analysis (*.lta.json)")
        if not found:
            return None
        path = Path(found)
    try:
        viewer = CellViewer(path)
    except (OSError, ValueError, KeyError) as exc:
        QMessageBox.warning(parent, APP_NAME, f"The analysis cannot be opened:\n{exc}")
        return None
    viewer.show()
    return viewer
