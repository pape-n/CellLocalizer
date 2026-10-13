"""Settings of an analysis (CLAUDE.md: every default can be changed and is saved in the analysis file).

All numbers of AnalysisParams on a few tabs, plus the vocabulary for conditions from file names.
The dialog only edits a copy. MainWindow decides what happens to cells computed with the old values.
"""

from __future__ import annotations

from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout,
    QHeaderView, QLineEdit, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from localization_tool.core.params import (
    ACTIN_DEFINITIONS, MASK_KEYS, THRESHOLD_METHODS, Z_REFERENCES, AnalysisParams,
)
from localization_tool.gui.widgets import button, label

MASK_NAMES = {"cell": "Cell edge", "nucleus": "Nucleus", "actin": "Actin", "actin_rich": "Actin-rich",
              "target": "Target mask", "comparison": "Comparison mask"}
TARGET_SIGNAL_NAMES = {"mask": "Only inside the target mask (correcting the mask changes the numbers)",
                       "all": "All target signal in the cell (the mask is only for checking)"}
POLARITY_NAMES = {"actin_rich": "Actin-rich mask (follows its factor and painted masks)",
                  "actin_signal": "Whole actin signal (as before 2.1)"}
METHOD_NAMES = {"otsu": "Otsu", "li": "Li", "triangle": "Triangle", "manual": "Manual value"}
NUCLEUS_METHOD_NAMES = {"halfmax": "Half maximum", **METHOD_NAMES}
Z_NAMES = {"auto": "From the file (recommended)", "first_plane": "First plane", "last_plane": "Last plane"}
ACTIN_NAMES = {"threshold": "Threshold in the actin channel", "shell": "Shell under the cell surface"}


def _spin(lo: float, hi: float, step: float, decimals: int, value: float, suffix: str = "", tip: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    if tip:
        sb.setToolTip(tip)
    sb.setMinimumWidth(_spin_width(lo, hi, decimals, suffix))   # no clipped "0.300 µm", also on Windows
    return sb


def _spin_width(lo: float, hi: float, decimals: int, suffix: str) -> int:
    """Width for the longest value with its unit in the program font, plus the arrow buttons."""
    from qtpy.QtGui import QFont, QFontMetrics

    from localization_tool.gui import theme
    from localization_tool.prefs import font_scale

    font = QFont(theme.FONT_FAMILY)
    font.setPointSizeF(theme.pt("body", font_scale()))
    longest = max((f"{v:.{decimals}f}{suffix}" for v in (lo, hi)), key=len)
    return max(80, QFontMetrics(font).horizontalAdvance(longest) + 48)


def _form_row(text: str, widget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(label(text, "hint", wrap=False))
    row.addWidget(widget)
    row.addStretch(1)
    return row


def _combo(names: dict, current: str) -> QComboBox:
    cb = QComboBox()
    for key, text in names.items():
        cb.addItem(text, key)
    cb.setCurrentIndex(max(0, cb.findData(current)))
    return cb


class SettingsDialog(QDialog):
    def __init__(self, parent, params: AnalysisParams, channel_labels: list[str], vocabulary: list[str],
                 estimate_crosstalk: Callable[[str, str], dict] | None = None, blind: bool = True,
                 program_defaults: bool = False, channels: list[tuple[str, str, str]] | None = None,
                 dark_levels: dict[str, float] | None = None):
        """channels: (label, fluorophore, role) of the open image in file order, for the bead measurement.
        dark_levels: camera dark level per channel label of the open image (D80), for 'Estimate dark level'."""
        super().__init__(parent)
        from localization_tool.gui import theme

        theme.style(self)
        self.setWindowTitle("Default settings for new analyses" if program_defaults else "Settings of this analysis")
        theme.fit_to_screen(self, 860, 620)        # every tab name visible at 100 %
        self._labels = list(channel_labels)
        self._channels = list(channels or [])
        self._dark = dict(dark_levels or {})
        self._program_defaults = program_defaults
        self._estimate = estimate_crosstalk
        self._blind = blind
        root = QVBoxLayout(self)
        root.addWidget(label("Every new analysis starts with these settings. An analysis that is already saved "
                             "keeps its own settings." if program_defaults else
                             "These settings are saved with the analysis. Cells already computed keep their "
                             "numbers until you apply the settings to all cells.", "hint"))
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build(params, vocabulary)
        row = QHBoxLayout()
        self.defaults_btn = button("Restore program defaults")
        self.defaults_btn.clicked.connect(lambda: self._build(AnalysisParams(), vocabulary, keep_tab=True))
        row.addWidget(self.defaults_btn)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        from qtpy.QtGui import QIcon

        for b in buttons.buttons():
            b.setIcon(QIcon())                   # no platform icons, the program style instead
        buttons.button(QDialogButtonBox.Ok).setProperty("role", "primary")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        root.addLayout(row)
        self.error = label("", "warn")
        root.addWidget(self.error)
        self.result_params: AnalysisParams | None = None
        self.result_vocabulary: list[str] = list(vocabulary)
        self.result_blind = blind

    # ------------------------------------------------------------------ build
    def _build(self, p: AnalysisParams, vocabulary: list[str], keep_tab: bool = False) -> None:
        current = self.tabs.currentIndex()
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        self.tabs.addTab(self._masks_tab(p), "Masks")
        self.tabs.addTab(self._zones_tab(p), "Zones and height")
        self.tabs.addTab(self._extra_tab(p), "Extras and review")
        self.tabs.addTab(self._channels_tab(p), "Channels")
        self.tabs.addTab(self._offsets_tab(p), "Channel offset")
        self.tabs.addTab(self._names_tab(vocabulary), "File names")
        if keep_tab:
            self.tabs.setCurrentIndex(max(0, current))

    def _masks_tab(self, p: AnalysisParams) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Each mask: smoothing, automatic threshold method and a factor on that threshold "
                            "(below 1 larger, above 1 smaller). The factors in steps 3 and 4 change single cells; "
                            "the factors here are the starting values for every cell.", "hint"))
        grid = QGridLayout()
        for c, head in enumerate(("", "Method", "Manual value", "Factor", "Smoothing σ", "Min. volume")):
            grid.addWidget(label(head, "hint", wrap=False), 0, c)
        self.mask_w = {}
        for r, key in enumerate(MASK_KEYS, start=1):
            mp = getattr(p, key)
            method = _combo(NUCLEUS_METHOD_NAMES if key in ("nucleus", "cell") else METHOD_NAMES, mp.method)
            manual = _spin(0, 65535, 10, 1, mp.manual_threshold or 0.0, tip="Only used with 'Manual value'")
            factor = _spin(0.2, 3.0, 0.05, 2, mp.factor)
            sigma = _spin(0.0, 1.0, 0.01, 3, mp.sigma_um, " µm")
            vol = _spin(0.0, 100.0, 0.1, 2, mp.min_volume_um3, " µm³", "Pieces smaller than this are dropped")
            for c, wdg in enumerate((label(MASK_NAMES[key], wrap=False), method, manual, factor, sigma, vol)):
                grid.addWidget(wdg, r, c)
            self.mask_w[key] = (method, manual, factor, sigma, vol)
        lay.addLayout(grid)
        form = QFormLayout()
        self.max_hole = _spin(0.0, 5.0, 0.05, 2, p.nucleus_max_hole_um2, " µm²",
                              "Holes in the nucleus up to this area per plane count as nucleus")
        form.addRow("Fill holes in the nucleus up to", self.max_hole)
        self.dim_hole = _spin(0.0, 0.95, 0.05, 2, p.nucleus_dim_hole_fraction, "",
                              "Larger holes count as nucleus when their DAPI reaches this fraction of the "
                              "nucleus threshold (dim chromatin). 0 switches this off.")
        form.addRow("Fill dim nucleus holes from (× threshold)", self.dim_hole)
        self.actin_def = _combo(ACTIN_NAMES, p.actin_definition)
        form.addRow("Actin compartment from", self.actin_def)
        self.actin_planes = QCheckBox("Also per plane (finds the dimmer cortex above a bright layer at the coverslip)")
        self.actin_planes.setChecked(p.actin_per_plane)
        self.actin_planes.setToolTip("Actin is everything above the threshold of the whole cell, or above the "
                                     "half-maximum threshold of its own plane (between cytoplasm and actin of "
                                     "that plane). Off: one threshold for the whole stack, as before 1.2.1.")
        form.addRow("", self.actin_planes)
        self.target_signal = _combo(TARGET_SIGNAL_NAMES, p.target_signal)
        self.target_signal.setToolTip("D89: inside the mask, diffuse background and the faint halo of bright spots "
                                      "do not count, and the values follow the target mask factor. All signal: "
                                      "independent of a threshold, faint even staining counts as well.")
        form.addRow("Target signal counted", self.target_signal)
        lay.addLayout(form)
        lay.addStretch(1)
        return w

    def _zones_tab(self, p: AnalysisParams) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.addWidget(label("Zones are extra views next to the compartments and overlap them: the nuclear "
                              "envelope zone lies on both sides of the nucleus edge, the cortex shell under the cell "
                              "surface, the basal layer above the coverslip. The axial scale factor corrects the z "
                              "step for the mounting medium (1 = as in the file). The two notes only flag cells, "
                              "they never change numbers.", "hint"))
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)
        self.envelope = _spin(0.05, 2.0, 0.05, 2, p.envelope_um, " µm", "Half width of the nuclear envelope zone")
        self.cortex = _spin(0.05, 3.0, 0.05, 2, p.cortex_um, " µm", "Thickness of the cortex shell")
        self.basal = _spin(0.05, 3.0, 0.05, 2, p.basal_um, " µm", "Thickness of the basal layer")
        self.bin = _spin(0.05, 2.0, 0.05, 2, p.profile_bin_um, " µm", "Bin width of the profiles")
        self.zref = _combo(Z_NAMES, p.z_reference)
        self.axial = _spin(0.5, 1.5, 0.01, 3, p.axial_scale, "",
                           "Correction of the z step for the refractive index of the mounting medium (D23)")
        self.trunc = _spin(0.0, 1.0, 0.05, 2, p.truncation_fraction, "",
                           "Note 'stack cut off' when the cell in the first or last plane is larger than this "
                           "share of its widest plane")
        self.bottom = _spin(0.0, 1.0, 0.05, 2, p.bottom_fraction, "",
                            "The methods text counts a cell as 'stack starts inside the cell' when the plane at "
                            "the coverslip reaches this share of the brightest plane (actin, target). Shown in "
                            "Results.xlsx as column bottom_plane_ratio, not as a note.")
        self.sat = _spin(0.0, 0.1, 0.0005, 4, p.saturation_fraction, "",
                         "Note 'saturated' when more than this share of the cell voxels is at the maximum")
        self.lateral = QCheckBox("Nuclear envelope zones only at the side of the nucleus")
        self.lateral.setChecked(p.envelope_lateral_only)
        self.lateral.setToolTip("Leaves out the top and bottom of the nucleus, where the axial blur (about three "
                                "times the lateral) carries light across the nucleus edge (D86)")
        form.addRow("Nuclear envelope zone ±", self.envelope)
        form.addRow("", self.lateral)
        form.addRow("Cortex shell", self.cortex)
        form.addRow("Basal layer", self.basal)
        form.addRow("Profile bins", self.bin)
        form.addRow("Coverslip side", self.zref)
        form.addRow("Axial scale factor", self.axial)
        form.addRow("Note 'cut off' above", self.trunc)
        form.addRow("Lowest plane counts as bright above", self.bottom)
        form.addRow("Note 'saturated' above", self.sat)
        return w

    def _extra_tab(self, p: AnalysisParams) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.pol_on = QCheckBox("Polarity: front direction from the actin channel, share in the front half")
        self.pol_on.setChecked(p.polarity)
        self.col_on = QCheckBox("Colocalization: Pearson and Manders (Costes threshold) with actin and other targets")
        self.col_on.setChecked(p.colocalization)
        self.sens_on = QCheckBox("Sensitivity span: shares again with cell, nucleus and target edge at 0.8 and 1.2 x threshold")
        self.sens_on.setChecked(p.sensitivity)
        self.sens_on.setToolTip("Shows for every cell how much its values depend on the exact edges. Takes a few "
                                "seconds more per cell.")
        lay.addWidget(self.pol_on)
        self.pol_from = _combo(POLARITY_NAMES, p.polarity_from)
        lay.addLayout(_form_row("Front direction towards", self.pol_from))
        self.pol_clear = _spin(0.0, 2.0, 0.01, 2, p.polarity_clear_index, "",
                               "Cells with a polarity index from this value on count as polar (column "
                               "polarity_clear, bold arrow). Below it the arrow is short and thin.")
        lay.addLayout(_form_row("Clear front from polarity index", self.pol_clear))
        lay.addWidget(self.col_on)
        lay.addWidget(self.sens_on)
        lay.addWidget(label("Review", "title"))
        self.blind_on = QCheckBox("Blind review: hide file names, folders and conditions while checking cells")
        self.blind_on.setChecked(self._blind)
        self.blind_on.setToolTip("Images get a random code. 'Show name' reveals one image and is noted in the "
                                 "analysis file. Results and Excel always show all names.")
        lay.addWidget(self.blind_on)
        lay.addWidget(label("Text size (whole program)", "title"))
        from localization_tool.prefs import FONT_SCALES, font_scale

        self.text_size = QComboBox()
        for v in FONT_SCALES:
            self.text_size.addItem(f"{int(round(v * 100))} %", v)
        self.text_size.setCurrentIndex(max(0, self.text_size.findData(font_scale())))
        lay.addLayout(_form_row("Text and buttons", self.text_size))
        lay.addWidget(label("For the whole program on this computer. Takes effect after a restart.", "hint"))
        lay.addStretch(1)
        return w

    def _offsets_tab(self, p: AnalysisParams) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Chromatic offset between channels. Each channel is moved back by its offset in "
                            "whole voxels before masks and measurements; voxel values are not interpolated. Measure "
                            "it once per microscope and objective: record multicolour beads (e.g. 0.1 µm TetraSpeck) "
                            "with the same settings as the cells, then 'Measure from a bead image'. Without an "
                            "offset, heights and thin zones (basal layer, cortex) can be shifted by the optics.",
                            "hint"))
        labels = list(dict.fromkeys(self._labels + [o["channel"] for o in p.channel_offsets]))
        self._off_fluor = {o["channel"]: o["fluor"] for o in p.channel_offsets if o.get("fluor")}
        self.off = QTableWidget(0, 4)
        self.off.setHorizontalHeaderLabels(["Channel", "z offset", "y offset", "x offset"])
        self.off.verticalHeader().setVisible(False)
        self.off.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.off.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)
        cur = {o["channel"]: o for o in p.channel_offsets}
        for lab in labels:
            r = self._offset_row(lab)
            o = cur.get(lab, {})
            for c, key in enumerate(("dz_um", "dy_um", "dx_um"), start=1):
                self.off.cellWidget(r, c).setValue(float(o.get(key, 0.0)))
        self.off.setMinimumHeight(44 + 36 * max(1, min(len(labels), 4)))
        lay.addWidget(self.off)
        row = QHBoxLayout()
        self.ref_channel = QComboBox()
        self.ref_channel.setToolTip("The other channels are aligned to this one. Usually the nucleus or actin "
                                    "channel; empty: the first channel of the bead image.")
        self.ref_channel.addItems(labels)
        meas = button("Measure from a bead image ...", primary=True)
        meas.clicked.connect(self._measure_beads)
        zero = button("Set all to 0")
        zero.clicked.connect(lambda: [self.off.cellWidget(r, c).setValue(0.0) for r in range(self.off.rowCount())
                                      for c in (1, 2, 3)])
        row.addWidget(label("Reference channel", "hint", wrap=False))
        row.addWidget(self.ref_channel)
        row.addWidget(meas)
        row.addWidget(zero)
        row.addStretch(1)
        lay.addLayout(row)
        self.off_info = label("Without an open image the offsets are stored per fluorophore and used for every "
                              "channel with that fluorophore." if not self._channels else "", "hint")
        lay.addWidget(self.off_info)
        lay.addStretch(1)
        return w

    def _offset_row(self, lab: str) -> int:
        """Row of a channel in the offset table, added if missing."""
        for r in range(self.off.rowCount()):
            if self.off.item(r, 0).text() == lab:
                return r
        r = self.off.rowCount()
        self.off.insertRow(r)
        it = QTableWidgetItem(lab)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        if self._off_fluor.get(lab):
            it.setToolTip(self._off_fluor[lab])
        self.off.setItem(r, 0, it)
        for c in (1, 2, 3):
            sb = _spin(-5.0 if c == 1 else -1.0, 5.0 if c == 1 else 1.0, 0.005, 3, 0.0, " µm")
            self.off.setCellWidget(r, c, sb)
            self.off.setColumnWidth(c, max(self.off.columnWidth(c), sb.minimumWidth() + 10))
        return r

    def bead_labels(self, bead_fluors: list[str]) -> tuple[list[str] | None, str]:
        from localization_tool.core.offsets import match_channels

        return match_channels(bead_fluors, self._channels)

    def _measure_beads(self) -> None:
        from qtpy.QtWidgets import QApplication, QFileDialog

        from localization_tool.core.offsets import estimate_offsets
        from localization_tool.core.reader import read_image

        path, _ = QFileDialog.getOpenFileName(self, "Bead image", "", "Images (*.czi *.nd2 *.tif *.tiff)")
        if not path:
            return
        self.off_info.setText("Measuring ...")
        QApplication.processEvents()
        try:
            stack = read_image(path)
        except Exception as exc:              # shown, the dialog stays open
            self.off_info.setText(str(exc))
            return
        fluors = [c.fluor or c.name for c in stack.channels]
        names, how = self.bead_labels(fluors)
        if names is None:
            self.off_info.setText(how)
            return
        ignore = {lab for lab, _, role in self._channels if role == "Ignore"}
        want = self.ref_channel.currentText()
        ref = names.index(want) if want in names else next((k for k, n in enumerate(names) if n not in ignore), 0)
        res = estimate_offsets(stack.data, stack.voxel_um, ref, names)
        lines = [f"Channels matched {how}. Reference {res['reference']}: {res['reference_beads']} beads found."]
        for k, lab in enumerate(names):
            if lab in ignore:
                continue
            self._off_fluor[lab] = fluors[k]
            r = self._offset_row(lab)
            if self.ref_channel.findText(lab) < 0:
                self.ref_channel.addItem(lab)
            if k == ref:
                for c in (1, 2, 3):
                    self.off.cellWidget(r, c).setValue(0.0)
                continue
            o = res["channels"].get(lab, {})
            if "dz_um" not in o:
                lines.append(f"{lab}: {o.get('error', 'no result')}, offset not changed.")
                continue
            if abs(o["dz_um"]) > 5 or abs(o["dy_um"]) > 1 or abs(o["dx_um"]) > 1:
                lines.append(f"{lab}: implausible offset (z {o['dz_um']:.2f}, y {o['dy_um']:.2f}, x {o['dx_um']:.2f} µm), "
                             "not used. Are these beads seen in every channel?")
                continue
            for c, key in enumerate(("dz_um", "dy_um", "dx_um"), start=1):
                self.off.cellWidget(r, c).setValue(o[key])
            sz, sy, sx = o["spread_um"]
            lines.append(f"{lab}: z {o['dz_um'] * 1000:+.0f}, y {o['dy_um'] * 1000:+.0f}, x {o['dx_um'] * 1000:+.0f} nm "
                         f"from {o['beads']} beads (spread {sz * 1000:.0f} / {sy * 1000:.0f} / {sx * 1000:.0f} nm)")
        self.off_info.setText("\n".join(lines))

    def _fill_dark_levels(self) -> None:
        for r in range(self.bg.rowCount()):
            lab = self.bg.item(r, 0).text()
            if lab in self._dark:
                self.bg.cellWidget(r, 1).setValue(round(self._dark[lab]))

    def _channels_tab(self, p: AnalysisParams) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Background per channel, subtracted from every voxel before measuring. "
                            "0 fits Airyscan-processed images, which are clipped at 0. Camera images (spinning disk, "
                            "widefield) carry an offset: use 'Estimate dark level' below."
                            + (" The channels are listed once an image is open; here only values saved "
                               "earlier appear." if not self._labels else ""), "hint"))
        labels = list(dict.fromkeys(self._labels + list(p.background)))
        self.bg = QTableWidget(len(labels), 2)
        self.bg.setHorizontalHeaderLabels(["Channel", "Background"])
        self.bg.verticalHeader().setVisible(False)
        self.bg.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for r, lab in enumerate(labels):
            it = QTableWidgetItem(lab)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.bg.setItem(r, 0, it)
            self.bg.setCellWidget(r, 1, _spin(0, 65535, 1, 1, p.background_for(lab)))
        self.bg.setMaximumHeight(40 + 32 * max(1, len(labels)))
        lay.addWidget(self.bg)
        dark_row = QHBoxLayout()
        dark = button("Estimate dark level from the open image",
                      tip="Fills in the 0.1st percentile of each channel of the open image: the camera offset of "
                          "images that are not clipped at 0 (spinning disk, widefield, camera-based confocal). "
                          "Airyscan-processed images are clipped at 0 and keep background 0.")
        dark.setEnabled(bool(self._dark))
        dark.clicked.connect(self._fill_dark_levels)
        dark_row.addWidget(dark)
        dark_row.addStretch(1)
        lay.addLayout(dark_row)
        lay.addWidget(label("Crosstalk correction (off unless you add a row): target = target − k × source. "
                            "Measure k on a control stained only for the source dye: set the background above "
                            "first, open the control, find the cell in step 3, then press 'Estimate k'.", "hint"))
        self.xt = QTableWidget(0, 3)
        self.xt.setHorizontalHeaderLabels(["Correct channel", "for light from", "k"])
        self.xt.verticalHeader().setVisible(False)
        self.xt.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.xt)
        for c in p.crosstalk:
            self._add_xt(c["into"], c["from"], c["k"])
        row = QHBoxLayout()
        add = button("Add row")
        rem = button("Remove row")
        est = button("Estimate k from the open image")
        est.setEnabled(self._estimate is not None)
        add.clicked.connect(lambda: self._add_xt())
        rem.clicked.connect(lambda: self.xt.removeRow(self.xt.currentRow()) if self.xt.currentRow() >= 0 else None)
        est.clicked.connect(self._estimate_k)
        for b in (add, rem, est):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.xt_info = label("", "hint")
        lay.addWidget(self.xt_info)
        lay.addStretch(1)
        return w

    def _add_xt(self, into: str = "", src: str = "", k: float = 0.0) -> None:
        r = self.xt.rowCount()
        self.xt.insertRow(r)
        names = list(dict.fromkeys(self._labels + [x for x in (into, src) if x]))
        for c, cur in ((0, into), (1, src)):
            cb = QComboBox()
            cb.addItems(names)
            if cur:
                cb.setCurrentText(cur)
            elif c == 1 and len(names) > 1:
                cb.setCurrentIndex(1)
            self.xt.setCellWidget(r, c, cb)
        self.xt.setCellWidget(r, 2, _spin(0.0, 0.99, 0.005, 4, k))
        self.xt.setCurrentCell(r, 0)

    def _estimate_k(self) -> None:
        r = self.xt.currentRow()
        if r < 0 or self._estimate is None:
            self.xt_info.setText("Select a row first.")
            return
        into = self.xt.cellWidget(r, 0).currentText()
        src = self.xt.cellWidget(r, 1).currentText()
        try:
            res = self._estimate(into, src)
        except Exception as exc:           # shown, not raised: the dialog stays open
            self.xt_info.setText(str(exc))
            return
        if res.get("k") is None or res["k"] != res["k"]:
            self.xt_info.setText("k could not be estimated (no clear signal in the source channel).")
            return
        self.xt.cellWidget(r, 2).setValue(res["k"])
        self.xt_info.setText(f"k = {res['k']:.4f} from {res['voxels']} voxels (correlation {res['r']:.2f}). "
                             "Only valid if the open image is stained for the source dye alone.")

    def _names_tab(self, vocabulary: list[str]) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Words that are looked for in file and folder names. Found words fill in the condition "
                            "of new images, e.g. 'WT treated'. Separate the words by commas. The experiment is "
                            "taken from a date in the folder name.", "hint"))
        self.vocab = QLineEdit(", ".join(vocabulary))
        lay.addWidget(self.vocab)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ read back
    def values(self) -> tuple[AnalysisParams, list[str]]:
        d = {}
        for key, (method, manual, factor, sigma, vol) in self.mask_w.items():
            m = method.currentData()
            d[key] = {"method": m, "manual_threshold": manual.value() if m == "manual" else None,
                      "factor": factor.value(), "sigma_um": sigma.value(), "min_volume_um3": vol.value()}
        d.update({
            "nucleus_max_hole_um2": self.max_hole.value(),
            "nucleus_dim_hole_fraction": self.dim_hole.value(), "actin_definition": self.actin_def.currentData(),
            "envelope_um": self.envelope.value(), "cortex_um": self.cortex.value(), "basal_um": self.basal.value(),
            "profile_bin_um": self.bin.value(), "z_reference": self.zref.currentData(),
            "axial_scale": self.axial.value(), "truncation_fraction": self.trunc.value(),
            "bottom_fraction": self.bottom.value(),
            "saturation_fraction": self.sat.value(),
            "polarity": self.pol_on.isChecked(), "colocalization": self.col_on.isChecked(),
            "sensitivity": self.sens_on.isChecked(), "actin_per_plane": self.actin_planes.isChecked(),
            "target_signal": self.target_signal.currentData(), "polarity_from": self.pol_from.currentData(),
            "polarity_clear_index": self.pol_clear.value(), "envelope_lateral_only": self.lateral.isChecked(),
        })
        offs = []
        for r in range(self.off.rowCount()):
            vals = {key: self.off.cellWidget(r, c).value() for c, key in enumerate(("dz_um", "dy_um", "dx_um"), start=1)}
            if any(abs(v) > 0 for v in vals.values()):
                lab = self.off.item(r, 0).text()
                entry = {"channel": lab, **vals}
                if self._off_fluor.get(lab):
                    entry["fluor"] = self._off_fluor[lab]
                offs.append(entry)
        d["channel_offsets"] = offs
        bg = {}
        for r in range(self.bg.rowCount()):
            v = self.bg.cellWidget(r, 1).value()
            if v > 0:
                bg[self.bg.item(r, 0).text()] = v
        d["background"] = bg
        xt = []
        for r in range(self.xt.rowCount()):
            k = self.xt.cellWidget(r, 2).value()
            if k > 0:
                xt.append({"into": self.xt.cellWidget(r, 0).currentText(),
                           "from": self.xt.cellWidget(r, 1).currentText(), "k": k})
        d["crosstalk"] = xt
        vocab = [w.strip() for w in self.vocab.text().split(",") if w.strip()]
        return AnalysisParams.from_dict(d), vocab

    def _accept(self) -> None:
        try:
            self.result_params, self.result_vocabulary = self.values()
        except (ValueError, TypeError) as exc:
            self.error.setText(f"Please check the values: {exc}")
            return
        self.result_blind = self.blind_on.isChecked()
        from localization_tool.prefs import set_font_scale

        set_font_scale(self.text_size.currentData())
        self.accept()
