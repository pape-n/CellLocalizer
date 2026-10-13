"""Small reusable widgets for the workflow panel."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QTableWidget, QWidget,
)

from localization_tool.core.channels import DISPLAY_COLORS, ROLES, ChannelAssignment


def label(text: str, role: str | None = None, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    if role:
        lab.setProperty("role", role)
    return lab


def button(text: str, primary: bool = False, tip: str = "", large: bool = False, flat: bool = False) -> QPushButton:
    """Push button in the program style: primary (filled accent), secondary (frame) or flat (link).
    large: the step buttons (Back, Next, Accept, ...), 40 px high (D61)."""
    b = QPushButton(text)
    if primary:
        b.setProperty("role", "primary")
    elif flat:
        b.setProperty("role", "flat")
    if large:
        b.setProperty("size", "large")
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


def separator() -> QFrame:
    f = QFrame()
    f.setObjectName("separator")
    f.setFixedHeight(1)
    return f


def restyle(widget) -> None:
    """Re-apply the style sheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class ElidedLabel(QLabel):
    """One line that shortens itself with "..." instead of widening the panel; the full text is the tooltip."""

    def __init__(self, text: str = "", role: str | None = None):
        super().__init__()
        if role:
            self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self._full = ""
        self.set_full_text(text)

    def set_full_text(self, text: str, tip: str | None = None) -> None:
        self._full = text
        self.setToolTip(tip or text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, max(self.width(), 40)))


def _short_fluor(name: str) -> str:
    """'Alexa Fluor 647' -> 'AF647'; other names stay as they are."""
    n = name.strip()
    if n.lower().startswith("alexa fluor"):
        return "AF" + n[len("alexa fluor"):].strip()
    return n


class ChannelTable(QTableWidget):
    """Role, name and colour per channel. Emits changed() after every edit."""

    changed = Signal()
    COLS = ("Channel", "Role", "Name", "Colour")

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def populate(self, channels, assignments: list[ChannelAssignment]) -> None:
        self.blockSignals(True)
        self.setRowCount(len(channels))
        for r, (ch, a) in enumerate(zip(channels, assignments)):
            desc = label(f"{ch.index + 1}  {_short_fluor(ch.short)}", wrap=False)
            desc.setToolTip(ch.describe())
            desc.setContentsMargins(6, 2, 6, 2)
            self.setCellWidget(r, 0, desc)
            role = QComboBox()
            role.addItems(ROLES)
            role.setCurrentText(a.role)
            role.currentTextChanged.connect(self._emit)
            self.setCellWidget(r, 1, role)
            name = QLineEdit(a.label)
            name.setMinimumWidth(80)
            name.editingFinished.connect(self._emit)
            self.setCellWidget(r, 2, name)
            color = QComboBox()
            for cname, hexval in DISPLAY_COLORS.items():
                color.addItem(cname, hexval)
            if a.color not in DISPLAY_COLORS.values():
                color.addItem(f"Original ({a.color})", a.color)
            color.setCurrentIndex(max(0, color.findData(a.color)))
            color.currentIndexChanged.connect(self._emit)
            self.setCellWidget(r, 3, color)
        self.resizeRowsToContents()
        self.setFixedHeight(self.horizontalHeader().height()
                            + sum(self.rowHeight(r) for r in range(self.rowCount())) + 4)
        self.blockSignals(False)

    def assignments(self) -> list[ChannelAssignment]:
        out = []
        for r in range(self.rowCount()):
            role = self.cellWidget(r, 1).currentText()
            name = self.cellWidget(r, 2).text().strip()
            color = self.cellWidget(r, 3).currentData()
            out.append(ChannelAssignment(role=role, label=name or f"C{r + 1}", color=color))
        return out

    def _emit(self, *_):
        self.changed.emit()


class StepBar(QWidget):
    """The five steps as a timeline, as in the Chemotaxis Tool (D62, D68): "1 Open — 2 Channels — ...".
    The steps look like text, not like buttons, but can be clicked: the current step in the accent colour
    and underlined, finished steps in normal text, later steps grey and only clickable when every step
    before them is complete. A locked step says why in its tooltip. Emits stepClicked(index)."""

    STEPS = ("Open", "Channels", "Cell", "Compartments", "Results")
    SHORT = ("Open", "Chan.", "Cell", "Comp.", "Results")      # narrow panels and large text
    stepClicked = Signal(int)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.buttons: list[QPushButton] = []
        self.links: list[QFrame] = []
        for i, name in enumerate(self.STEPS):
            if i:
                line = QFrame()
                line.setObjectName("stepline")
                line.setFixedHeight(1)
                line.setMinimumWidth(6)
                line.setMaximumWidth(18)
                line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                lay.addWidget(line, 0, Qt.AlignVCenter)
                self.links.append(line)
            b = QPushButton(f"{i + 1}  {name}")
            b.setProperty("step", "open")
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.TabFocus)       # a click gives no focus frame, the Tab key does
            b.setAutoDefault(False)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, k=i: self.stepClicked.emit(k))
            lay.addWidget(b)
            self.buttons.append(b)
        lay.addStretch(1)
        self._short = False

    def _width_needed(self, names) -> int:
        from qtpy.QtGui import QFontMetrics

        font = self.buttons[0].font()
        font.setBold(True)                      # the current step is bold, the widest case
        fm = QFontMetrics(font)
        text = sum(fm.horizontalAdvance(f"{i + 1}  {n}") + 18 for i, n in enumerate(names))
        return text + 10 * len(self.links) + 2 * (len(self.buttons) + len(self.links))

    def _labels_fit(self) -> bool:
        return self._width_needed(self.STEPS) <= self.width()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        short = not self._labels_fit()
        if short != self._short:
            self._short = short
            names = self.SHORT if short else self.STEPS
            for i, (b, n) in enumerate(zip(self.buttons, names)):
                b.setText(f"{i + 1}  {n}")

    def set_state(self, current: int, reachable: list[bool], reasons: list[str]) -> None:
        """reachable[i]: step i can be opened now. reasons[i]: why not (tooltip of a locked step)."""
        for i, b in enumerate(self.buttons):
            if i == current:
                state, tip = "current", f"Step {i + 1}: {self.STEPS[i]} (this step)"
            elif reachable[i]:
                state, tip = ("done" if i < current else "open"), f"Go to step {i + 1}: {self.STEPS[i]}"
            else:
                state, tip = "locked", reasons[i] or "Finish the steps before"
            b.setProperty("step", state)
            b.setEnabled(state != "locked")
            b.setToolTip(tip)
            restyle(b)
        for i, line in enumerate(self.links):     # the line up to a finished step is drawn stronger
            line.setProperty("done", i < current)
            restyle(line)
        # a disabled button shows no tooltip on most platforms: keep the reason reachable on the bar
        self.setToolTip("\n".join(f"{i + 1} {self.STEPS[i]}: {r}" for i, r in enumerate(reasons)
                                   if r and not reachable[i] and i != current))
