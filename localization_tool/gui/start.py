"""Start window (D59), About (D60) and methods text dialog (D58).

The start window comes first, like in the Chemotaxis Tool: two large cards for a new analysis and for
opening a saved one (with the last analyses), below them the user guide, the methods text, the
program-wide settings and the synthetic example, at the bottom About and the version. The image
window (napari) opens only when an analysis starts; "Home" in the image window brings this back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QSizePolicy, QTextBrowser, QVBoxLayout,
    QWidget,
)

from localization_tool import APP_NAME, REPOSITORY, __version__
from localization_tool.core.methods_text import AUTHOR, CONTACT, LANGUAGES, methods_text, save_docx  # noqa: F401
from localization_tool.gui import theme
from localization_tool.gui.widgets import button, label, separator

SUBTITLE = ("Where does a protein sit in the cell? Signal shares and enrichment in nucleus, cytoplasm and actin, "
            "measured in 3D confocal stacks.")


class Card(QFrame):
    """Large clickable area with a title and one sentence, as on the Chemotaxis Tool start page."""

    clicked = Signal()

    def __init__(self, title: str, text: str):
        super().__init__()
        self.setObjectName("card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(150)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # the three cards share the width equally
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        t = label(title, "page")
        lay.addWidget(t)
        lay.addWidget(label(text, "hint"))
        self.extra = QVBoxLayout()
        lay.addLayout(self.extra)
        lay.addStretch(1)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(event)


class StartWindow(QWidget):
    """Actions are callables of MainWindow, so this window holds no analysis state."""

    def __init__(self, new_analysis: Callable[[], None], open_analysis: Callable[[Path | None], None],
                 open_guide: Callable[[], None], open_methods: Callable[[], None], open_settings: Callable[[], None],
                 open_example: Callable[[], None], open_about: Callable[[], None],
                 dropped: Callable[[list[Path]], None], recent: Callable[[], list[Path]],
                 continue_current: Callable[[], None] | None = None, confirm_close: Callable[[], bool] | None = None,
                 open_viewer: Callable[[], None] | None = None):
        super().__init__()
        self.setObjectName("start")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle(APP_NAME)
        theme.style(self)
        self.setAcceptDrops(True)
        self.setMinimumSize(700, 480)
        theme.fit_to_screen(self, 1040, 700)
        self._open = open_analysis
        self._recent = recent
        self._dropped = dropped
        self._confirm_close = confirm_close
        outer = QVBoxLayout(self)
        outer.setContentsMargins(64, 52, 64, 28)
        outer.setSpacing(12)
        outer.addWidget(label(APP_NAME, "apptitle", wrap=False))
        outer.addWidget(label(SUBTITLE, "hint"))
        outer.addSpacing(24)
        self.continue_btn = button("Continue the open analysis", primary=True, large=True)
        if continue_current is not None:
            self.continue_btn.clicked.connect(continue_current)
        self.continue_btn.setVisible(False)
        outer.addWidget(self.continue_btn, 0, Qt.AlignLeft)
        cards = QHBoxLayout()
        cards.setSpacing(20)
        self.card_new = Card("New analysis", "Choose the images of an experiment: a whole folder or single files. "
                                             "You can also drop .czi files on this window.")
        self.card_new.clicked.connect(new_analysis)
        self.card_open = Card("Open analysis", "Continue a saved analysis (analysis.lta.json): check more cells, "
                                               "change settings, save again.")
        self.card_open.clicked.connect(lambda: open_analysis(None))
        self.recent_box = QVBoxLayout()
        self.card_open.extra.addLayout(self.recent_box)
        self.card_view = Card("Cell viewer", "Look at the cells of a saved analysis in planes, sections "
                                                         "and 3D, and save pictures for figures.")
        if open_viewer is not None:
            self.card_view.clicked.connect(open_viewer)
        cards.addWidget(self.card_new, 1)            # equal widths, also with long recent entries
        cards.addWidget(self.card_open, 1)
        cards.addWidget(self.card_view, 1)
        outer.addLayout(cards)
        outer.addSpacing(16)
        row = QHBoxLayout()
        row.setSpacing(10)
        for text, fn, tip in (("User guide", open_guide, "How to use the program, step by step (English, Deutsch)"),
                              ("Methods text", open_methods, "Complete methods section for thesis or paper, and "
                               "how every value is computed (English, Deutsch)"),
                              ("Settings", open_settings, "Default settings for new analyses and the text size")):
            b = button(text, tip=tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)
        outer.addWidget(separator())
        foot = QHBoxLayout()
        about = button("About", flat=True)
        about.clicked.connect(open_about)
        example = button("Practice with a synthetic cell", flat=True,
                         tip="A simulated cell with known values, no real data")
        example.clicked.connect(open_example)
        foot.addWidget(about)
        foot.addWidget(example)
        foot.addStretch(1)
        foot.addWidget(label(f"Version {__version__}", "hint", wrap=False))
        outer.addLayout(foot)
        self.refresh()

    def refresh(self, has_analysis: bool = False, name: str = "") -> None:
        self.continue_btn.setVisible(has_analysis)
        self.continue_btn.setText(f"Continue: {name}" if name else "Continue the open analysis")
        while self.recent_box.count():
            item = self.recent_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        paths = self._recent()
        if paths:
            self.recent_box.addWidget(label("Recent", "hint", wrap=False))
        from datetime import datetime

        from qtpy.QtGui import QFontMetrics

        for p in paths:
            try:
                when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y")
            except OSError:
                when = ""
            text = f"{p.parent.name}   {when}"
            b = button(text, flat=True, tip=str(p))
            b.setText(QFontMetrics(b.font()).elidedText(text, Qt.ElideMiddle, 360))
            b.setStyleSheet("text-align: left;")
            b.clicked.connect(lambda _=False, q=p: self._open(q))
            self.recent_box.addWidget(b)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self._dropped(paths)

    def closeEvent(self, event):  # noqa: N802
        if self._confirm_close is not None and not self._confirm_close():
            event.ignore()
            return
        QApplication.quit()
        super().closeEvent(event)


def about_text() -> str:
    return (f"<p style='font-size:16pt'><b>{APP_NAME}</b> {__version__}</p>"
            "<p>Measures where a protein sits in a cell (nucleus, cytoplasm, actin, zones near the nuclear "
            "envelope, the cell surface and the coverslip) in 3D confocal stacks.</p>"
            f"<p>Developed by <b>{AUTHOR}</b>, Institute of Physiology II, University of Münster. Open source under "
            "the BSD 3-Clause License. Source code, releases and issues: "
            f"<a style='color:{theme.TOKENS["accent_text"]}' href='{REPOSITORY}'>{REPOSITORY}</a></p>"
            "<p><b>If you publish results obtained with this program, please cite the program version given in "
            "the methods text (Help &gt; Methods text)</b> and let the author know "
            f"(<a style='color:{theme.TOKENS["accent_text"]}' href='mailto:{CONTACT}'>{CONTACT}</a>).</p>"
            "<p>Built with Python, napari (BSD), PySide6 (LGPL), NumPy, SciPy, scikit-image (BSD), czifile, "
            "tifffile (BSD) and openpyxl (MIT).</p>")


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        theme.style(self)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        text = QLabel(about_text())
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        text.setOpenExternalLinks(True)
        lay.addWidget(text)
        ok = button("Close", primary=True)
        ok.clicked.connect(self.accept)
        lay.addWidget(ok, 0, Qt.AlignRight)


def methods_html(text: str) -> str:
    """The methods text for the dialog: the first line of a block is a section title (as in the Word
    file), parts in [square brackets] are marked as still to fill in."""
    import html
    import re

    warn = theme.TOKENS["warn"]
    out = []
    for block in text.strip().split("\n\n"):
        lines = [html.escape(x) for x in block.split("\n")]
        lines = [re.sub(r"(\[[^\]]*[A-Za-zÄÖÜäöü][^\]]*\])", rf"<span style='color:{warn}'>\1</span>", x)
                 for x in lines]
        if len(lines) >= 2 and not lines[0].startswith(("-", "1.")):
            out.append(f"<p style='font-weight:bold; font-size:{theme.pt('section')}pt; margin-top:12px'>"
                       f"{lines[0]}</p>")
            lines = lines[1:]
        out.append("<p>" + "<br>".join(lines) + "</p>")
    return "".join(out)


class MethodsDialog(QDialog):
    """The methods text of the open analysis (or of the default settings) in English or German, to copy
    or to save as Word or text file."""

    def __init__(self, parent, params, summary, source: str):
        super().__init__(parent)
        theme.style(self)
        self.setWindowTitle("Methods text")
        theme.fit_to_screen(self, 900, 760)
        self._params, self._summary = params, summary
        lay = QVBoxLayout(self)
        lay.addWidget(label(f"Built from {source}. Parts in [square brackets] are not known to the program, "
                            "please fill them in. The text also explains how every value is computed.", "hint"))
        top = QHBoxLayout()
        self.lang = QComboBox()
        for key, name in LANGUAGES.items():
            self.lang.addItem(name, key)
        self.lang.currentIndexChanged.connect(self._fill)
        top.addWidget(label("Language", "hint", wrap=False))
        top.addWidget(self.lang)
        top.addStretch(1)
        lay.addLayout(top)
        self.text = QTextBrowser()               # section titles bold, placeholders highlighted
        self.text.setOpenLinks(False)
        lay.addWidget(self.text, 1)
        row = QHBoxLayout()
        copy = button("Copy", primary=True)
        copy.clicked.connect(self._copy)
        save = button("Save as Word file ...")
        save.clicked.connect(self._save_docx)
        save_txt = button("Save as text file ...")
        save_txt.clicked.connect(self._save_txt)
        close = button("Close")
        close.clicked.connect(self.accept)
        self.info = label("", "hint")
        for b in (copy, save, save_txt):
            row.addWidget(b)
        row.addWidget(self.info, 1)
        row.addWidget(close)
        lay.addLayout(row)
        self._fill()

    def _copy(self) -> None:
        QApplication.clipboard().setText(self.current_text())
        self.info.setText("Copied. Paste it into your document and fill in the [brackets].")

    def current_text(self) -> str:
        return methods_text(self._params, self._summary, self.lang.currentData())

    def _fill(self, *_):
        self.text.setHtml(methods_html(self.current_text()))

    def _save_txt(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save methods text", "methods.txt", "Text (*.txt)")
        if path:
            Path(path).write_text(self.current_text(), encoding="utf-8")
            self.info.setText(f"Saved {Path(path).name}")

    def _save_docx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save methods text", "methods.docx", "Word (*.docx)")
        if path:
            save_docx(self.current_text(), Path(path))
            self.info.setText(f"Saved {Path(path).name}")
