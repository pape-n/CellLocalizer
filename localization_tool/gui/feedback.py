"""Help > Report a problem (D46): what to write, where to send it, and the system details to copy along."""

from __future__ import annotations

import platform
import sys

from qtpy.QtWidgets import QApplication, QDialog, QHBoxLayout, QPushButton, QTextBrowser, QVBoxLayout

from localization_tool import APP_NAME, REPOSITORY, __version__

TEXT = """
<h3>Report a problem or a wish</h3>
<p>Every remark helps: what was unclear, what was slow, where a mask looked wrong, what you missed.
Please open an issue at <a href="{repo}/issues">{repo}/issues</a> or write to {contact}.</p>
<p><b>Please write:</b></p>
<ol>
<li>What you wanted to do and what you did, step by step.</li>
<li>What you expected and what happened instead.</li>
<li>The image (file name) and the cell, and a screenshot if possible.</li>
<li>For a wrong mask: the control image of the cell (folder <i>Control images</i> of the analysis).</li>
<li>After a crash: the file <i>crash_previous.log</i> from the folder shown below.</li>
</ol>
<p>Rückmeldungen auf Deutsch sind ebenso willkommen.</p>
<p><b>Details of this computer</b> (copied with the button below):</p>
<pre>{details}</pre>
"""


def system_details() -> str:
    from localization_tool.gui.app import crash_log_path

    try:
        import napari
        nap = napari.__version__
    except Exception:
        nap = "?"
    screen = QApplication.primaryScreen()
    scale = f"{screen.devicePixelRatio():.2f}" if screen else "?"
    return "\n".join([
        f"{APP_NAME} {__version__}",
        f"System: {platform.platform()}",
        f"Python {sys.version.split()[0]}, napari {nap}, screen scale {scale}",
        f"Settings and crash log: {crash_log_path().parent}",
    ])


class FeedbackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from localization_tool.gui import theme

        theme.style(self)
        self.setWindowTitle("Report a problem")
        from localization_tool.gui import theme as _t

        _t.fit_to_screen(self, 640, 560)
        self.details = system_details()
        lay = QVBoxLayout(self)
        text = QTextBrowser()
        from localization_tool.core.methods_text import CONTACT

        text.setOpenExternalLinks(True)
        text.setHtml(TEXT.format(details=self.details, repo=REPOSITORY, contact=CONTACT))
        lay.addWidget(text, 1)
        row = QHBoxLayout()
        copy = QPushButton("Copy details")
        close = QPushButton("Close")
        row.addWidget(copy)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.details))
        close.clicked.connect(self.accept)
