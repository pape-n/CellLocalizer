"""Performance test of the viewer on the current computer (phase 0 acceptance)."""

from __future__ import annotations

import platform
import re
import sys
import time

from localization_tool import APP_NAME, __version__


def _gl_info() -> list[str]:
    try:
        from napari.utils.info import sys_info

        text = re.sub(r"<[^>]+>", "\n", sys_info())
        keep = [ln.strip() for ln in text.splitlines()
                if any(k in ln for k in ("GL version", "MAX_TEXTURE_SIZE", "MAX_3D_TEXTURE", "screen ", "Qt:", "VisPy:"))]
    except Exception as exc:  # pragma: no cover
        keep = [f"system info not available ({exc})"]
    try:
        from vispy.gloo import gl

        renderer = str(gl.glGetParameter(gl.GL_RENDERER) or "").strip()
        if renderer:
            keep.insert(0, f"- GPU: {renderer} ({gl.glGetParameter(gl.GL_VENDOR)})")
    except Exception:
        pass
    return keep


def _verdict(fps: float) -> str:
    if fps >= 15:
        return "fluid"
    if fps >= 5:
        return "usable"
    return "slow"


def run_performance_test(main) -> str:
    from qtpy.QtWidgets import QApplication

    viewer = main.viewer
    stack = main.state.shown
    canvas = viewer.window._qt_viewer.canvas.native

    def draw():
        canvas.repaint()
        QApplication.processEvents()

    nd_before = viewer.dims.ndisplay
    step_before = viewer.dims.current_step
    angles_before = viewer.camera.angles

    import napari

    lines = [f"{APP_NAME} {__version__} performance test (napari {napari.__version__})",
             f"{time.strftime('%Y-%m-%d %H:%M')}, {platform.platform()}, Python {sys.version.split()[0]}",
             ""]
    lines += _gl_info()
    lines += ["", f"Image: {stack.display_name}", f"Shape (C, Z, Y, X): {stack.data.shape}, "
              f"{stack.nbytes_mb:.0f} MB, {stack.data.dtype}",
              f"Load time: {main.state.load_seconds:.2f} s", ""]

    # 2D: step through all planes
    viewer.dims.ndisplay = 2
    draw()
    nsteps = viewer.dims.nsteps[0]
    t = time.perf_counter()
    for i in range(nsteps):
        viewer.dims.set_current_step(0, i)
        draw()
    dt = time.perf_counter() - t
    fps2 = nsteps / dt if dt > 0 else float("inf")
    lines.append(f"2D plane stepping: {nsteps} planes in {dt:.2f} s = {fps2:.0f} planes/s ({_verdict(fps2)})")

    # 3D: first switch includes texture upload
    t = time.perf_counter()
    viewer.dims.ndisplay = 3
    draw()
    lines.append(f"Switch to 3D (texture upload): {time.perf_counter() - t:.2f} s")
    n = 36
    t = time.perf_counter()
    for i in range(n):
        viewer.camera.angles = (0.0, 30.0, i * 10.0)
        draw()
    dt = time.perf_counter() - t
    fps3 = n / dt if dt > 0 else float("inf")
    lines.append(f"3D rotation: {n} frames in {dt:.2f} s = {fps3:.1f} frames/s ({_verdict(fps3)})")

    # restore
    viewer.camera.angles = angles_before
    viewer.dims.ndisplay = nd_before
    viewer.dims.current_step = step_before
    draw()

    lines += ["", "Reading: fluid >= 15 frames/s, usable 5 to 15, slow < 5."]
    return "\n".join(lines)


class PerformanceDialog:
    """Help > Performance test: runs the test with the current image and saves the report."""

    def __init__(self, main):
        from pathlib import Path

        from qtpy.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout

        from localization_tool.gui import theme
        from localization_tool.gui.widgets import button, label

        self.main = main
        self.dialog = QDialog(main.viewer.window._qt_window)
        self.dialog.setWindowTitle("Performance test")
        self.dialog.setStyleSheet(theme.dock_stylesheet())
        lay = QVBoxLayout(self.dialog)
        lay.addWidget(label("Measures loading, plane stepping and 3D rendering on this computer with the current "
                            "image (or the example). Please save the report and send it back.", "hint"))
        row = QHBoxLayout()
        run_btn = button("Run test", primary=True)
        save_btn = button("Save report ...")
        row.addWidget(run_btn)
        row.addWidget(save_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumSize(560, 300)
        lay.addWidget(self.output)

        def run():
            if main.state.shown is None:
                main.load_example()
            run_btn.setEnabled(False)
            self.output.setPlainText("Running ...")
            try:
                self.output.setPlainText(run_performance_test(main))
            finally:
                run_btn.setEnabled(True)

        def save():
            text = self.output.toPlainText()
            if not text.strip():
                return
            path, _ = QFileDialog.getSaveFileName(self.dialog, "Save report",
                                                  str(Path.home() / "CellLocalizer_performance.txt"), "Text (*.txt)")
            if path:
                Path(path).write_text(text, encoding="utf-8")

        run_btn.clicked.connect(run)
        save_btn.clicked.connect(save)

    def exec_(self):
        return self.dialog.exec_()
