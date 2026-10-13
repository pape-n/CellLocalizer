"""The computational core must not import GUI libraries (ARCHITECTURE §3)."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "localization_tool" / "core"
FORBIDDEN = ("qtpy", "PySide6", "PyQt5", "PyQt6", "napari", "vispy", "matplotlib.pyplot", "magicgui")


def test_core_has_no_gui_imports():
    for f in CORE.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not n.startswith(FORBIDDEN), f"{f.name} imports {n}"
