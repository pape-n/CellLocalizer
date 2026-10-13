"""File locations that work from source and from the PyInstaller build."""

import os
import sys
from pathlib import Path


def user_config_dir() -> Path:
    """Per-user settings folder (panels, preferences). Created on demand. LT_CONFIG_DIR overrides it
    (tests, portable use)."""
    if os.environ.get("LT_CONFIG_DIR"):
        path = Path(os.environ["LT_CONFIG_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        return path
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / "LocalizationTool"          # name before 2.0.1, kept so settings survive the rename
    path.mkdir(parents=True, exist_ok=True)
    return path


def panels_file() -> Path:
    return user_config_dir() / "panels.json"


def resource(name: str) -> Path:
    """A file in localization_tool/resources, from source and from the PyInstaller build."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "localization_tool" / "resources" / name
