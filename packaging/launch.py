"""Entry point of the installed program (PyInstaller)."""

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from localization_tool.__main__ import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run(sys.argv))
