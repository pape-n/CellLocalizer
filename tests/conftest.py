import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_API", "pyside6")
# tests never read or write the developer's own preferences (saved defaults, recent analyses)
if "LT_CONFIG_DIR" not in os.environ:
    import tempfile

    os.environ["LT_CONFIG_DIR"] = tempfile.mkdtemp(prefix="lt-test-config-")
