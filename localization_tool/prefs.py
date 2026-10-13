"""Program-wide preferences in the user's settings folder (no Qt here).

- defaults.json: the settings every new analysis starts with (Settings on the start page, D59)
- recent.json: the analyses opened or saved last (start page)
- ui.json: interface preferences, e.g. the text size (D61)
"""

from __future__ import annotations

import json
from pathlib import Path

from localization_tool.core.params import AnalysisParams
from localization_tool.paths import user_config_dir

RECENT_MAX = 6
FONT_SCALES = (1.0, 1.15, 1.3)


def _read(name: str) -> dict:
    try:
        d = json.loads((user_config_dir() / name).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(name: str, data: dict) -> None:
    try:
        (user_config_dir() / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass                                   # preferences are a convenience, never stop the program


def load_defaults() -> tuple[AnalysisParams, bool]:
    """Settings and blind review for a new analysis. A damaged file gives the program defaults."""
    d = _read("defaults.json")
    try:
        params = AnalysisParams.from_dict(d.get("params") or {})
    except ValueError:
        params = AnalysisParams()
    return params, bool(d.get("blind", True))


def save_defaults(params: AnalysisParams, blind: bool, vocabulary: list[str] | None = None) -> None:
    d = {"params": params.to_dict(), "blind": bool(blind)}
    if vocabulary is not None:
        d["vocabulary"] = [str(v) for v in vocabulary]
    _write("defaults.json", d)


def load_vocabulary() -> list[str] | None:
    """Condition words for new analyses (D37), None if never changed on the start page."""
    v = _read("defaults.json").get("vocabulary")
    return [str(x) for x in v] if isinstance(v, list) else None


def recent_analyses() -> list[Path]:
    return [Path(p) for p in _read("recent.json").get("paths", []) if Path(p).exists()][:RECENT_MAX]


def add_recent(path: Path) -> None:
    path = Path(path)
    paths = [str(path)] + [p for p in _read("recent.json").get("paths", []) if p != str(path)]
    _write("recent.json", {"paths": paths[:RECENT_MAX]})


def font_scale() -> float:
    v = _read("ui.json").get("font_scale", 1.0)
    return float(v) if v in FONT_SCALES else 1.0


def set_font_scale(value: float) -> None:
    d = _read("ui.json")
    d["font_scale"] = float(value)
    _write("ui.json", d)
