"""Channel roles and panels (saved channel assignments)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .image import ChannelInfo

ROLES = ("Nucleus", "Actin", "Target", "Comparison", "Membrane", "Other", "Ignore")   # Comparison: D83
SINGLE_ROLES = ("Nucleus", "Actin", "Membrane")   # at most one channel each

NUCLEAR_DYES = ("dapi", "hoechst", "h3342", "h33342", "draq5", "sytox", "topro", "to-pro")

# Okabe-Ito based display colours for channels without a colour in the file
DEFAULT_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00", "#0072B2", "#F0E442")
DISPLAY_COLORS = {
    "Blue": "#3D7DFF", "Green": "#2BD46B", "Red": "#FF3B30", "Magenta": "#FF2BD6",
    "Cyan": "#00D7E6", "Yellow": "#FFD60A", "Orange": "#FF9500", "Gray": "#C8C8C8",
}


@dataclass
class ChannelAssignment:
    role: str
    label: str
    color: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")


@dataclass
class Panel:
    """A named set of channel assignments for one channel layout."""

    name: str
    signature: tuple[str, ...]
    assignments: list[ChannelAssignment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "signature": list(self.signature),
                "assignments": [asdict(a) for a in self.assignments]}

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        return cls(name=str(d["name"]), signature=tuple(d["signature"]),
                   assignments=[ChannelAssignment(**a) for a in d["assignments"]])


def _is_nuclear(ch: ChannelInfo) -> bool:
    text = f"{ch.fluor or ''} {ch.name}".lower().replace(" ", "")
    return any(dye.replace("-", "") in text.replace("-", "") for dye in NUCLEAR_DYES)


ACTIN_WORDS = ("act", "actin", "phalloidin", "phall", "pha")
NOT_A_STAIN = {"dapi", "hoechst", "af", "alexa", "fluor", "alexafluor", "airyscan", "processing", "hoxb", "cell",
               "cells", "ab", "nd", "control", "staining", "treated", "il", "anti", "image", "neutrophils",
               "neutrophil", "new", "fmlp", "c5a", "lps", "kc", "il8", "dmso", "vehicle", "wt", "ko", "kd"}


def _wavelength(ch: ChannelInfo) -> str | None:
    """'647' for 'Alexa Fluor 647', 'AF647-T8' or an excitation of 647 nm."""
    import re

    m = re.search(r"(?<!\d)([3-7]\d\d)(?!\d)", f"{ch.fluor or ''} {ch.name}")
    if m:
        return m.group(1)
    if ch.excitation_nm:
        return str(int(round(ch.excitation_nm)))
    return None


def _stains_from_name(text: str, wavelengths: list[str], skip_words=()) -> dict[str, str]:
    """Words written next to wavelengths in a file name: 'TTN647', '488Act', 'TTN 647', '647-TTN'.

    The name is split into tokens. A token with the wavelength attached ('TTN647', '568TTN') gives its
    word directly. A wavelength standing alone takes the stain word before it, else the one after it,
    in the order the wavelengths appear, and every word is used once. So '647 TTN 555 Vim' and
    'TTN 647 Vim 555' are both read right. Condition words (skip_words) and dye names are not stains.
    """
    import re

    stop = NOT_A_STAIN | {str(w).lower() for w in skip_words}
    clean = re.sub(r"(?i)(alexa\s*fluor|alexa|af)[ _-]?(?=[3-7]\d\d(?!\d))", "", text)   # 'AF647' -> '647'
    tokens = [t for t in re.split(r"[\s_+.,;()\[\]-]+", clean) if t]

    def stain(word: str) -> bool:
        w = word.lower()
        return (len(w) >= 2 and w.isalnum() and not w.isdigit() and w not in stop
                and not w.startswith(("hoxb", "airyscan")) and not re.fullmatch(r"\d+[a-z]?", w))

    found: dict[str, str] = {}
    used: set[int] = set()
    for nm in wavelengths:                                   # attached forms first
        for k, tok in enumerate(tokens):
            m = re.fullmatch(rf"([A-Za-z][A-Za-z0-9]*?){nm}|{nm}([A-Za-z][A-Za-z0-9]*)", tok)
            if m:
                word = re.split(r"[3-7]\d\d", m.group(1) or m.group(2))[-1 if m.group(1) else 0]
                if stain(word):
                    found[nm] = word
                    used.add(k)
                    break
    alone = sorted((k, tok) for k, tok in enumerate(tokens) if tok in wavelengths and tok not in found)
    for k, nm in alone:
        for j in (k - 1, k + 1):
            if 0 <= j < len(tokens) and j not in used and stain(tokens[j]) and tokens[j] not in wavelengths:
                found[nm] = tokens[j]
                used.add(j)
                break
    return found


def suggest_assignments(channels: list[ChannelInfo], file_name: str = "", skip_words=()) -> list[ChannelAssignment]:
    """First guess, the user decides: nuclear dyes become Nucleus. A wavelength written next to a word
    in the file name ('Act488', 'TTN647', '568TTN', '647 TTN') gives Actin for actin words, else a Target
    with that name. Everything else is Other. Colours that are too similar are replaced."""
    from .conditions import DEFAULT_VOCABULARY

    waves = [_wavelength(ch) for ch in channels]
    words = _stains_from_name(file_name, [w for w in waves if w], tuple(DEFAULT_VOCABULARY) + tuple(skip_words)) \
        if file_name else {}
    result = []
    nucleus_taken = actin_taken = False
    for i, ch in enumerate(channels):
        word = words.get(waves[i]) if waves[i] else None
        if _is_nuclear(ch) and not nucleus_taken:
            role, label = "Nucleus", "Nucleus"
            nucleus_taken = True
        elif word and word.lower() in ACTIN_WORDS and not actin_taken:
            role, label = "Actin", "Actin"
            actin_taken = True
        elif word and word.lower() not in ACTIN_WORDS:
            role, label = "Target", word
        else:
            role, label = "Other", ch.short
        if role == "Nucleus":
            color = ch.color or DISPLAY_COLORS["Blue"]
        elif role == "Actin":
            color = DISPLAY_COLORS["Green"]
        else:
            color = ch.color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        result.append(ChannelAssignment(role=role, label=label, color=color))
    _distinct_colors(result)
    labels = [a.label for a in result]
    for k, a in enumerate(result):                 # two channels with the same word: keep names unique
        if labels.count(a.label) > 1:
            a.label = f"{a.label}{k + 1}"
    return result


def _rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _distinct_colors(assignments: list[ChannelAssignment], min_distance: float = 140.0) -> None:
    """Replace a colour that is hard to tell from an earlier one (e.g. two oranges for 555 and 647)."""
    palette = [DISPLAY_COLORS[k] for k in ("Magenta", "Cyan", "Yellow", "Red", "Orange", "Gray")]
    used: list[tuple[int, int, int]] = []
    for a in assignments:
        rgb = _rgb(a.color)
        if any(sum((x - y) ** 2 for x, y in zip(rgb, u)) ** 0.5 < min_distance for u in used):
            for c in palette:
                cand = _rgb(c)
                if all(sum((x - y) ** 2 for x, y in zip(cand, u)) ** 0.5 >= min_distance for u in used):
                    a.color = c
                    rgb = cand
                    break
        used.append(rgb)


def validate(assignments: list[ChannelAssignment]) -> list[str]:
    """Problems that prevent an analysis. Empty list means the assignment is usable."""
    problems = []
    for role in SINGLE_ROLES:
        n = sum(a.role == role for a in assignments)
        if n > 1:
            problems.append(f"Only one channel can have the role {role} ({n} chosen).")
    if not any(a.role == "Target" for a in assignments):
        problems.append("At least one channel must be a Target (the protein to localize).")
    if not any(a.role == "Nucleus" for a in assignments):
        problems.append("No Nucleus channel chosen. Nucleus and cytoplasm values stay empty.")
    labels = [a.label.strip() for a in assignments if a.role != "Ignore"]
    if any(not lab for lab in labels):
        problems.append("Every channel that is not ignored needs a name.")
    if len(set(labels)) != len(labels):
        problems.append("Channel names must be unique.")
    return problems


def load_panels(path: Path) -> list[Panel]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return [Panel.from_dict(d) for d in raw.get("panels", [])]
    except (ValueError, KeyError, TypeError, AttributeError):
        try:                                       # keep the damaged file instead of overwriting it later
            path.replace(path.with_name(path.stem + ".damaged.json"))
        except OSError:
            pass
        return []


def save_panels(path: Path, panels: list[Panel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "panels": [p.to_dict() for p in panels]}, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


def find_panel(panels: list[Panel], signature: tuple[str, ...]) -> Panel | None:
    for p in reversed(panels):   # most recently saved wins
        if tuple(p.signature) == tuple(signature):
            return p
    return None


def upsert_panel(panels: list[Panel], panel: Panel) -> list[Panel]:
    """Replace a panel with the same name, else append."""
    out = [p for p in panels if p.name != panel.name]
    out.append(panel)
    return out
