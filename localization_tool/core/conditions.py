"""Suggest experiment and condition from the file name (D37). Only a suggestion: the fields stay editable.

Condition: the words of a vocabulary that appear in the file name or its folder, in the order of
the vocabulary, e.g. "WT treated". Matching ignores case and treats "-", "_", "+", "." and spaces
as separators, so "IL-8", "IL8" and "il_8" all match the word "IL8" of a vocabulary that has it.
Experiment: the nearest folder above the file whose name holds a date, as ISO date if the date is
unambiguous (yyyymmdd, yyyy-mm-dd, or dd.mm.yyyy / mm.dd.yyyy where one part is above 12), otherwise
that folder's name. Without such a folder: a date in the file name, else the folder of the file.
Folders are searched first because file names are sometimes copied between experiments.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

# general words; a lab adds its own under Settings (saved as program defaults or with the analysis)
DEFAULT_VOCABULARY = ("WT", "KO", "KD", "control", "treated", "untreated", "vehicle", "stimulated", "resting")


def _norm(text: str) -> str:
    return re.sub(r"[\s\-_+.]+", " ", text).lower()


def _key(word: str) -> str:
    return re.sub(r"[\s\-_+.]+", "", word).lower()


LEVELS = 4


def suggest_condition(path: str | Path, vocabulary) -> str:
    p = Path(path)
    parts = [q.name for q in list(p.parents)[:LEVELS]]
    tokens = _norm(" ".join(parts[::-1] + [p.stem])).split()
    joined = {t for t in tokens}
    # also pairs of neighbouring tokens, so "IL 8" matches "IL8"
    joined |= {a + b for a, b in zip(tokens, tokens[1:])}
    found = []
    for word in vocabulary:
        w = str(word).strip()
        if w and _key(w) in joined and w not in found:
            found.append(w)
    return " ".join(found)


def _valid(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _date_in(text: str) -> tuple[bool, str | None]:
    """(text holds a date, ISO date if unambiguous)."""
    m = re.search(r"(?<!\d)(20\d\d)[-_.]?(\d\d)[-_.]?(\d\d)(?!\d)", text)
    if m:
        iso = _valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            return True, iso
    m = re.search(r"(?<!\d)(\d\d)[-_.](\d\d)[-_.](20\d\d)(?!\d)", text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 and b <= 12:
            return True, _valid(y, b, a)
        if b > 12 and a <= 12:
            return True, _valid(y, a, b)
        if a == b:
            return True, _valid(y, a, b)
        return True, None                     # day and month cannot be told apart
    return False, None


def suggest_experiment(path: str | Path) -> str:
    p = Path(path)
    for q in list(p.parents)[:LEVELS]:
        has, iso = _date_in(q.name)
        if has:
            return iso or q.name
    has, iso = _date_in(p.stem)
    if has and iso:
        return iso
    return p.parent.name
