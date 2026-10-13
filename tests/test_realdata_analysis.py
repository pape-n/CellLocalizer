"""Analysis on real lab files (only local: LT_TEST_DATA=<folder> pytest -m realdata).

Checks plausibility, not biology: the run finishes, shares add up to 100 %, masks are nested,
the z orientation comes from the metadata, and a cell takes less than 75 s (cloud container with two cores;
since 1.2 with the sensitivity span, about +50 %; lab computers are faster).
"""

import os
import time
from pathlib import Path

import pytest

from localization_tool.core.analysis import analyze_cell
from localization_tool.core.cli import assign_by_rules, parse_roles
from localization_tool.core.reader import ReaderError, read_image

REAL = os.environ.get("LT_TEST_DATA")
RULES = parse_roles("DAPI=Nucleus,488=Actin,647=Target:TTN,568=Target:TTN,555=Target:VIM")


@pytest.mark.realdata
@pytest.mark.skipif(not REAL, reason="LT_TEST_DATA not set")
def test_analysis_on_real_files():
    files = [f for f in sorted(Path(REAL).rglob("*.czi")) if "non processed" not in str(f).lower()]
    files = files[:: max(1, len(files) // 12)]          # a spread of about a dozen files
    done = 0
    for f in files:
        try:
            s = read_image(f)
        except ReaderError:
            continue
        roles = assign_by_rules(s.channels, RULES)
        if roles is None:
            continue
        t = time.perf_counter()
        r = analyze_cell(s, roles)
        assert time.perf_counter() - t < 75, f.name          # largest test image about 50 s in the cloud container (1.2)
        assert s.info.get("coverslip_first") is True, f.name
        m = r.masks
        assert m["cell"].any(), f.name
        assert not (m["nucleus"] & ~m["cell"]).any()
        assert not (m["actin"] & ~m["cell"]).any()
        for label, v in r.target_values.items():
            total = v["signal_nucleus_pct"] + v["signal_cytoplasm_pct"]
            if total != total:                   # D89: no staining above noise, no mask, no values
                assert not m[f"target_{label}"].any() and any("no staining" in n for n in r.notes), (f.name, label)
                continue
            assert total == pytest.approx(100.0, abs=1e-6), (f.name, label)
        done += 1
    assert done >= 3
