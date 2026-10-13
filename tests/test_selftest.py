from localization_tool.selftest import main


def test_selftest_passes(tmp_path):
    report = tmp_path / "r.txt"
    assert main(str(report)) == 0, report.read_text()
    assert "PASSED" in report.read_text()
