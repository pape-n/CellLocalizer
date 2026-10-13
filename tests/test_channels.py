from localization_tool.core.channels import (
    ChannelAssignment, Panel, find_panel, load_panels, save_panels, suggest_assignments, upsert_panel, validate,
)
from localization_tool.core.image import ChannelInfo


def chans():
    return [ChannelInfo(0, "AF647-T8", "Alexa Fluor 647", 653, 668),
            ChannelInfo(1, "AF488-T10", "Alexa Fluor 488", 493, 517, "#00FF33"),
            ChannelInfo(2, "DAPI-T11", "DAPI", 353, 465, "#00A0FF")]


def test_suggest_marks_dapi_as_nucleus_only():
    a = suggest_assignments(chans())
    assert [x.role for x in a] == ["Other", "Other", "Nucleus"]
    assert a[1].color == "#00FF33"


def test_hoechst_recognised():
    a = suggest_assignments([ChannelInfo(0, "H3342", "Hoechst 33342")])
    assert a[0].role == "Nucleus"


def test_validate():
    a = suggest_assignments(chans())
    assert any("Target" in p for p in validate(a))
    a[0].role, a[0].label = "Target", "TTN"
    a[1].role, a[1].label = "Actin", "Actin"
    assert validate(a) == []
    a[1].role = "Nucleus"
    assert any("Only one channel" in p for p in validate(a))


def test_two_targets_allowed():
    a = [ChannelAssignment("Target", "TTN", "#FFFFFF"), ChannelAssignment("Target", "Vim", "#FF0000"),
         ChannelAssignment("Nucleus", "DAPI", "#0000FF")]
    assert validate(a) == []


def test_duplicate_labels_rejected():
    a = [ChannelAssignment("Target", "X", "#FFFFFF"), ChannelAssignment("Nucleus", "X", "#0000FF")]
    assert any("unique" in p for p in validate(a))


def test_panel_store(tmp_path):
    path = tmp_path / "panels.json"
    sig = ("Alexa Fluor 647", "Alexa Fluor 488", "DAPI")
    p1 = Panel("TTN/Actin/DAPI", sig, [ChannelAssignment("Target", "TTN", "#FFFFFF"),
                                        ChannelAssignment("Actin", "Actin", "#00FF00"),
                                        ChannelAssignment("Nucleus", "DAPI", "#0000FF")])
    panels = upsert_panel([], p1)
    save_panels(path, panels)
    loaded = load_panels(path)
    assert find_panel(loaded, sig).assignments[0].label == "TTN"
    assert find_panel(loaded, ("other",)) is None
    p2 = Panel("TTN/Actin/DAPI", sig, p1.assignments[::-1])
    assert len(upsert_panel(loaded, p2)) == 1


def test_corrupt_panel_file_is_ignored(tmp_path):
    path = tmp_path / "panels.json"
    path.write_text("{not json")
    assert load_panels(path) == []
