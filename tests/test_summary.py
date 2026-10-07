from sanketswasth.summary import build_summary
from sanketswasth.pictograms import find_pictograms


def test_diabetes_prompt():
    r = build_summary(["thirst", "tiredness", "three_days"])
    assert any("blood sugar" in s for s in r["suggestions"])
    assert "excessive thirst" in r["sentence"]


def test_urgent_first():
    r = build_summary(["thirst", "tiredness", "chest", "breathless"])
    assert r["suggestions"][0].startswith("Urgent:")


def test_unknown_signs_ignored_and_empty():
    assert build_summary(["nonsense"])["sentence"] == "No signs confirmed yet."


def test_pictograms():
    assert find_pictograms("Take this tablet and drink water") != []