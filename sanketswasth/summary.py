"""Turns recognised signs into a structured summary + preventive suggestions.

Pure Python (no ML deps) so it is easy to test and audit. Rules here are
illustrative screening *prompts* for a health worker, not diagnoses.
Have them reviewed by a clinician before any real-world use.
"""
from __future__ import annotations

import sys
from .vocab import SIGNS

# (required signs, suggestion)  -> fires if ALL required signs are present
PREVENTIVE_RULES = [
    ({"thirst", "tiredness"}, "Consider a blood sugar test (possible diabetes screening)."),
    ({"thirst", "frequent_urination"}, "Consider a blood sugar test (possible diabetes screening)."),
    ({"blurred_vision", "diabetes_family"}, "Consider blood sugar test and eye screening."),
    ({"chest", "breathless"}, "Urgent: check blood pressure and refer to a doctor immediately."),
    ({"headache", "dizziness"}, "Consider a blood pressure check."),
    ({"pregnant", "dizziness"}, "Check blood pressure and haemoglobin (anaemia / pre-eclampsia screening)."),
    ({"pregnant", "tiredness"}, "Check haemoglobin (anaemia screening) and confirm ANC visit schedule."),
    ({"cough", "long_time"}, "Consider TB screening (cough lasting over 2 weeks)."),
    ({"fever", "one_week"}, "Fever for a week: consider malaria/dengue/typhoid tests."),
]

URGENT_PREFIX = "Urgent:"


def build_summary(signs: list[str]) -> dict:
    """signs: ordered list of confirmed sign labels."""
    seen = list(dict.fromkeys(s for s in signs if s in SIGNS))  # dedupe, keep order
    by_cat: dict[str, list[str]] = {}
    for s in seen:
        cat, text = SIGNS[s]
        by_cat.setdefault(cat, []).append(text)

    parts = []
    if by_cat.get("symptom"):
        parts.append("Patient reports " + ", ".join(by_cat["symptom"]))
    if by_cat.get("body"):
        parts.append("location: " + ", ".join(by_cat["body"]))
    if by_cat.get("duration"):
        parts.append("duration: " + ", ".join(by_cat["duration"]))
    if by_cat.get("status"):
        parts.append("notes: " + ", ".join(by_cat["status"]))
    sentence = "; ".join(parts) + "." if parts else "No signs confirmed yet."

    present = set(seen)
    suggestions = []
    for required, text in PREVENTIVE_RULES:
        if required <= present and text not in suggestions:
            suggestions.append(text)
    suggestions.sort(key=lambda t: not t.startswith(URGENT_PREFIX))  # urgent first

    return {"signs": seen, "sentence": sentence, "suggestions": suggestions}


if __name__ == "__main__":
    # Try without a camera:  python -m sanketswasth.summary thirst tiredness three_days
    result = build_summary(sys.argv[1:])
    print(result["sentence"])
    for s in result["suggestions"]:
        print(" ->", s)