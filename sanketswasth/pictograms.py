"""Maps a doctor's reply to pictogram images for the patient."""
import os

KEYWORDS = {
    "tablet": ["tablet", "medicine", "pill", "dawai"],
    "water": ["water", "drink", "fluids"],
    "rest": ["rest", "sleep", "relax"],
    "hospital": ["hospital", "refer", "emergency", "urgent"],
    "morning": ["morning", "breakfast"],
    "night": ["night", "evening", "bedtime"],
    "blood_test": ["blood test", "sugar test", "lab"],
    "come_back": ["come back", "follow up", "review", "return"],
}


def find_pictograms(text: str, folder: str = "assets/pictograms") -> list[str]:
    t = text.lower()
    hits = []
    for name, words in KEYWORDS.items():
        if any(w in t for w in words):
            path = os.path.join(folder, f"{name}.png")
            hits.append(path if os.path.exists(path) else name)
    return hits