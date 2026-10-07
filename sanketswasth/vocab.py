"""Starter medical vocabulary.

Each key is a sign label (a folder name under data/). The labels are
placeholders: replace/extend them with the signs your deaf community partners
actually use (ISL varies by region). Category drives sentence building.
"""

# label -> (category, human-readable text)
SIGNS = {
    # symptoms
    "fever": ("symptom", "fever"),
    "cough": ("symptom", "cough"),
    "pain": ("symptom", "pain"),
    "vomiting": ("symptom", "vomiting"),
    "dizziness": ("symptom", "dizziness"),
    "tiredness": ("symptom", "unusual tiredness"),
    "thirst": ("symptom", "excessive thirst"),
    "frequent_urination": ("symptom", "frequent urination"),
    "breathless": ("symptom", "shortness of breath"),
    "blurred_vision": ("symptom", "blurred vision"),
    "headache": ("symptom", "headache"),
    # body parts
    "head": ("body", "head"),
    "chest": ("body", "chest"),
    "stomach": ("body", "stomach"),
    "back": ("body", "back"),
    "leg": ("body", "leg"),
    # duration
    "one_day": ("duration", "1 day"),
    "three_days": ("duration", "3 days"),
    "one_week": ("duration", "1 week"),
    "long_time": ("duration", "a long time (over 2 weeks)"),
    # status
    "pregnant": ("status", "pregnant"),
    "medicine_taking": ("status", "currently taking medicine"),
    "diabetes_family": ("status", "family history of diabetes"),
    "allergy": ("status", "has an allergy"),
}

LABELS = sorted(SIGNS)