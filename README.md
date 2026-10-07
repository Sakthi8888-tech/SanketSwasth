# SanketSwasth 🤟🩺

**A sign-language health consultation bridge with preventive triage.**
*Giving every deaf patient a voice at the clinic, even where no interpreter exists.*

Built for the Seva Innovation Challenge, track **Swasth & Samavesh Bharat** (Healthcare innovation, Preventive health, Assistive technology, Inclusive solutions).

> ⚠️ **Status: research prototype.** This repo is a working scaffold. It ships **no trained model and no dataset**; you collect signs with deaf community partners and train it yourself. It is a communication aid for clinicians, **not** a diagnostic device and not a replacement for human interpreters.

## The problem
India has very few certified Indian Sign Language (ISL) interpreters, and almost none in village clinics. Deaf patients often can't describe symptoms, so diagnoses are delayed and preventive screening never reaches them.

## How it works
1. **Patient → doctor:** camera → MediaPipe body/hand landmarks → small LSTM recognises a restricted **medical** sign vocabulary → structured symptom summary.
2. **Doctor → patient:** the doctor's reply is matched to pictograms (later: ISL clips/avatar).
3. **Preventive layer:** rules suggest screening prompts (e.g. thirst + tiredness → blood sugar test).
4. **Confidence-aware safety:** high confidence → accept; medium → *ask the patient to confirm*; low → ignore. Never guesses silently.

## What makes it different
- Medical-domain-first (small vocabulary → realistic accuracy), not "translate all of ISL"
- Works offline (TFLite export, on-device)
- Privacy: only landmark numbers are stored, never video/faces
- Confirmation loop instead of silent guessing
- Preventive triage built in
- Roadmap: few-shot regional sign adaptation, federated learning, community screening-gap dashboard

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Try the summary + triage engine (no camera needed)
python -m sanketswasth.summary thirst tiredness three_days

# 2) Collect samples (repeat per sign, ~30+ per sign, multiple volunteers)
python -m sanketswasth.collect --sign fever --samples 30

# 3) Train + export TFLite
python -m sanketswasth.train --data data

# 4) Run the live consultation demo
python -m sanketswasth.app

# Tests
pytest
```

## Project structure
```
sanketswasth/
  vocab.py        starter medical signs (edit with community input)
  landmarks.py    MediaPipe landmark extraction
  collect.py      record training sequences
  train.py        LSTM training + TFLite export
  app.py          live demo with confidence-aware confirmation
  summary.py      sentence builder + preventive rules
  pictograms.py   doctor reply -> pictograms
tests/            unit tests for the rule engine
```

## Roadmap
- [ ] Collect dataset with deaf associations / special schools (with informed consent)
- [ ] Clinician review of preventive rules
- [ ] Mobile app (Android, TFLite) for ASHA/ANM workers
- [ ] ISL video clips / avatar for doctor → patient direction
- [ ] Few-shot "teach the app a new regional sign"
- [ ] Federated learning across clinics
- [ ] Anonymised community screening dashboard

## Responsible use
Sign vocabularies vary by region; always validate with the local deaf community. Recognition errors are possible: a human must confirm anything clinically important. Obtain informed consent for all data collection and follow applicable Indian data-protection requirements.

## License
MIT