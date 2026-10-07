"""Real-time consultation app with the confidence-aware safety layer.

    python -m sanketswasth.app

Keys:  c = confirm suggested sign   x = reject   ENTER = finish & show summary
       r = doctor reply (type in terminal)        q = quit

Safety logic: a sign is accepted automatically only when the model is very
sure. If it is only moderately sure, the app ASKS the patient to confirm
(rather than guessing). Below that, it ignores the input.
"""
import collections
import json
import os
import cv2
import numpy as np
import tensorflow as tf
from .landmarks import make_holistic, process, to_vector, draw
from .summary import build_summary
from .pictograms import find_pictograms

SEQ_LEN = 30
AUTO_ACCEPT = 0.90   # accept without asking
ASK_CONFIRM = 0.60   # between this and AUTO_ACCEPT -> ask patient
STABLE_FRAMES = 6    # same prediction this many times in a row


def main(model_dir="models", camera=0):
    model = tf.keras.models.load_model(os.path.join(model_dir, "sign_model.keras"))
    labels = json.load(open(os.path.join(model_dir, "labels.json")))
    window = collections.deque(maxlen=SEQ_LEN)
    recent = collections.deque(maxlen=STABLE_FRAMES)
    confirmed, pending, banner = [], None, "Sign now..."
    cap = cv2.VideoCapture(camera)

    with make_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = process(frame, holistic)
            draw(frame, res)
            window.append(to_vector(res))

            if len(window) == SEQ_LEN and pending is None:
                probs = model.predict(np.expand_dims(np.array(window), 0), verbose=0)[0]
                i = int(probs.argmax())
                conf = float(probs[i])
                recent.append(i if conf >= ASK_CONFIRM else -1)
                if len(recent) == STABLE_FRAMES and len(set(recent)) == 1 and recent[0] != -1:
                    label = labels[i]
                    if conf >= AUTO_ACCEPT:
                        confirmed.append(label)
                        banner = f"Recognised: {label} ({conf:.0%})"
                    else:
                        pending = label
                        banner = f"Did you mean '{label}'?  c=yes  x=no"
                    recent.clear()
                    window.clear()

            cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, ", ".join(confirmed[-5:]), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("SanketSwasth", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if pending and key == ord("c"):
                confirmed.append(pending); banner = f"Confirmed: {pending}"; pending = None
            elif pending and key == ord("x"):
                banner = "Rejected. Please sign again."; pending = None
            elif key == 13:  # ENTER
                s = build_summary(confirmed)
                print("\n=== SUMMARY FOR CLINICIAN ===\n" + s["sentence"])
                for tip in s["suggestions"]:
                    print(" ->", tip)
            elif key == ord("r"):
                text = input("Doctor says: ")
                shown = find_pictograms(text)
                print("Pictograms shown to patient:", shown or "(none matched)")
                for path in shown:
                    img = cv2.imread(path)
                    if img is not None:
                        cv2.imshow(os.path.basename(path), img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()