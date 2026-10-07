"""Record training samples for each sign.

    python -m sanketswasth.collect --sign fever --samples 30

Press SPACE to record one 30-frame sequence, q to quit.
Get informed consent from every volunteer. Only landmark numbers are saved,
never the video.
"""
import argparse
import os
import cv2
import numpy as np
from .landmarks import make_holistic, process, to_vector, draw
from .vocab import LABELS

SEQ_LEN = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", required=True, choices=LABELS)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--out", default="data")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    folder = os.path.join(a.out, a.sign)
    os.makedirs(folder, exist_ok=True)
    count = len(os.listdir(folder))
    cap = cv2.VideoCapture(a.camera)
    with make_holistic() as holistic:
        while count < a.samples:
            ok, frame = cap.read()
            if not ok:
                break
            res = process(frame, holistic)
            draw(frame, res)
            cv2.putText(frame, f"{a.sign}: {count}/{a.samples}  SPACE=record q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                seq = []
                for _ in range(SEQ_LEN):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    res = process(frame, holistic)
                    seq.append(to_vector(res))
                    draw(frame, res)
                    cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.imshow("collect", frame)
                    cv2.waitKey(1)
                if len(seq) == SEQ_LEN:
                    np.save(os.path.join(folder, f"{count}.npy"), np.array(seq))
                    count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()