"""Camera frame -> compact landmark vector using MediaPipe Holistic.

Using body/hand skeleton points (not raw video) keeps the model tiny,
fast on a phone, and avoids storing faces: good for privacy.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

FEATURES = 33 * 4 + 21 * 3 + 21 * 3  # pose(x,y,z,vis) + left hand + right hand = 258


def make_holistic():
    return mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def process(frame_bgr, holistic):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return holistic.process(rgb)


def to_vector(results) -> np.ndarray:
    pose = (np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    lh = (np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3))
    rh = (np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, lh, rh]).astype(np.float32)


def draw(frame, results):
    for lm, conn in ((results.pose_landmarks, mp_holistic.POSE_CONNECTIONS),
                     (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
                     (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)):
        if lm:
            mp_draw.draw_landmarks(frame, lm, conn)