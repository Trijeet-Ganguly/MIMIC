"""Stage 1 - SEE (real hardware): overhead webcam -> table coordinates.

  * TableCalibration : image pixels  -> table cm (4 clicked corners, homography)
  * ObjectTracker    : ArUco markers stuck on each tool -> {name: (x, y)}   (keeps last-known when hidden by the hand)
  * HandTracker      : MediaPipe hand -> (fingertip midpoint in table cm, pinching?)

Needs: pip install opencv-contrib-python mediapipe
(cv2.aruco lives in opencv-contrib; ArucoDetector needs OpenCV >= 4.7)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Scene


def _cv2():
    import cv2
    return cv2


class TableCalibration:
    def __init__(self, H: np.ndarray):
        self.H = np.asarray(H, dtype=np.float64)

    @classmethod
    def from_points(cls, img_pts: List[Tuple[float, float]], width_cm: float, height_cm: float):
        """img_pts order: front-left, front-right, back-right, back-left
        ('front' = the table edge nearest the robot; left/right as seen from the robot)."""
        cv2 = _cv2()
        dst = np.float32([[0, 0], [width_cm, 0], [width_cm, height_cm], [0, height_cm]])
        return cls(cv2.getPerspectiveTransform(np.float32(img_pts), dst))

    def to_table(self, px) -> np.ndarray:
        cv2 = _cv2()
        pt = cv2.perspectiveTransform(np.float32([[[px[0], px[1]]]]), self.H)[0, 0]
        return np.array([float(pt[0]), float(pt[1])])

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"H": self.H.tolist()}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(np.array(json.load(f)["H"]))


def calibrate_interactive(cap, width_cm: float, height_cm: float, path: str) -> TableCalibration:
    cv2 = _cv2()
    pts: List[Tuple[int, int]] = []
    labels = ["front-left", "front-right", "back-right", "back-left"]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 6, (0, 255, 0), -1)
            cv2.putText(frame, labels[i], (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        msg = f"click: {labels[len(pts)]}" if len(pts) < 4 else "done - press any key"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(30)
        if len(pts) == 4 and key != -1:
            break
    cv2.destroyWindow("calibrate")
    cal = TableCalibration.from_points(pts, width_cm, height_cm)
    cal.save(path)
    return cal


class ObjectTracker:
    """Print one ArUco marker (DICT_4X4_50) per tool and stick it on top.
    id_to_name e.g. {0: "tape", 1: "hammer", 2: "wrench"}"""

    def __init__(self, calib: TableCalibration, id_to_name: Dict[int, str]):
        cv2 = _cv2()
        self.calib, self.id_to_name = calib, id_to_name
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.last: Scene = {}

    def update(self, frame_bgr) -> Scene:
        corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                name = self.id_to_name.get(int(i))
                if name:
                    self.last[name] = self.calib.to_table(c[0].mean(axis=0))
        return {k: v.copy() for k, v in self.last.items()}  # hidden objects keep last-known position


class HandTracker:
    """Pinch = thumb tip close to index tip (relative to hand size), with hysteresis to avoid flicker."""

    def __init__(self, calib: TableCalibration, pinch_on: float = 0.35, pinch_off: float = 0.55):
        import mediapipe as mp
        self.calib = calib
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                                              min_tracking_confidence=0.6)
        self.pinch_on, self.pinch_off = pinch_on, pinch_off
        self.pinching = False

    def update(self, frame_bgr) -> Tuple[Optional[np.ndarray], bool]:
        cv2 = _cv2()
        res = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None, self.pinching
        lm = res.multi_hand_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        def P(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        thumb, index, wrist, mcp = P(4), P(8), P(0), P(9)
        ratio = np.linalg.norm(thumb - index) / (np.linalg.norm(wrist - mcp) + 1e-6)
        if self.pinching and ratio > self.pinch_off:
            self.pinching = False
        elif not self.pinching and ratio < self.pinch_on:
            self.pinching = True
        return self.calib.to_table((thumb + index) / 2), self.pinching
