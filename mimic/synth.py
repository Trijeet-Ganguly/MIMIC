"""Generate a human-like synthetic demonstration (noisy hand path, pinch events, a mistake + correction).

Lets you develop and test the whole pipeline with no camera, no mediapipe, no robot.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .models import Frame, Scene


class DemoBuilder:
    REST = np.array([-8.0, -8.0])  # where the hand waits, off the table

    def __init__(self, scene: Scene, seed: int = 1, fps: float = 30.0):
        self.objs = {k: np.array(v, dtype=float) for k, v in scene.items()}
        self.hand = self.REST.copy()
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / fps
        self.t = 0.0
        self.holding = None
        self.pinch = False
        self.frames: List[Frame] = []
        self._emit()

    def _emit(self) -> None:
        if self.holding:
            self.objs[self.holding] = self.hand.copy()
        self.frames.append(Frame(self.t, {k: v.copy() for k, v in self.objs.items()},
                                 self.hand.copy(), self.pinch))
        self.t += self.dt

    def goto(self, p, n: int = 25, wobble: float = 0.15) -> None:
        start, p = self.hand.copy(), np.asarray(p, dtype=float)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep: humans accelerate/decelerate
            self.hand = start + (p - start) * s + self.rng.normal(0, wobble, 2)
            self._emit()

    def grab(self, name: str) -> None:
        self.pinch, self.holding = True, name
        for _ in range(5):
            self._emit()

    def release(self) -> None:
        self.pinch, self.holding = False, None
        for _ in range(5):
            self._emit()

    def pick_place(self, name: str, dst, sloppiness: float = 0.5) -> None:
        self.goto(self.objs[name] + self.rng.normal(0, 0.3, 2))
        self.grab(name)
        self.goto(np.asarray(dst, dtype=float) + self.rng.normal(0, sloppiness, 2), n=35)
        self.release()

    def finish(self) -> List[Frame]:
        self.goto(self.REST)
        for _ in range(10):
            self._emit()
        return self.frames
