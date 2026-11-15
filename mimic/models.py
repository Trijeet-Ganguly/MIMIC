"""Shared data types for MIMIC.

All positions are TABLE coordinates in centimetres:
  x grows to the robot's right, y grows away from the robot (front edge of table is y=0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

Scene = Dict[str, np.ndarray]  # object name -> (x, y)
TABLE = "TABLE"                # anchor meaning "absolute table position"


@dataclass
class Frame:
    """One time-step of the demonstration."""
    t: float
    objects: Scene                          # last known position of every tagged object
    hand_xy: Optional[np.ndarray] = None    # midpoint of thumb+index tips (None = hand not visible)
    pinching: bool = False                  # True while the hand is gripping something


@dataclass
class Action:
    """One pick-and-place observed in the demo (low-level, before 'understanding')."""
    obj: str
    pick_xy: np.ndarray
    place_xy: np.ndarray
    t_start: float
    t_end: float
    waypoints: int


@dataclass
class Placement:
    """Learned goal for one object: 'be at <anchor position> + <offset>'.

    anchor == TABLE  ->  `offset` is an absolute table position.
    """
    obj: str
    anchor: str
    offset: np.ndarray
    why: str


@dataclass
class Goal:
    """What the robot believes the demonstrator wanted (the 'why')."""
    hypothesis: str
    confidence: float
    placements: List[Placement]
    explanation: List[str] = field(default_factory=list)
    tolerance: float = 1.5  # cm

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "tolerance": self.tolerance,
            "explanation": self.explanation,
            "placements": [
                {"obj": p.obj, "anchor": p.anchor, "offset": [float(v) for v in p.offset], "why": p.why}
                for p in self.placements
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Goal":
        return cls(
            hypothesis=d["hypothesis"],
            confidence=d["confidence"],
            tolerance=d.get("tolerance", 1.5),
            explanation=d.get("explanation", []),
            placements=[
                Placement(p["obj"], p["anchor"], np.array(p["offset"], dtype=float), p["why"])
                for p in d["placements"]
            ],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Goal":
        with open(path) as f:
            return cls.from_dict(json.load(f))
