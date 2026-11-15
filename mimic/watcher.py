"""Stage 2 - WATCH: turn raw frames (hand + objects) into discrete pick/place actions."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .models import Action, Frame


def _nearest(objects, xy) -> Tuple[Optional[str], float]:
    best, best_d = None, float("inf")
    for name, pos in objects.items():
        d = float(np.linalg.norm(pos - xy))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


def segment(frames: List[Frame], grasp_radius: float = 5.0, min_move: float = 2.0) -> List[Action]:
    """Detect 'pinch near object ... release' episodes.

    grasp_radius: how close (cm) the hand must be to an object when pinching to count as grabbing it.
    min_move:     ignore grabs that moved the object less than this (fidgeting).
    """
    actions: List[Action] = []
    holding: Optional[str] = None
    pick_xy = np.zeros(2)
    t_start = 0.0
    waypoints = 0

    for f in frames:
        if f.hand_xy is None:
            continue
        if holding is None:
            if f.pinching:
                name, d = _nearest(f.objects, f.hand_xy)
                if name is not None and d <= grasp_radius:
                    holding = name
                    pick_xy = f.objects[name].copy()
                    t_start = f.t
                    waypoints = 0
        else:
            waypoints += 1
            if not f.pinching:  # released
                place_xy = f.hand_xy.copy()
                if np.linalg.norm(place_xy - pick_xy) >= min_move:
                    actions.append(Action(holding, pick_xy, place_xy, t_start, f.t, waypoints))
                holding = None
    return actions


def initial_scene(frames: List[Frame]):
    return {k: v.copy() for k, v in frames[0].objects.items()}


def final_scene(frames: List[Frame]):
    return {k: v.copy() for k, v in frames[-1].objects.items()}
