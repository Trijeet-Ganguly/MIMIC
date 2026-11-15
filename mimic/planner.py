"""Stage 4 - PLAN: turn the learned Goal into concrete steps for the scene as it is NOW.

Because the goal is relational ("6 cm right of tape"), targets are recomputed from wherever the
anchor objects currently are. The planner also handles conflicts (a target occupied by another
object) by reordering or by parking the blocker somewhere free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .models import Goal, Scene, TABLE

DEFAULT_WORKSPACE = ((0.0, 0.0), (36.0, 24.0))  # (min_xy, max_xy) in cm


class GoalInfeasible(Exception):
    pass


@dataclass
class Step:
    obj: str
    pick_xy: np.ndarray
    place_xy: np.ndarray
    reason: str


def resolve_targets(goal: Goal, scene: Scene) -> Dict[str, np.ndarray]:
    pl = {p.obj: p for p in goal.placements}
    memo: Dict[str, np.ndarray] = {}

    def tgt(o: str, depth: int = 0) -> np.ndarray:
        if o in memo:
            return memo[o]
        if depth > len(pl) + 1:
            raise GoalInfeasible("circular anchors in goal")
        p = pl[o]
        if p.anchor == TABLE:
            r = p.offset.copy()
        elif p.anchor in pl:
            r = tgt(p.anchor, depth + 1) + p.offset
        elif p.anchor in scene:
            r = scene[p.anchor] + p.offset
        else:
            raise GoalInfeasible(f"can't see anchor object '{p.anchor}'")
        memo[o] = r
        return r

    return {o: tgt(o) for o in pl}


def _free_spot(world: Scene, targets: Dict[str, np.ndarray], near: np.ndarray, ws, clearance: float) -> np.ndarray:
    (x0, y0), (x1, y1) = ws
    taken = list(world.values()) + list(targets.values())
    best, best_d = None, float("inf")
    for x in np.arange(x0 + 2, x1 - 1, 2.0):
        for y in np.arange(y0 + 2, y1 - 1, 2.0):
            p = np.array([x, y])
            if all(np.linalg.norm(p - q) >= clearance for q in taken):
                d = float(np.linalg.norm(p - near))
                if d < best_d:
                    best, best_d = p, d
    if best is None:
        raise GoalInfeasible("no free space on the table")
    return best


def plan(goal: Goal, scene: Scene, workspace=DEFAULT_WORKSPACE, clearance: float = 5.0) -> List[Step]:
    for p in goal.placements:
        if p.obj not in scene:
            raise GoalInfeasible(f"can't see object '{p.obj}'")
    targets = resolve_targets(goal, scene)
    (x0, y0), (x1, y1) = workspace
    for o, t in targets.items():
        if not (x0 <= t[0] <= x1 and y0 <= t[1] <= y1):
            raise GoalInfeasible(f"target for '{o}' ({t[0]:.1f}, {t[1]:.1f}) is outside the workspace")

    pending = [p.obj for p in goal.placements
               if np.linalg.norm(scene[p.obj] - targets[p.obj]) > goal.tolerance]
    world = {k: v.copy() for k, v in scene.items()}
    steps: List[Step] = []
    defers: Dict[str, int] = {}

    for _ in range(100):
        if not pending:
            return steps
        obj = pending[0]
        tgt = targets[obj]
        blockers = [o for o, p in world.items() if o != obj and np.linalg.norm(p - tgt) < clearance]
        if blockers:
            b = blockers[0]
            if b in targets and b not in pending:
                raise GoalInfeasible(f"'{b}' already sits where '{obj}' must go")
            if b in pending and defers.get(obj, 0) < 1:
                defers[obj] = defers.get(obj, 0) + 1
                pending.remove(b)
                pending.insert(0, b)
                continue
            spot = _free_spot(world, targets, world[b], workspace, clearance)
            steps.append(Step(b, world[b].copy(), spot, f"clear the target spot for '{obj}'"))
            world[b] = spot
            continue
        steps.append(Step(obj, world[obj].copy(), tgt.copy(), "move to learned goal position"))
        world[obj] = tgt.copy()
        pending.pop(0)
    raise GoalInfeasible("planner could not resolve conflicts")
