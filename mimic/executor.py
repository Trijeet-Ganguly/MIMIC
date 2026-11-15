"""Stage 5/6 - ACT + ADAPT: closed-loop execution.

After EVERY step the robot looks again, re-plans from what it sees, and only then does the next step.
So if a human moves something, a grasp slips, or a placed object gets knocked, the next plan simply
differs - no special-case code needed.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .arm import Arm, Unreachable
from .models import Goal, Scene
from .planner import DEFAULT_WORKSPACE, GoalInfeasible, Step, plan


class Executor:
    def __init__(self, arm: Arm, observe: Callable[[], Scene], goal: Goal,
                 workspace=DEFAULT_WORKSPACE, log: Callable[[str], None] = print):
        self.arm, self.observe, self.goal = arm, observe, goal
        self.workspace, self.log = workspace, log

    def perform(self, step: Step) -> bool:
        g = self.arm.geometry
        a = self.arm
        a.set_gripper(False)
        a.move_to(step.pick_xy, g.safe_z)
        a.move_to(step.pick_xy, g.grasp_z)
        if not a.set_gripper(True):
            self.log(f"   ✘ couldn't grasp '{step.obj}' — will look again and retry")
            a.set_gripper(False)
            a.move_to(step.pick_xy, g.safe_z)
            a.home()
            return False
        a.move_to(step.pick_xy, g.safe_z)
        a.move_to(step.place_xy, g.safe_z)
        a.move_to(step.place_xy, g.place_z)
        a.set_gripper(False)
        a.move_to(step.place_xy, g.safe_z)
        a.home()  # get out of the camera's view before re-observing
        return True

    def run(self, on_step: Optional[Callable[[int, Step, bool], None]] = None,
            max_iter: Optional[int] = None) -> bool:
        n = len(self.goal.placements)
        max_iter = max_iter or 4 * n + 8
        expected: Optional[Scene] = None

        for it in range(max_iter):
            scene = self.observe()

            if expected is not None:  # did the world match what I expected?
                for o, exp in expected.items():
                    if o in scene and np.linalg.norm(scene[o] - exp) > 2.5:
                        self.log(f"   ⟳ '{o}' is not where I expected "
                                 f"({exp[0]:.0f},{exp[1]:.0f}) → seen at ({scene[o][0]:.0f},{scene[o][1]:.0f}); re-planning")

            try:
                steps = plan(self.goal, scene, self.workspace)
            except GoalInfeasible as e:
                self.log(f"✘ can't complete the task: {e}")
                return False

            if not steps:
                self.log("✔ goal satisfied — everything is where the demonstration implied")
                return True

            step = steps[0]
            self.log(f"[{it + 1}] move '{step.obj}'  ({step.pick_xy[0]:.1f},{step.pick_xy[1]:.1f}) → "
                     f"({step.place_xy[0]:.1f},{step.place_xy[1]:.1f})   ({step.reason}; {len(steps)} step(s) left)")
            try:
                ok = self.perform(step)
            except Unreachable as e:
                self.log(f"✘ unreachable: {e}")
                return False

            expected = {k: v.copy() for k, v in scene.items()}
            if ok:
                expected[step.obj] = step.place_xy.copy()
            if on_step:
                on_step(it, step, ok)

        self.log("✘ gave up: too many re-plans")
        return False
