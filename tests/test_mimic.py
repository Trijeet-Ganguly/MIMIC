import numpy as np
import pytest

from mimic import (Executor, SimArm, ArmGeometry, ik, fk, Unreachable, segment, infer_goal,
                   initial_scene, final_scene, plan, GoalInfeasible)
from mimic.synth import DemoBuilder


def test_ik_fk_roundtrip():
    g = ArmGeometry()
    for xy in [(18, 12), (6, 20), (30, 5), (18, 2)]:
        for z in (1.0, 9.0):
            a = ik(g, xy, z)
            x, y, zz = fk(g, a)
            assert np.allclose([x, y, zz], [xy[0], xy[1], z], atol=1e-6)


def test_unreachable():
    with pytest.raises(Unreachable):
        ik(ArmGeometry(), (18, 80), 1.0)


def _demo(seed=3):
    start = {"a": (18, 18), "b": (30, 5), "c": (6, 8), "d": (10, 21)}
    b = DemoBuilder(start, seed=seed)
    b.pick_place("b", (30.2, 18))
    b.pick_place("c", (12, 18))
    b.pick_place("d", (24, 18))
    return b.finish()


def test_learns_row_not_coordinates():
    fr = _demo()
    goal = infer_goal(initial_scene(fr), final_scene(fr), segment(fr))
    assert "row" in goal.hypothesis
    assert all(p.anchor == "a" for p in goal.placements)  # relative to the fixed object, not the table


def test_generalises_to_new_layout():
    fr = _demo()
    goal = infer_goal(initial_scene(fr), final_scene(fr), segment(fr))
    arm = SimArm({"a": (20, 10), "b": (4, 4), "c": (30, 20), "d": (8, 18)}, observe_noise=0.0)
    assert Executor(arm, arm.observe, goal, log=lambda *_: None).run()
    ys = [arm.world[k][1] for k in "bcd"]
    assert np.allclose(ys, arm.world["a"][1], atol=1.5)  # row formed around the anchor's NEW position


def test_infeasible_goal_reported():
    fr = _demo()
    goal = infer_goal(initial_scene(fr), final_scene(fr), segment(fr))
    with pytest.raises(GoalInfeasible):
        plan(goal, {"a": (34, 12), "b": (4, 4), "c": (30, 20), "d": (8, 18)})
