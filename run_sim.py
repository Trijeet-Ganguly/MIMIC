"""MIMIC end-to-end demo in simulation (needs only numpy; matplotlib optional for --plot).

    python run_sim.py            # learn from one synthetic demo, then replay in a NEW layout with disturbances
    python run_sim.py --plot     # also save mimic_sim.png
"""
import argparse

import numpy as np

from mimic import (Executor, SimArm, final_scene, infer_goal, initial_scene, segment)
from mimic.synth import DemoBuilder

# ---- 1. THE DEMONSTRATION: a human lines tools up in a row around the tape -------------------
DEMO_START = {"tape": (18, 18), "hammer": (30, 5), "wrench": (6, 8), "pliers": (22, 4), "screwdriver": (10, 21)}
ROW_Y, S = 18.0, 6.2
ROW_SLOTS = {"hammer": 18 - 2 * S, "wrench": 18 - S, "pliers": 18 + S, "screwdriver": 18 + 2 * S}


def make_demo():
    b = DemoBuilder(DEMO_START, seed=3)
    b.pick_place("hammer", (ROW_SLOTS["hammer"], ROW_Y))
    b.pick_place("wrench", (14, 9))                          # a mistake...
    b.pick_place("wrench", (ROW_SLOTS["wrench"], ROW_Y))     # ...corrected
    b.pick_place("pliers", (ROW_SLOTS["pliers"], ROW_Y))
    b.pick_place("screwdriver", (ROW_SLOTS["screwdriver"], ROW_Y))
    return b.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print("=" * 72, "\n1) WATCHING one demonstration\n" + "=" * 72)
    frames = make_demo()
    actions = segment(frames)
    print(f"{len(frames)} frames → {len(actions)} raw pick-and-place actions: " +
          ", ".join(a.obj for a in actions))

    print("\n" + "=" * 72, "\n2) UNDERSTANDING why\n" + "=" * 72)
    init, fin = initial_scene(frames), final_scene(frames)
    goal = infer_goal(init, fin, actions)
    print(f"Intent (confidence {goal.confidence:.0%}):\n  {goal.hypothesis}\n")
    for line in goal.explanation:
        print("  •", line)
    goal.save("learned_task.json")

    print("\n" + "=" * 72, "\n3) REPRODUCING in a DIFFERENT layout (tape moved, tools scattered)\n" + "=" * 72)
    new_layout = {"tape": (20, 12), "hammer": (4, 20), "wrench": (12, 3), "pliers": (30, 22), "screwdriver": (6, 11)}
    arm = SimArm(new_layout, seed=7)
    start_world = {k: v.copy() for k, v in arm.world.items()}

    def disturbances(it, step, ok):
        if it == 1:
            print("   👋 (a human shoves 'pliers' to a new spot)")
            arm.disturb("pliers", (28, 4))
        if it == 2:
            print("   ⚙  (next grasp will slip)")
            arm.fail_next_grasp()
        if it == 4 and ok:
            print(f"   👋 (a human knocks '{step.obj}' off its spot)")
            arm.disturb(step.obj, step.place_xy + np.array([-5.0, -4.0]))

    ok = Executor(arm, arm.observe, goal, log=print).run(on_step=disturbances)

    print("\nFinal positions vs. learned targets:")
    from mimic import resolve_targets
    tg = resolve_targets(goal, arm.world)
    for o, t in tg.items():
        e = np.linalg.norm(arm.world[o] - t)
        print(f"  {o:12s} at ({arm.world[o][0]:5.1f},{arm.world[o][1]:5.1f})  target ({t[0]:5.1f},{t[1]:5.1f})  error {e:.2f} cm")
    print("\nRESULT:", "SUCCESS" if ok else "FAILED")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - skipping plot")
            return
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        def draw(a, scene, title, trace=None):
            for n, p in scene.items():
                a.scatter(*p, s=140); a.annotate(n, p, textcoords="offset points", xytext=(0, 9), ha="center")
            if trace:
                xs, ys, _ = zip(*trace); a.plot(xs, ys, lw=0.5, color="gray", alpha=0.6)
            a.set_xlim(-2, 38); a.set_ylim(-2, 26); a.set_aspect("equal"); a.set_title(title); a.grid(alpha=.3)
        draw(ax[0], fin, "Demo result (what the human built)")
        draw(ax[1], start_world, "New layout (before)")
        draw(ax[2], arm.world, "After MIMIC (gray = arm path)", arm.trace)
        fig.tight_layout(); fig.savefig("mimic_sim.png", dpi=130)
        print("saved mimic_sim.png")


if __name__ == "__main__":
    main()
