"""MIMIC with a real webcam (and optionally a real Arduino arm).

  python run_live.py calibrate --cam 0 --width 36 --height 24
  python run_live.py learn     --cam 0            # press R to start/stop recording the demo, Q to quit
  python run_live.py replay    --cam 0 --arm sim            # dry run: camera sees, simulated arm acts
  python run_live.py replay    --cam 0 --arm serial --port COM3   # real arm (e.g. /dev/ttyUSB0 on Linux)

Edit OBJECTS below: ArUco id -> tool name (print markers from DICT_4X4_50 and stick one on each tool).
"""
import argparse
import time

import cv2
import numpy as np

from mimic import Executor, Frame, Goal, SerialArm, SimArm, final_scene, infer_goal, initial_scene, segment
from mimic.perception import HandTracker, ObjectTracker, TableCalibration, calibrate_interactive

OBJECTS = {0: "tape", 1: "hammer", 2: "wrench", 3: "pliers", 4: "screwdriver"}
CALIB, TASK = "calib.json", "learned_task.json"


def open_cam(i):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {i}")
    return cap


def cmd_calibrate(a):
    calibrate_interactive(open_cam(a.cam), a.width, a.height, CALIB)
    print("saved", CALIB)


def cmd_learn(a):
    cap, cal = open_cam(a.cam), TableCalibration.load(CALIB)
    objs, hand = ObjectTracker(cal, OBJECTS), HandTracker(cal)
    frames, recording = [], False
    print("Make sure all tools are visible with your hand OUT of view, then press R. Do the task ONCE. "
          "Take your hand away and press R again.")
    while True:
        ok, img = cap.read()
        if not ok:
            continue
        scene = objs.update(img)
        hxy, pinch = hand.update(img)
        if recording:
            frames.append(Frame(time.time(), scene, hxy, pinch))
        cv2.putText(img, "REC" if recording else "idle (R=record, Q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (200, 200, 200), 2)
        cv2.imshow("MIMIC - watching", img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            return
        if k == ord("r"):
            if recording:
                break
            recording = True
    cap.release(); cv2.destroyAllWindows()
    if len(frames) < 10:
        raise SystemExit("demo too short")
    missing = set(OBJECTS.values()) - set(frames[0].objects)
    if missing:
        print("warning: never saw", missing)
    actions = segment(frames)
    goal = infer_goal(initial_scene(frames), final_scene(frames), actions)
    print(f"\nI saw {len(actions)} pick-and-place action(s).\nMy understanding ({goal.confidence:.0%}): {goal.hypothesis}")
    for line in goal.explanation:
        print("  •", line)
    goal.save(TASK)
    print("saved", TASK)


def cmd_replay(a):
    cap, cal = open_cam(a.cam), TableCalibration.load(CALIB)
    objs, goal = ObjectTracker(cal, OBJECTS), Goal.load(TASK)

    def observe():
        for _ in range(5):  # flush stale frames, let tracker settle
            ok, img = cap.read()
        return objs.update(img)

    world = observe()
    arm = SerialArm(a.port) if a.arm == "serial" else SimArm(world)
    # In sim mode the arm moves its own simulated copy of the world, so camera and sim won't agree after step 1;
    # sim mode here is only useful for checking the plan/IK. Use --arm serial for the real closed loop.
    Executor(arm, observe, goal).run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "learn", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--cam", type=int, default=0)
        if name == "calibrate":
            p.add_argument("--width", type=float, default=36.0)
            p.add_argument("--height", type=float, default=24.0)
        if name == "replay":
            p.add_argument("--arm", choices=["sim", "serial"], default="sim")
            p.add_argument("--port", default="COM3")
    args = ap.parse_args()
    {"calibrate": cmd_calibrate, "learn": cmd_learn, "replay": cmd_replay}[args.cmd](args)
