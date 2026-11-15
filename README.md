# MIMIC – Robot that learns by watching once

Show a task one time. MIMIC records it, works out **why** you did it, and rebuilds it with a robot arm – even if the objects start somewhere else, or something moves halfway through.

## Try it now (no camera, no robot, only numpy)

```bash
pip install numpy
python run_sim.py --plot      # learn from one synthetic demo → replay in a new layout with disturbances
```

You will see the robot (1) segment the demo, (2) print what it understood, (3) re-create it in a *different* layout while a "human" shoves objects and a grasp slips.

## How it works

```
camera ─► perception.py ─► watcher.py ─► intent.py ─► planner.py ─► executor.py ─► arm.py
(hand + tools)  (frames)   (pick/place)  (the WHY)    (steps now)   (closed loop)  (IK → servos)
                                                          ▲                │
                                                          └──── re-observe after every step
```

**The "why" (intent.py).** MIMIC never stores hand coordinates. It compares the scene *before* and *after* the demo and asks what relationship the human created:

| Hypothesis | Example learned goal | Why it generalises |
|---|---|---|
| **Row** | "hammer, wrench, **tape**, pliers, screwdriver in an evenly spaced line; tape is the fixed reference" | targets are recomputed around wherever the tape is *now* |
| **Relative** | "wrench 6 cm right of pliers" | follows the anchor object |
| **Absolute** | "tape belongs at (18, 18)" | fallback when no relation is found |

It also deliberately ignores: the hand path, hesitations, **mistakes the human corrected** (an object moved twice → only the last placement counts), and sloppy spacing (a roughly even row becomes exactly even).

**Adapting (executor.py).** After every single step the robot looks again and re-plans from what it sees, so a moved object, a knocked-off object, or a failed grasp just produces a different next step. The planner also resolves conflicts (a target occupied by another tool → move that one first, or park it).

## Files

| File | Role |
|---|---|
| `mimic/perception.py` | webcam → table cm: calibration, ArUco tool tracking, MediaPipe hand + pinch |
| `mimic/watcher.py` | frames → pick/place actions |
| `mimic/intent.py` | actions + before/after scene → **Goal** (the why) |
| `mimic/planner.py` | Goal + current scene → ordered steps, conflict handling |
| `mimic/arm.py` | IK/FK, `SimArm`, `SerialArm` (Arduino) |
| `mimic/executor.py` | closed-loop execute / verify / re-plan |
| `mimic/synth.py` | synthetic human demos for testing |
| `mimic/llm_intent.py` | optional: Claude phrases the intent in plain language |
| `arduino/mimic_arm/mimic_arm.ino` | servo firmware |
| `run_sim.py`, `run_live.py`, `tests/` | demos and tests |

## Real hardware

```bash
pip install -r requirements.txt
python run_live.py calibrate --cam 0 --width 36 --height 24   # click 4 table corners
python run_live.py learn --cam 0                               # R = start/stop, do the task once
python run_live.py replay --cam 0 --arm serial --port COM3
```

Setup notes:
- **Camera:** mount overhead, fixed. Table coordinates: origin = front-left corner (nearest the robot), x → robot's right, y → away from robot.
- **Tools:** print ArUco markers (DICT_4X4_50) and stick one on top of each tool; set ids→names in `OBJECTS` in `run_live.py`.
- **Demo etiquette:** start with hand out of view and all tools visible; end with hand out of view.
- **Arm:** upload the `.ino`, wire servos to pins 3/5/6/9/10, set your link lengths in `ArmGeometry` (`arm.py`), then **calibrate `SerialArm.SERVO_MAP`** (servo = offset + sign × angle) with the arm moving slowly and no tools on the table. Test IK first: `move_to` a point, measure where the gripper lands.
- Workspace size is `DEFAULT_WORKSPACE` in `planner.py` (36 × 24 cm); change it with your table.

## Known limits / next steps

- Grabs are detected by *pinch*; for big tools use a fist/palm heuristic or a gripper-force sensor to confirm real grasps (`SerialArm.set_gripper` currently assumes success).
- Intent hypotheses cover rows and relative placement. Easy extensions: sort by size/colour, stack (needs z), grid/zones, "put each tool in its outline". Add a new hypothesis function in `intent.py` and try it before the fallbacks.
- Orientation of tools is not modelled yet (position only).
- For open-ended intents, `llm_intent.explain_with_llm` can cross-check the symbolic reading.
- Camera-in-the-loop SIM replay (`--arm sim` in `run_live.py`) is only a plan/IK dry run; true closed-loop needs the real arm.
