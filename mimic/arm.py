"""Stage 5 - ACT: arm kinematics + drivers (simulation and real Arduino servo arm).

Arm model: 4-DOF (base yaw, shoulder, elbow, wrist) + gripper, gripper always pointing straight down.
Change ArmGeometry to match your arm's link lengths (cm).
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import Scene


class Unreachable(Exception):
    pass


@dataclass
class ArmGeometry:
    base_xy: Tuple[float, float] = (18.0, -8.0)  # where the arm base sits in table coordinates
    shoulder_h: float = 14.0                      # shoulder joint height above table
    l1: float = 20.0                              # upper arm
    l2: float = 20.0                              # forearm
    l3: float = 10.0                              # wrist-to-fingertip (gripper pointing down)
    min_r: float = 4.0                            # can't reach closer than this to the base
    safe_z: float = 9.0                           # tool height while travelling
    grasp_z: float = 1.0                          # tool height when grabbing
    place_z: float = 1.5                          # tool height when releasing


@dataclass
class JointAngles:
    yaw: float       # deg, 0 = straight ahead (+y), positive = towards +x
    shoulder: float  # deg above horizontal
    elbow: float     # deg relative to upper arm (negative = elbow up)
    wrist: float     # deg relative to forearm (keeps gripper vertical)


def ik(g: ArmGeometry, xy, z: float) -> JointAngles:
    dx, dy = xy[0] - g.base_xy[0], xy[1] - g.base_xy[1]
    r = math.hypot(dx, dy)
    if r < g.min_r:
        raise Unreachable(f"({xy[0]:.1f}, {xy[1]:.1f}) is too close to the base")
    wz = z + g.l3 - g.shoulder_h  # wrist-centre height relative to shoulder
    c = (r * r + wz * wz - g.l1 ** 2 - g.l2 ** 2) / (2 * g.l1 * g.l2)
    if abs(c) > 1:
        raise Unreachable(f"({xy[0]:.1f}, {xy[1]:.1f}, z={z:.1f}) is out of reach")
    e = -math.acos(c)  # elbow-up solution
    phi1 = math.atan2(wz, r) - math.atan2(g.l2 * math.sin(e), g.l1 + g.l2 * math.cos(e))
    w = -math.pi / 2 - phi1 - e
    return JointAngles(math.degrees(math.atan2(dx, dy)), math.degrees(phi1), math.degrees(e), math.degrees(w))


def fk(g: ArmGeometry, a: JointAngles) -> Tuple[float, float, float]:
    """Return fingertip (x, y, z) in table coordinates - used to sanity-check IK."""
    p1, e, w = math.radians(a.shoulder), math.radians(a.elbow), math.radians(a.wrist)
    r = g.l1 * math.cos(p1) + g.l2 * math.cos(p1 + e) + g.l3 * math.cos(p1 + e + w)
    z = g.shoulder_h + g.l1 * math.sin(p1) + g.l2 * math.sin(p1 + e) + g.l3 * math.sin(p1 + e + w)
    yaw = math.radians(a.yaw)
    return g.base_xy[0] + r * math.sin(yaw), g.base_xy[1] + r * math.cos(yaw), z


class Arm(ABC):
    geometry: ArmGeometry

    @abstractmethod
    def move_to(self, xy, z: float) -> None: ...

    @abstractmethod
    def set_gripper(self, closed: bool) -> bool:
        """Close/open gripper. Returns True if (it is believed that) an object is now held."""

    @abstractmethod
    def home(self) -> None: ...


# --------------------------------------------------------------------------- simulation
class SimArm(Arm):
    """Kinematically-checked simulation. Holds the 'world' so you can test everything without hardware."""

    def __init__(self, world: Scene, geometry: Optional[ArmGeometry] = None,
                 place_noise: float = 0.4, observe_noise: float = 0.2, seed: int = 0):
        self.geometry = geometry or ArmGeometry()
        self.world = {k: np.array(v, dtype=float) for k, v in world.items()}
        self.tool_xy = np.array(self.geometry.base_xy, dtype=float) + np.array([0.0, 8.0])
        self.tool_z = self.geometry.safe_z
        self.holding: Optional[str] = None
        self.place_noise, self.observe_noise = place_noise, observe_noise
        self.rng = np.random.default_rng(seed)
        self._fail_grasp = 0
        self.trace: List[Tuple[float, float, float]] = []

    # --- helpers for demos/tests
    def observe(self) -> Scene:
        return {k: v + self.rng.normal(0, self.observe_noise, 2) for k, v in self.world.items()}

    def disturb(self, obj: str, xy) -> None:
        """Simulate a human nudging an object."""
        self.world[obj] = np.array(xy, dtype=float)

    def fail_next_grasp(self, n: int = 1) -> None:
        self._fail_grasp = n

    # --- Arm API
    def move_to(self, xy, z: float) -> None:
        ik(self.geometry, xy, z)  # raises Unreachable if impossible
        self.tool_xy, self.tool_z = np.array(xy, dtype=float), z
        self.trace.append((float(xy[0]), float(xy[1]), float(z)))
        if self.holding:
            self.world[self.holding] = self.tool_xy.copy()

    def set_gripper(self, closed: bool) -> bool:
        if closed:
            if self._fail_grasp > 0:
                self._fail_grasp -= 1
                return False
            if self.tool_z <= 3.0 and self.holding is None:
                name = min(self.world, key=lambda k: np.linalg.norm(self.world[k] - self.tool_xy))
                if np.linalg.norm(self.world[name] - self.tool_xy) <= 2.5:
                    self.holding = name
            return self.holding is not None
        if self.holding:
            self.world[self.holding] = self.tool_xy + self.rng.normal(0, self.place_noise, 2)
            self.holding = None
        return False

    def home(self) -> None:
        self.tool_xy = np.array(self.geometry.base_xy, dtype=float) + np.array([0.0, 8.0])
        self.tool_z = self.geometry.safe_z


# --------------------------------------------------------------------------- real hardware
class SerialArm(Arm):
    """Drives an Arduino running arduino/mimic_arm/mimic_arm.ino over USB serial.

    SERVO_MAP converts IK angles (degrees) to servo angles: servo = offset + sign * angle.
    These defaults are only a starting point - CALIBRATE them for your arm (see README).
    """
    SERVO_MAP: Dict[str, Tuple[float, float]] = {
        "yaw": (90.0, 1.0),
        "shoulder": (0.0, 1.0),
        "elbow": (180.0, 1.0),
        "wrist": (90.0, 1.0),
    }

    def __init__(self, port: str, geometry: Optional[ArmGeometry] = None, baud: int = 115200,
                 home_pose: Tuple[int, int, int, int] = (90, 90, 90, 90)):
        import serial  # pip install pyserial
        self.geometry = geometry or ArmGeometry()
        self.ser = serial.Serial(port, baud, timeout=10)
        self.home_pose = home_pose
        time.sleep(2.0)  # Arduino resets when the port opens
        self.ser.reset_input_buffer()

    def _cmd(self, line: str) -> None:
        self.ser.write((line + "\n").encode())
        reply = self.ser.readline().decode().strip()
        if reply != "OK":
            raise RuntimeError(f"Arduino replied {reply!r} to {line!r}")

    def _servo(self, joint: str, angle: float) -> int:
        off, sign = self.SERVO_MAP[joint]
        return int(round(min(180, max(0, off + sign * angle))))

    def move_to(self, xy, z: float) -> None:
        a = ik(self.geometry, xy, z)
        s = [self._servo(j, getattr(a, j)) for j in ("yaw", "shoulder", "elbow", "wrist")]
        self._cmd("J,%d,%d,%d,%d" % tuple(s))

    def set_gripper(self, closed: bool) -> bool:
        self._cmd("G,%d" % (1 if closed else 0))
        return closed  # no grasp sensing yet: add a current/force/IR sensor and return the real result

    def home(self) -> None:
        self._cmd("J,%d,%d,%d,%d" % self.home_pose)
