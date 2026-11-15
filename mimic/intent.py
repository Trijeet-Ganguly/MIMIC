"""Stage 3 - UNDERSTAND: infer *why* the human did what they did.

Instead of storing hand coordinates, we compare the scene BEFORE and AFTER the demo and ask:
"what relationship between objects did the human create?"  The answer (a Goal) is expressed as
object-to-object relations, so it still works when the objects start somewhere else.

Hypotheses tried, in order of specificity:
  1. ROW       - moved objects (plus fixed ones) form an evenly spaced line  -> "line the tools up"
  2. RELATIVE  - each moved object sits at a consistent offset from a neighbour -> "put A right of B"
  3. ABSOLUTE  - fallback: the object simply belongs at that spot on the table
Things deliberately ignored: hand path, hesitations, corrected mistakes, sloppy spacing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .models import Action, Goal, Placement, Scene, TABLE


# ---------------------------------------------------------------- symbolic relations
def relations(vec: np.ndarray, align_tol: float = 1.5, near: float = 12.0) -> List[str]:
    """Symbolic relations of object A relative to B, where vec = pos(A) - pos(B)."""
    r = []
    r.append("right_of" if vec[0] > align_tol else "left_of" if vec[0] < -align_tol else "aligned_x")
    r.append("behind" if vec[1] > align_tol else "in_front_of" if vec[1] < -align_tol else "aligned_y")
    if np.linalg.norm(vec) < near:
        r.append("near")
    return r


def _dir_word(v: np.ndarray) -> str:
    if abs(v[0]) >= abs(v[1]):
        return "right of" if v[0] > 0 else "left of"
    return "behind" if v[1] > 0 else "in front of"


# ---------------------------------------------------------------- row hypothesis
def _find_row(final: Scene, moved: Set[str], tol: float) -> Optional[List[str]]:
    """Largest set (>=3 objects) of near-collinear objects that contains every moved object."""
    names = list(final)
    best = None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a not in moved and b not in moved:
                continue
            p = final[a]
            d = final[b] - p
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            d = d / n
            dev = {c: abs(d[0] * (final[c] - p)[1] - d[1] * (final[c] - p)[0]) for c in names}
            inl = [c for c in names if dev[c] <= tol]
            if len(inl) < 3 or not moved <= set(inl):
                continue
            key = (len(inl), -sum(dev[c] for c in inl))
            if best is None or key > best[0]:
                best = (key, inl)
    return best[1] if best else None


def _row_goal(final: Scene, moved_order: List[str], members: List[str]):
    moved = set(moved_order)
    pts = np.array([final[n] for n in members])
    c = pts.mean(axis=0)
    d = np.linalg.svd(pts - c)[2][0]
    if abs(d[0]) >= abs(d[1]):
        d = d if d[0] > 0 else -d
    else:
        d = d if d[1] > 0 else -d

    t = {n: float((final[n] - c) @ d) for n in members}
    order = sorted(members, key=t.get)
    ts = np.array([t[n] for n in order])
    diffs = np.diff(ts)
    spacing = float(np.median(diffs))
    even = bool(np.all(np.abs(diffs - spacing) <= 0.4 * spacing))
    # Regularise: a sloppy-but-intended even row becomes perfectly even.
    pos = {n: k * spacing for k, n in enumerate(order)} if even else {n: t[n] - ts[0] for n in order}

    stationary = [n for n in order if n not in moved]
    mid = (len(order) - 1) / 2
    placements: List[Placement] = []
    axis = "x (left→right)" if abs(d[0]) >= abs(d[1]) else "y (near→far)"

    if stationary:
        anchor = min(stationary, key=lambda n: abs(order.index(n) - mid))
        for m in moved_order:
            off = (pos[m] - pos[anchor]) * d
            why = (f"{m} → {np.linalg.norm(off):.1f} cm {_dir_word(off)} '{anchor}' "
                   f"(slot {order.index(m) + 1} of {len(order)} in the row)")
            placements.append(Placement(m, anchor, off, why))
        anchor_txt = f"'{anchor}' stayed put, so it is the fixed reference"
    else:
        first = moved_order[0]
        centre_t = float(np.mean(list(pos.values())))
        abs_first = c + (pos[first] - centre_t) * d
        placements.append(Placement(first, TABLE, abs_first,
                                    f"{first} → starts the row at table position ({abs_first[0]:.1f}, {abs_first[1]:.1f})"))
        for m in moved_order[1:]:
            off = (pos[m] - pos[first]) * d
            placements.append(Placement(m, first, off,
                                        f"{m} → {np.linalg.norm(off):.1f} cm {_dir_word(off)} '{first}' "
                                        f"(slot {order.index(m) + 1} of {len(order)})"))
        anchor_txt = f"'{first}' was placed first, so the others are positioned relative to it"

    sp = f"{spacing:.1f} cm apart" if even else "irregular spacing kept"
    hyp = (f"Line the objects up in a row along {axis}, order: {' → '.join(order)} ({sp}). {anchor_txt}.")
    return hyp, (0.9 if even else 0.75), placements


# ---------------------------------------------------------------- main entry point
def infer_goal(initial: Scene, final: Scene, actions: Sequence[Action],
               tol: float = 2.5, move_eps: float = 2.0, far: float = 30.0) -> Goal:
    names = [n for n in final if n in initial]
    moved_set = {n for n in names if np.linalg.norm(final[n] - initial[n]) > move_eps}

    last_t: Dict[str, float] = {}
    count: Dict[str, int] = {}
    for a in actions:
        last_t[a.obj] = a.t_end
        count[a.obj] = count.get(a.obj, 0) + 1
    moved_order = sorted(moved_set, key=lambda n: last_t.get(n, float("inf")))

    notes: List[str] = []
    for n, c in count.items():
        if n not in moved_set:
            notes.append(f"'{n}' was picked up but ended where it started → treated as a no-op.")
        elif c > 1:
            notes.append(f"'{n}' was moved {c}× → earlier placements read as corrections; only the final one is the goal.")
    wp = sum(a.waypoints for a in actions)
    notes.append(f"Discarded {wp} hand-trajectory samples: only the effect on the scene is learned, not the path.")

    if not moved_order:
        return Goal("Nothing meaningful changed in the demo.", 0.0, [], notes)

    # 1) ROW ---------------------------------------------------------------
    members = _find_row(final, moved_set, tol)
    if members:
        hyp, conf, placements = _row_goal(final, moved_order, members)
        return Goal(hyp, conf, placements, [p.why for p in placements] + notes)

    # 2) RELATIVE / 3) ABSOLUTE ---------------------------------------------
    stationary = set(names) - moved_set
    placements, placed = [], set()
    for obj in moved_order:
        cands = [o for o in names if o != obj and (o in stationary or o in placed)]
        anchor = min(cands, key=lambda o: np.linalg.norm(final[o] - final[obj])) if cands else None
        if anchor is None or np.linalg.norm(final[anchor] - final[obj]) > far:
            placements.append(Placement(obj, TABLE, final[obj].copy(),
                                        f"{obj} → belongs at table position ({final[obj][0]:.1f}, {final[obj][1]:.1f})"))
        else:
            off = final[obj] - final[anchor]
            off = np.where(np.abs(off) < 1.5, 0.0, off)  # tiny deviations = "aligned" was intended
            before = relations(initial[obj] - initial[anchor]) if anchor in initial else []
            after = relations(off)
            why = (f"{obj} → {np.linalg.norm(off):.1f} cm {_dir_word(off)} '{anchor}'; "
                   f"relation {'+'.join(before)} ⇒ {'+'.join(after)}")
            placements.append(Placement(obj, anchor, off, why))
        placed.add(obj)

    anchors = {p.anchor for p in placements if p.anchor != TABLE}
    if len(anchors) == 1:
        hyp = f"Place {', '.join(moved_order)} at fixed offsets around '{next(iter(anchors))}'."
    elif anchors:
        hyp = "Place each moved object at a fixed offset from its nearest neighbour."
    else:
        hyp = "Put each moved object at its own spot on the table."
    conf = 0.7 if anchors else 0.5
    return Goal(hyp, conf, placements, [p.why for p in placements] + notes)
