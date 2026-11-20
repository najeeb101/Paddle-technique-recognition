"""Synthetic faults for the flawed-stroke test (plan day 8, evaluation protocol in project-plan.md section 8).

Each fault changes one thing about a real stroke by a known amount, so we can check that the score drops
and that the feedback names the changed feature. Passing shows the score reacts to the features it uses;
it does not show the score matches a coach's judgement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import wrap


@dataclass(frozen=True)
class Fault:
    feature: str | None  # the feature that is edited (None for timing)
    amount: float        # added to the feature at size 1 (frames for timing)
    text: str


FAULTS = {
    "elbow": Fault("elbow_angle", -20.0, "elbow 20 degrees more bent"),
    "knee": Fault("knee_angle_dom", 20.0, "racket-side knee 20 degrees straighter"),
    "rotation": Fault("shoulder_rotation", -20.0, "shoulder line rotated 20 degrees (separation changed to match)"),
    "wrist_height": Fault("wrist_height", -0.3, "racket wrist 0.3 torso lengths lower"),
    "timing": Fault(None, 3.0, "contact 3 frames (0.1 s) late relative to the swing"),
}


def apply_fault(feat: np.ndarray, contact: int, names, fault: str, size: float = 1.0) -> tuple[np.ndarray, int]:
    """Return a flawed copy of a (T,F) stroke and its contact index. `size` scales the fault (2.0 = twice as big)."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose from {sorted(FAULTS)}")
    f = FAULTS[fault]
    out = np.array(feat, dtype=float)
    names = list(names)
    if f.feature is None:  # the body is late: the contact label sits earlier in the motion
        return out, int(np.clip(contact - round(f.amount * size), 1, len(out) - 2))
    out[:, names.index(f.feature)] += f.amount * size
    if fault == "rotation":
        j = names.index("separation")
        out[:, j] = wrap(out[:, j] + f.amount * size)
    return out, contact


def auc(clean, flawed) -> float:
    """Chance that a random clean stroke scores higher than a random flawed one (ties count half). 0.5 = no separation."""
    a, b = np.asarray(clean, dtype=float)[:, None], np.asarray(flawed, dtype=float)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())
