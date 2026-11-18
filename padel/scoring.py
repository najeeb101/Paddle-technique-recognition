"""Scoring a stroke against a reference (design.md section 7.3).

The score is similarity to professional technique, not a verified measure of correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import CONTACT_STEP, N_STEPS, align_to, resample

# wrist_speed is left out by default: contact is defined from the wrist speed peak, so it would be circular.
DEFAULT_FEATURES = (
    "elbow_angle", "shoulder_angle", "knee_angle_dom", "knee_angle_off",
    "shoulder_rotation", "hip_rotation", "separation", "torso_lean", "wrist_height",
)
CONTACT_WINDOW = (30, 50)
Z_LIMIT = 2.0


@dataclass
class Deviation:
    feature: str
    phase: str       # "before", "at" or "after" contact
    direction: str   # "higher" or "lower" than the reference mean
    size: float      # in the feature's own unit
    z: float


@dataclass
class Result:
    score: float                       # 0-100 similarity
    feature_scores: dict[str, float]   # share of weighted steps with |z| <= 2
    deviations: list[Deviation]        # the three lowest-scoring features
    coverage: float                    # share of feature values that were measured (not gap-filled)


def load_reference(path) -> dict:
    d = np.load(path)
    return {"mean": d["mean"], "std": d["std"], "scale": d["scale"], "names": [str(n) for n in d["feature_names"]]}


def step_weights(n: int = N_STEPS, window: tuple[int, int] = CONTACT_WINDOW) -> np.ndarray:
    w = np.ones(n)
    w[window[0]:window[1] + 1] = 2.0
    return w


def phase_of(step: int, window: tuple[int, int] = CONTACT_WINDOW) -> str:
    return "before" if step < window[0] else "after" if step > window[1] else "at"


def score_stroke(feat: np.ndarray, contact: int, ref: dict, features: tuple[str, ...] = DEFAULT_FEATURES) -> Result:
    """(T,F) features of one stroke with contact at frame index `contact` -> Result against `ref`."""
    names = ref["names"]
    cols = [names.index(f) for f in features]
    coverage = float(np.isfinite(feat[:, cols]).mean())
    seq = resample(feat, contact)
    scale = ref["scale"]
    use = [c for c in cols if np.isfinite(seq[:, c]).all()]
    if not use:
        return Result(float("nan"), {}, [], coverage)
    aligned = align_to(seq[:, use] / scale[use], ref["mean"][:, use] / scale[use]) * scale[use]
    z = (aligned - ref["mean"][:, use]) / ref["std"][:, use]
    w = step_weights(len(z))
    per = {names[c]: float((w * (np.abs(z[:, k]) <= Z_LIMIT)).sum() / w.sum()) for k, c in enumerate(use)}
    worst = sorted(per, key=per.get)[:3]
    devs = []
    for f in worst:
        k = use.index(names.index(f))
        step = int(np.abs(z[:, k]).argmax())
        diff = float(aligned[step, k] - ref["mean"][step, use[k]])
        devs.append(Deviation(f, phase_of(step), "higher" if diff > 0 else "lower", abs(diff), float(z[step, k])))
    return Result(100.0 * float(np.mean(list(per.values()))), per, devs, coverage)
