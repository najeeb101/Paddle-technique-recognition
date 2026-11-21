"""Scoring a stroke against a reference (design.md section 7.3).

The score is similarity to professional technique, not a verified measure of correctness.

Each feature gets a similarity in [0, 1]: the contact-weighted mean over steps of exp(-0.5 (z / SOFTNESS)^2),
so any shift away from the reference mean lowers it a little (the old rule only counted steps past |z| = 2).
Timing is one more term: when the trunk turns fastest relative to contact (turn_lead), against the professionals'
own spread. DTW cannot measure this against the averaged reference, whose curves are too blurred to pin a shift.
Timing is off by default: on validation it did not help (docs/results.md section 6b).
The overall score mixes the mean of all terms with the mean of the WORST_K lowest, so one clearly wrong feature
is not hidden by eight normal ones. SOFTNESS and WORST_MIX were chosen on a validation split of
training strokes only (scripts/tune_scoring.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference import N_STEPS, align_to, fill_nan, resample

# wrist_speed is left out by default: contact is defined from the wrist speed peak, so it would be circular.
DEFAULT_FEATURES = (
    "elbow_angle", "shoulder_angle", "knee_angle_dom", "knee_angle_off",
    "shoulder_rotation", "hip_rotation", "separation", "torso_lean", "wrist_height",
)
CONTACT_WINDOW = (30, 50)
SOFTNESS = 2.0
WORST_MIX = 0.5
WORST_K = 2
TIMING = "timing"


@dataclass
class Deviation:
    feature: str
    phase: str       # "before", "at" or "after" contact
    direction: str   # "higher" or "lower" than the reference mean ("later" or "earlier" for timing)
    size: float      # in the feature's own unit (seconds for timing)
    z: float


@dataclass
class Result:
    score: float                       # 0-100 similarity
    feature_scores: dict[str, float]   # 0-1 similarity per feature (and timing when the reference has it)
    deviations: list[Deviation]        # the three lowest-scoring terms
    coverage: float                    # share of feature values that were measured (not gap-filled)


def load_reference(path) -> dict:
    d = np.load(path)
    ref = {"mean": d["mean"], "std": d["std"], "scale": d["scale"], "names": [str(n) for n in d["feature_names"]]}
    if "timing_std" in d:
        ref["timing_mean"], ref["timing_std"] = float(d["timing_mean"]), float(d["timing_std"])
    return ref


def step_weights(n: int = N_STEPS, window: tuple[int, int] = CONTACT_WINDOW) -> np.ndarray:
    w = np.ones(n)
    w[window[0]:window[1] + 1] = 2.0
    return w


def phase_of(step: int, window: tuple[int, int] = CONTACT_WINDOW) -> str:
    return "before" if step < window[0] else "after" if step > window[1] else "at"


def turn_lead(feat: np.ndarray, contact: int, names, fps: float = 30.0) -> float:
    """Seconds from the peak shoulder-turn speed to contact (negative = the turn peaks before contact).

    The peak is searched from 0.4 s before contact to 0.13 s after; NaN if the shoulder line is mostly missing.
    """
    x = np.asarray(feat, dtype=float)[:, list(names).index("shoulder_rotation")]
    if len(x) < 3 or np.isnan(x).mean() > 0.3:
        return float("nan")
    x = np.unwrap(np.radians(fill_nan(x[:, None])[:, 0]))  # the shoulder line angle can wrap at +-180
    x = np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    v = np.abs(np.gradient(x))
    lo, hi = max(1, contact - round(0.4 * fps)), min(len(x) - 1, contact + round(0.13 * fps))
    if hi <= lo:
        return float("nan")
    return (lo + int(np.argmax(v[lo:hi])) - contact) / fps


def timing_stats(strokes, names, fps: float = 30.0) -> dict:
    """{"timing_mean", "timing_std"} of turn_lead over (feat, contact) pairs, to add to a reference."""
    t = np.array([turn_lead(f, c, names, fps) for f, c in strokes])
    t = t[np.isfinite(t)]
    return {"timing_mean": float(t.mean()), "timing_std": float(max(t.std(), 1.0 / fps))}


def similarity(z: np.ndarray, softness: float = SOFTNESS) -> np.ndarray:
    return np.exp(-0.5 * (np.asarray(z) / softness) ** 2)


def combine(values, worst_mix: float = WORST_MIX, worst_k: int = WORST_K) -> float:
    v = np.sort(np.asarray(list(values), dtype=float))
    return float((1 - worst_mix) * v.mean() + worst_mix * v[:worst_k].mean())


def score_stroke(feat: np.ndarray, contact: int, ref: dict, features: tuple[str, ...] = DEFAULT_FEATURES,
                 softness: float = SOFTNESS, worst_mix: float = WORST_MIX, use_timing: bool = False,
                 fps: float = 30.0) -> Result:
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
    per = {}
    for k, c in enumerate(use):
        ok = np.isfinite(z[:, k])
        if ok.any():
            per[names[c]] = float((w[ok] * similarity(z[ok, k], softness)).sum() / w[ok].sum())
    if not per:
        return Result(float("nan"), {}, [], coverage)
    lead = turn_lead(feat, contact, names, fps) if use_timing and "timing_std" in ref else float("nan")
    if np.isfinite(lead):
        zt = (lead - ref["timing_mean"]) / ref["timing_std"]
        per[TIMING] = float(similarity(zt, softness))
    worst = sorted(per, key=per.get)[:3]
    devs = []
    for f in worst:
        if f == TIMING:
            devs.append(Deviation(TIMING, "before", "later" if zt > 0 else "earlier",
                                  abs(lead - ref["timing_mean"]), float(zt)))
            continue
        k = use.index(names.index(f))
        step = int(np.nanargmax(np.abs(z[:, k])))
        diff = float(aligned[step, k] - ref["mean"][step, use[k]])
        devs.append(Deviation(f, phase_of(step), "higher" if diff > 0 else "lower", abs(diff), float(z[step, k])))
    return Result(100.0 * combine(per.values(), worst_mix), per, devs, coverage)
