"""analyze_clip: phone clip -> swings -> scores and feedback, as the result JSON of design.md section 3.4.

If a required check fails, `failure` holds a code and a message and `strokes` is empty, so the app never
shows a misleading score. `confidence` is pose-based only for now (share of measured values and of frames
with the racket arm seen); distance from the reliable reference range is not part of it yet.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .features import FEATURES, fix_lr_all, stroke_features
from .scoring import Deviation, load_reference, score_stroke
from .swings import detect_swings

MAX_CLIP_S = 20.0
MIN_ARM_FRACTION = 0.5   # below this a swing is not scored (proposal)
WARN_ARM_FRACTION = 0.8  # below this the result carries a warning (proposal)
ARM = {"right": (6, 8, 10), "left": (5, 7, 9)}  # shoulder, elbow, wrist on the racket side

FAILURES = {
    "clip_too_long": "Clip is over 20 seconds. Trim to one or two strokes.",
    "no_person": "No player detected. Step back so your whole body is visible.",
    "no_swing": "No stroke found. Make sure the swing is inside the clip.",
    "low_pose_quality": "Pose tracking was unreliable on this clip. Film from behind with your whole body in view.",
}
PHASE = {"before": "before contact", "at": "at contact", "after": "after contact"}
JOINTS = {"elbow_angle": "Elbow", "knee_angle_dom": "Racket-side knee", "knee_angle_off": "Other knee"}
LABELS = {"shoulder_rotation": "Shoulder turn angle", "hip_rotation": "Hip turn angle",
          "separation": "Shoulder-hip separation", "torso_lean": "Torso lean angle"}


def feedback_text(d: Deviation) -> str:
    """One plain sentence for a deviation, e.g. 'Elbow is about 18 degrees more bent than the professionals at contact.'"""
    when = PHASE[d.phase]
    if d.feature == "timing":
        return f"Your trunk turn peaks about {d.size:.2f} s {d.direction} than the professionals', relative to contact."
    if d.feature == "wrist_height":
        return f"Racket wrist is about {d.size:.1f} torso lengths {d.direction} than the professionals {when}."
    n = round(d.size)
    if d.feature in JOINTS:
        how = "more bent" if d.direction == "lower" else "straighter"
        return f"{JOINTS[d.feature]} is about {n} degrees {how} than the professionals {when}."
    if d.feature == "shoulder_angle":
        how = "closer to the body" if d.direction == "lower" else "further from the body"
        return f"Racket arm is about {n} degrees {how} than the professionals {when}."
    return f"{LABELS.get(d.feature, d.feature)} is about {n} degrees {d.direction} than the professionals {when}."


def empty_result(clip: str, hand: str, stroke_type: str, fps: float, warnings=()) -> dict:
    return dict(clip=clip, hand=hand, stroke_type=stroke_type, fps=float(fps), n_swings_found=0,
                strokes=[], warnings=list(warnings), failure=None)


def failed(out: dict, code: str) -> dict:
    out["failure"] = {"code": code, "message": FAILURES[code]}
    out["strokes"] = []
    return out


def analyze_keypoints(kp: np.ndarray, fps: float, hand: str, stroke_type: str, ref: dict, clip: str = "",
                      warnings=()) -> dict:
    """Cleaned (T,17,3) keypoints of one player -> result dict (design.md section 3.4)."""
    out = empty_result(clip, hand, stroke_type, fps, warnings)

    def fail(code):
        return failed(out, code)

    if len(kp) / fps > MAX_CLIP_S:
        return fail("clip_too_long")
    if not np.isfinite(kp[:, :, 0]).any():
        return fail("no_person")
    kp = fix_lr_all(kp)
    feat = stroke_features(kp, hand, fps)
    swings = detect_swings(feat[:, FEATURES.index("wrist_speed")], fps)
    out["n_swings_found"] = len(swings)
    if not swings:
        return fail("no_swing")
    arm_seen = (kp[:, list(ARM[hand]), 2] > 0).all(axis=1)
    for start, contact, end in swings:
        frac = float(arm_seen[start:end + 1].mean())
        if frac < MIN_ARM_FRACTION:
            continue
        r = score_stroke(feat[start:end + 1], contact - start, ref)
        if not np.isfinite(r.score):
            continue
        if frac < WARN_ARM_FRACTION:
            out["warnings"].append(f"Pose tracking was unreliable on the swing at frame {contact} "
                                   f"({100 * frac:.0f}% of frames). Result may be wrong.")
        out["strokes"].append(dict(
            start_frame=int(start), contact_frame=int(contact), end_frame=int(end),
            score=round(r.score, 1),
            confidence=round(min(r.coverage, frac), 2),
            feature_scores={k: round(v, 2) for k, v in r.feature_scores.items()},
            feedback=[dict(feature=d.feature, phase=d.phase, direction=d.direction, size=round(d.size, 2),
                           unit={"wrist_height": "torso lengths", "timing": "seconds"}.get(d.feature, "degrees"),
                           text=feedback_text(d)) for d in r.deviations],
            pose_quality=dict(frames=int(end - start + 1), confident_fraction=round(frac, 2)),
        ))
    if not out["strokes"]:
        return fail("low_pose_quality")
    return out


def clip_seconds(video: str | Path) -> float:
    """Length of a video in seconds from its header (0 if unknown)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps if fps > 0 else 0.0


def analyze_clip(video: str | Path, hand: str, stroke_type: str, weights: str | Path,
                 ref_dir: str | Path = Path(__file__).resolve().parents[1] / "reference", progress=print, pose=None):
    """Full phone-clip analysis. Returns (result dict, cleaned keypoints or None).

    `progress` gets stage names; `pose(video, weights)` defaults to `padel.pose.clip_keypoints` (tests pass a fake).
    The result's `timings_s` holds seconds per stage, for the latency measurement.
    """
    if pose is None:
        from .pose import clip_keypoints as pose

    ref = load_reference(Path(ref_dir) / f"{stroke_type}.npz")
    if clip_seconds(video) > MAX_CLIP_S:
        return failed(empty_result(Path(video).name, hand, stroke_type, 0.0), "clip_too_long"), None
    progress("pose")
    t0 = time.perf_counter()
    p = pose(video, weights)
    t1 = time.perf_counter()
    progress("analysis")
    result = analyze_keypoints(p["keypoints"], p["fps"], hand, stroke_type, ref, clip=Path(video).name,
                               warnings=p["warnings"])
    result["timings_s"] = {"pose": round(t1 - t0, 2), "analysis": round(time.perf_counter() - t1, 2)}
    return result, p["keypoints"]
