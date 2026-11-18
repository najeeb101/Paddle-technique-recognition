"""Per-frame stroke features (design.md section 4): (T,17,3) keypoints -> (T, F) feature matrix.

Left-handed strokes are mirrored (x -> -x, left and right joints swapped) so every stroke is analysed as
right-handed and one reference serves both. Missing keypoints (conf <= 0 or NaN) give NaN features.
"""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

FEATURES = (
    "elbow_angle", "shoulder_angle", "knee_angle_dom", "knee_angle_off",
    "shoulder_rotation", "hip_rotation", "separation", "torso_lean",
    "wrist_height", "wrist_speed",
)

# COCO-17 index of each joint after a left/right swap.
_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def as_right_handed(kp: np.ndarray, hand: str) -> np.ndarray:
    """Return (T,17,2) x, y with missing points as NaN; a left-handed stroke is mirrored and its sides swapped."""
    kp = np.asarray(kp, dtype=float)
    xy = kp[..., :2].copy()
    xy[~(kp[..., 2] > 0)] = np.nan
    if hand == "left":
        xy = xy[:, _SWAP]
        xy[..., 0] *= -1
    elif hand != "right":
        raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
    return xy


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b between b->a and b->c, in degrees, for (T,2) point arrays."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Image-plane angle of the vector a->b in degrees, counter-clockwise from image right (y points down)."""
    d = b - a
    return np.degrees(np.arctan2(-d[:, 1], d[:, 0]))


def wrap(deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def unwrap(deg: np.ndarray) -> np.ndarray:
    """Remove 360-degree jumps along time, skipping NaN frames."""
    out = deg.copy()
    ok = np.isfinite(deg)
    out[ok] = np.degrees(np.unwrap(np.radians(deg[ok])))
    return out


def stroke_features(kp: np.ndarray, hand: str, fps: float) -> np.ndarray:
    """(T,17,3) keypoints of one player -> (T, len(FEATURES)) features, columns in FEATURES order."""
    p = as_right_handed(kp, hand)
    mid_sh = (p[:, L_SH] + p[:, R_SH]) / 2
    mid_hip = (p[:, L_HIP] + p[:, R_HIP]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)

    sh_rot = unwrap(direction(p[:, L_SH], p[:, R_SH]))
    hip_rot = unwrap(direction(p[:, L_HIP], p[:, R_HIP]))
    lean_vec = mid_sh - mid_hip
    speed = np.full(len(p), np.nan)
    speed[1:] = np.linalg.norm(np.diff(p[:, R_WR], axis=0), axis=1) / torso[1:] * fps

    cols = {
        "elbow_angle": joint_angle(p[:, R_SH], p[:, R_EL], p[:, R_WR]),
        "shoulder_angle": joint_angle(p[:, R_HIP], p[:, R_SH], p[:, R_EL]),
        "knee_angle_dom": joint_angle(p[:, R_HIP], p[:, R_KN], p[:, R_ANK]),
        "knee_angle_off": joint_angle(p[:, L_HIP], p[:, L_KN], p[:, L_ANK]),
        "shoulder_rotation": sh_rot,
        "hip_rotation": hip_rot,
        "separation": wrap(sh_rot - hip_rot),
        "torso_lean": np.degrees(np.arctan2(lean_vec[:, 0], -lean_vec[:, 1])),
        "wrist_height": (p[:, R_SH, 1] - p[:, R_WR, 1]) / torso,
        "wrist_speed": speed,
    }
    return np.stack([cols[f] for f in FEATURES], axis=1)
