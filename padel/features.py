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


def swap_lr(kp: np.ndarray, joints: tuple[int, ...] | None = None) -> np.ndarray:
    """Swap left/right keypoint pairs of a (...,17,C) array; `joints` limits it to pairs with both indices in it."""
    kp = np.asarray(kp)
    if joints is None:
        return kp[..., _SWAP, :]
    idx = [j if (j not in joints or _SWAP[j] not in joints) else _SWAP[j] for j in range(17)]
    return kp[..., idx, :]


def fix_lr_flips(kp: np.ndarray, anchor_weight: float = 0.5, pair: tuple[int, ...] = (5, 6, 11, 12, 13, 14, 15, 16),
                 anchor: tuple[int, int] = (5, 6), swap_joints: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Undo left/right label swaps that the pose model makes mid-stroke.

    Two-state Viterbi over frames: state 1 means "this frame's sides are swapped". A frame must move less (in
    torso lengths, over the paired joints) from the last valid frame under its chosen labelling. A weak
    per-frame prior prefers the shoulder line pointing image-right, which is what a player seen from behind
    (near side) shows. Returns (fixed kp, bool mask of swapped frames).
    """
    kp = np.asarray(kp, dtype=float)
    T = len(kp)
    pair = list(pair)
    xy = kp[..., :2].copy()
    xy[~(kp[..., 2] > 0)] = np.nan
    torso = np.nanmedian(np.linalg.norm((xy[:, 5] + xy[:, 6]) / 2 - (xy[:, 11] + xy[:, 12]) / 2, axis=1)) if T else np.nan
    if not np.isfinite(torso) or torso <= 0:
        return kp.copy(), np.zeros(T, bool)

    def dist(a, b):
        d = np.linalg.norm(a[pair] - b[pair], axis=1)
        d = d[np.isfinite(d)]
        return d.mean() / torso if len(d) else 0.0

    unary = np.zeros((T, 2))
    for t in range(T):
        sh = xy[t, list(anchor)]
        if np.isfinite(sh).all():
            for s in (0, 1):
                a, b = (sh[0], sh[1]) if s == 0 else (sh[1], sh[0])
                unary[t, s] = anchor_weight * (1 - np.cos(np.arctan2(-(b - a)[1], (b - a)[0])))

    cost = np.zeros((T, 2))
    back = np.zeros((T, 2), int)
    cost[0] = unary[0]
    prev_valid = 0 if np.isfinite(xy[0, pair]).any() else None
    for t in range(1, T):
        cur = xy[t]
        has = np.isfinite(cur[pair]).any()
        for s in (0, 1):
            best = None
            for r in (0, 1):
                trans = 0.0
                if prev_valid is not None and has:
                    trans = dist(swap_lr(cur, swap_joints) if s else cur, swap_lr(xy[prev_valid], swap_joints) if r else xy[prev_valid])
                c = cost[t - 1, r] + trans
                if best is None or c < best[0]:
                    best = (c, r)
            cost[t, s] = best[0] + unary[t, s]
            back[t, s] = best[1]
        if has:
            prev_valid = t
    state = np.zeros(T, int)
    state[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    out = kp.copy()
    out[state == 1] = swap_lr(kp[state == 1], swap_joints)
    return out, state == 1


def fix_lr_all(kp: np.ndarray) -> np.ndarray:
    """Global left/right fix, then a lower-body pass: the model flips hips and legs independently of the shoulders."""
    kp, _ = fix_lr_flips(kp)
    lower = (11, 12, 13, 14, 15, 16)
    kp, _ = fix_lr_flips(kp, pair=lower, anchor=(11, 12), swap_joints=lower)
    return kp


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
