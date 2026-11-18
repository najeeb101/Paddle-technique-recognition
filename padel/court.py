"""Court coordinates from the PadelTracker100 v2 homography: image pixels -> metres on the court.

The court is 10 m wide (x) and 20 m long (y); the net is at y = 10 m and the camera end is y = 20 m,
so a player with feet at y > 10 m is on the near side (seen from behind).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 15, 16
NET_Y = 10.0


def load_homography(path: str | Path) -> np.ndarray:
    """Return the 3x3 pixel -> metre matrix of a match. It is the same for every frame, which is checked."""
    entries = json.load(open(path))["homography"]
    H = np.asarray(entries[0]["H"], dtype=float)
    if not all(np.allclose(e["H"], H) for e in entries):
        raise ValueError(f"{path}: homography changes between frames")
    return H


def to_court(H: np.ndarray, px) -> np.ndarray:
    """Map pixel points (..., 2) to court metres (..., 2)."""
    p = np.asarray(px, dtype=float)
    q = np.concatenate([p, np.ones(p.shape[:-1] + (1,))], axis=-1) @ H.T
    return q[..., :2] / q[..., 2:]


def feet_point(kp: np.ndarray) -> np.ndarray:
    """(17,3) keypoints -> pixel point on the ground: mean of visible ankles, else of visible hips."""
    for joints in ((L_ANK, R_ANK), (L_HIP, R_HIP)):
        pts = kp[list(joints)]
        ok = pts[:, 2] > 0
        if ok.any():
            return pts[ok, :2].mean(axis=0)
    return np.array([np.nan, np.nan])


def is_near_side(H: np.ndarray, kp: np.ndarray) -> bool:
    """True when the player's feet are on the camera's half of the court."""
    return bool(to_court(H, feet_point(kp))[1] > NET_Y)
