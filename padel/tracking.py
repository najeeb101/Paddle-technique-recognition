"""Track the four players across frames by hip-centre distance (the pose files have no track IDs)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

L_HIP, R_HIP = 11, 12


def hip_centre(kp: np.ndarray) -> np.ndarray:
    """(17,3) keypoints -> (x, y) hip centre, falling back to the bbox of all points if hips are missing."""
    h = kp[[L_HIP, R_HIP]]
    ok = h[:, 2] > 0
    pts = h[ok] if ok.any() else kp[kp[:, 2] > 0]
    return pts[:, :2].mean(axis=0)


def track_players(poses: dict[int, np.ndarray], n_players: int = 4, gate: float = 120.0, gate_growth: float = 30.0):
    """Return {frame: {track_id: (17,3) keypoints}} for frames in ascending order.

    Tracks are created from the first frames and never deleted (the same four players stay on court).
    A detection matches a track if it lies within `gate + gate_growth * frames_since_seen` pixels.
    """
    tracks: list[dict] = []  # each: {"pos": (2,), "seen": frame}
    out: dict[int, dict] = {}
    for f in sorted(poses):
        dets = poses[f]
        centres = np.array([hip_centre(k) for k in dets])
        assigned: dict[int, np.ndarray] = {}
        used: set[int] = set()
        if tracks:
            cost = np.full((len(tracks), len(dets)), 1e9)
            for i, t in enumerate(tracks):
                limit = gate + gate_growth * (f - t["seen"] - 1)
                d = np.linalg.norm(centres - t["pos"], axis=1)
                cost[i] = np.where(d <= limit, d, 1e9)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e9:
                    assigned[i] = dets[j]
                    used.add(j)
                    tracks[i].update(pos=centres[j], seen=f)
        for j in range(len(dets)):  # unmatched detections start new tracks while we have fewer than n_players
            if j not in used and len(tracks) < n_players:
                tracks.append({"pos": centres[j], "seen": f})
                assigned[len(tracks) - 1] = dets[j]
        out[f] = assigned
    return out
