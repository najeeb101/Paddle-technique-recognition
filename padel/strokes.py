"""Per stroke window: infer the hitter, their racket hand and the contact frame from wrist speed."""
from __future__ import annotations

import numpy as np

L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


def _series(tracks, tid, frames):
    """(n,17,3) keypoints of one track over `frames`, NaN where the track is missing."""
    out = np.full((len(frames), 17, 3), np.nan)
    for i, f in enumerate(frames):
        kp = tracks.get(f, {}).get(tid)
        if kp is not None:
            out[i] = kp
    return out


def wrist_speeds(kp: np.ndarray):
    """Per-frame speed of each wrist in torso lengths per frame. Returns (n, 2) for [left, right]."""
    torso = np.linalg.norm(np.nanmean(kp[:, [L_SH, R_SH], :2], 1) - np.nanmean(kp[:, [L_HIP, R_HIP], :2], 1), axis=1)
    t = np.nanmedian(torso) if np.isfinite(torso).any() else np.nan
    sp = np.full((len(kp), 2), np.nan)
    for c, w in enumerate((L_WR, R_WR)):
        d = np.linalg.norm(np.diff(kp[:, w, :2], axis=0), axis=1)
        sp[1:, c] = d / t
    return sp


def analyse_window(tracks, start: int, end: int, pad: int = 2, min_frames: int = 6):
    """Return dict(hitter, hand, peak, contact) or None if no track has enough frames.

    hitter: track whose faster wrist has the highest peak speed; hand: 'left'/'right' = that wrist;
    contact: frame of that wrist's speed peak (a proxy for ball contact, not verified against video).
    """
    frames = list(range(start - pad, end + pad + 1))
    ids = {i for f in frames for i in tracks.get(f, {})}
    best = None
    for tid in ids:
        kp = _series(tracks, tid, frames)
        if np.isfinite(kp[:, 0, 0]).sum() < min_frames:
            continue
        sp = wrist_speeds(kp)
        if not np.isfinite(sp).any():
            continue
        sp = np.nan_to_num(sp, nan=0.0)
        c = int(np.argmax(sp.max(axis=0)))
        peak = float(sp[:, c].max())
        if best is None or peak > best["peak"]:
            best = dict(hitter=tid, hand=("left", "right")[c], peak=peak, contact=frames[int(sp[:, c].argmax())])
    return best
