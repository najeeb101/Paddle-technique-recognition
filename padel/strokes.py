"""Per stroke window: infer the hitter, their racket hand and the contact frame (from the ball if labelled, else wrist speed)."""
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


def load_ball(path) -> dict:
    """PadelTracker100 ball JSON -> {frame: (x, y) ball centre}; 'Wall' boxes are skipped."""
    import json
    from pathlib import Path
    d = json.load(open(path))
    fname = {im["id"]: int(Path(im["file_name"]).stem.split("_")[1]) for im in d["images"]}
    return {fname[a["image_id"]]: np.array([a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2])
            for a in d["annotations"] if a["category_id"] == 1}


def ball_turn(ball: dict, frames) -> int | None:
    """Frame in `frames` where the ball's direction of travel changes most (needs the ball on both neighbours)."""
    best = None
    for f in frames:
        if not (f - 1 in ball and f in ball and f + 1 in ball):
            continue
        a, b = ball[f] - ball[f - 1], ball[f + 1] - ball[f]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        ang = float(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0)))
        if best is None or ang > best[0]:
            best = (ang, f)
    return None if best is None else best[1]


def analyse_window_ball(tracks, ball: dict, start: int, end: int, pad: int = 2, min_frames: int = 6):
    """Like analyse_window, but the hitter is the player whose wrist is nearest the ball (in torso lengths, over the
    turn frame and its two neighbours) where the ball changes direction, and the racket hand is that wrist.
    Contact stays the speed peak of the racket wrist, as in the phone app. Falls back to analyse_window when the
    ball gives no turn. Adds `turn` (frame or '') and `rule` ('ball' or 'speed')."""
    frames = list(range(start - pad, end + pad + 1))
    turn = ball_turn(ball, frames)
    if turn is None:
        a = analyse_window(tracks, start, end, pad, min_frames)
        return None if a is None else {**a, "turn": "", "rule": "speed"}
    near = [f for f in (turn - 1, turn, turn + 1) if f in frames]
    best = None
    for tid in {i for f in frames for i in tracks.get(f, {})}:
        kp = _series(tracks, tid, frames)
        if np.isfinite(kp[:, 0, 0]).sum() < min_frames:
            continue
        torso = np.linalg.norm(np.nanmean(kp[:, [L_SH, R_SH], :2], 1) - np.nanmean(kp[:, [L_HIP, R_HIP], :2], 1), axis=1)
        if not np.isfinite(torso).any():
            continue
        t = float(np.nanmedian(torso))
        for f in near:
            wr = kp[frames.index(f), [L_WR, R_WR], :2]
            d = np.linalg.norm(wr - ball[turn], axis=1) / t
            if not np.isfinite(d).any():
                continue
            c = int(np.nanargmin(d))
            if best is None or d[c] < best[0]:
                best = (float(d[c]), tid, c, kp)
    if best is None:
        a = analyse_window(tracks, start, end, pad, min_frames)
        return None if a is None else {**a, "turn": "", "rule": "speed"}
    _, tid, c, kp = best
    sp = np.nan_to_num(wrist_speeds(kp), nan=0.0)[:, c]
    return dict(hitter=tid, hand=("left", "right")[c], peak=float(sp.max()), contact=frames[int(sp.argmax())],
                turn=turn, rule="ball")
