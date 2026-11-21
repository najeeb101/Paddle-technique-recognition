"""Find swings in a player clip from the racket wrist's speed relative to the body (design.md section 6.1)."""
from __future__ import annotations

import numpy as np

from .features import L_HIP, L_SH, R_HIP, R_SH, R_WR, as_right_handed


def swing_speed(kp: np.ndarray, hand: str, fps: float) -> np.ndarray:
    """Racket wrist speed relative to the body centre (mid shoulders and hips), torso lengths per second.

    Unlike the wrist_speed feature this ignores the player running or the camera panning, which on match video
    lifts the clip's median speed so far that real swings fall under the peak threshold.
    """
    p = as_right_handed(kp, hand)
    centre = (p[:, L_SH, :2] + p[:, R_SH, :2] + p[:, L_HIP, :2] + p[:, R_HIP, :2]) / 4
    torso = np.linalg.norm((p[:, L_SH, :2] + p[:, R_SH, :2]) / 2 - (p[:, L_HIP, :2] + p[:, R_HIP, :2]) / 2, axis=1)
    rel = p[:, R_WR, :2] - centre
    out = np.full(len(p), np.nan)
    out[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) / torso[1:] * fps
    return out


def detect_swings(speed: np.ndarray, fps: float, peak_factor: float = 3.0, low_factor: float = 1.5,
                  min_peak: float = 2.0, min_gap_s: float = 0.8, max_side_s: float = 0.7, smooth: int = 5):
    """Wrist speed per frame (torso lengths per second) -> list of (start, contact, end) frame indices.

    A swing is a local maximum of the smoothed speed above both `peak_factor` x the clip's median and the
    absolute floor `min_peak` (so jitter in a still clip is not a swing). Peaks closer than `min_gap_s` keep
    only the highest. Contact is the peak; the window runs out to where speed drops below `low_factor` x the
    median, at most `max_side_s` each side. The thresholds are proposals to check on real clips.
    """
    s = np.asarray(speed, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return []
    t = np.arange(len(s))
    s = np.interp(t, t[ok], s[ok])
    s = np.convolve(s, np.ones(smooth) / smooth, mode="same")
    med = float(np.median(s))
    high, low = max(peak_factor * med, min_peak), low_factor * med
    peaks = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] > high]
    gap, side = round(min_gap_s * fps), round(max_side_s * fps)
    kept: list[int] = []
    for p in sorted(peaks, key=lambda i: -s[i]):
        if all(abs(p - k) >= gap for k in kept):
            kept.append(p)
    swings = []
    for c in sorted(kept):
        a = c
        while a > 0 and c - a < side and s[a] >= low:
            a -= 1
        b = c
        while b < len(s) - 1 and b - c < side and s[b] >= low:
            b += 1
        swings.append((a, c, b))
    return swings
