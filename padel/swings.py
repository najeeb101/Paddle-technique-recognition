"""Find swings in a player clip from the racket wrist's speed (design.md section 6.1)."""
from __future__ import annotations

import numpy as np


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
