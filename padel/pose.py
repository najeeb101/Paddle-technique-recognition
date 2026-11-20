"""Phone clip -> one player's keypoints: run the pose model, keep one person, drop unsure points, fill short gaps, smooth.

Missing keypoints are stored as x = y = NaN and conf = 0 (design.md sections 3.1, 6.2 and 6.3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

MISSING = np.array([np.nan, np.nan, 0.0])


def _runs(mask: np.ndarray):
    """Yield (start, end) of each run of True values in a 1-D mask, end exclusive."""
    f, T = 0, len(mask)
    while f < T:
        if not mask[f]:
            f += 1
            continue
        g = f
        while g < T and mask[g]:
            g += 1
        yield f, g
        f = g


def clean_keypoints(kp: np.ndarray, min_conf: float = 0.3, max_gap: int = 3) -> np.ndarray:
    """Mark points with conf below `min_conf` as missing, then fill gaps of up to `max_gap` frames linearly.

    Longer gaps, and gaps touching the start or end of the clip, stay missing (no guessing past the data).
    """
    out = np.array(kp, dtype=float)
    bad = ~(out[:, :, 2] >= min_conf)  # also catches NaN conf
    out[bad] = MISSING
    T = len(out)
    for j in range(out.shape[1]):
        for f, g in _runs(bad[:, j]):
            if f == 0 or g == T or g - f > max_gap:
                continue
            a, b = out[f - 1, j], out[g, j]
            w = (np.arange(f, g) - (f - 1)) / (g - (f - 1))
            out[f:g, j] = a + w[:, None] * (b - a)
    return out


def smooth_keypoints(kp: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing of x, y on each unbroken run of valid frames; runs shorter than `window` are kept as is."""
    out = np.array(kp, dtype=float)
    ok = np.isfinite(out[:, :, 0])
    for j in range(out.shape[1]):
        for f, g in _runs(ok[:, j]):
            if g - f >= window:
                out[f:g, j, :2] = savgol_filter(out[f:g, j, :2], window, order, axis=0)
    return out


def largest_person(boxes: np.ndarray) -> int | None:
    """Index of the box (x1, y1, x2, y2) with the largest area, or None when there are no boxes."""
    if len(boxes) == 0:
        return None
    b = np.asarray(boxes, dtype=float)
    return int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))


def extract_keypoints(frames, detect):
    """Run `detect(frame) -> (boxes (n,4), keypoints (n,17,3))` on every frame and keep the largest person.

    Returns ((T,17,3) float32 keypoints, list of warnings). Frames with nobody are all missing.
    """
    rows, n_none, n_many = [], 0, 0
    for frame in frames:
        boxes, kps = detect(frame)
        i = largest_person(boxes)
        if i is None:
            n_none += 1
            rows.append(np.tile(MISSING, (17, 1)))
        else:
            n_many += len(boxes) > 1
            rows.append(kps[i])
    warnings = []
    if n_many:
        warnings.append(f"more than one person in {n_many} frames; used the largest")
    if n_none:
        warnings.append(f"no person in {n_none} frames")
    return np.array(rows, dtype=np.float32).reshape(-1, 17, 3), warnings


def yolo_detector(weights: str | Path, device: str | None = None):
    """Wrap an Ultralytics YOLO pose model as a `detect` function for `extract_keypoints`.

    `device` is "cpu", "0" (first GPU) or None to let Ultralytics pick (a CUDA GPU if PyTorch can see one).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def detect(frame):
        r = model(frame, verbose=False, device=device)[0]
        if r.keypoints is None or r.keypoints.conf is None:
            return np.zeros((0, 4)), np.zeros((0, 17, 3))
        xy, conf = r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()
        return r.boxes.xyxy.cpu().numpy(), np.concatenate([xy, conf[..., None]], axis=2)

    return detect


def video_frames(path: str | Path):
    """Return (fps, generator of BGR frames) for a video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")

    def gen():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()

    return cap.get(cv2.CAP_PROP_FPS), gen()


def clip_keypoints(video: str | Path, weights: str | Path):
    """Full day-11 pipeline for one clip. Returns dict(raw, keypoints, fps, warnings)."""
    fps, frames = video_frames(video)
    raw, warnings = extract_keypoints(frames, yolo_detector(weights))
    keypoints = smooth_keypoints(clean_keypoints(raw)).astype(np.float32)
    return dict(raw=raw, keypoints=keypoints, fps=fps, warnings=warnings)
