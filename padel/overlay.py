"""Skeleton overlay video: keypoints drawn on every frame, the scored swing labelled, encoded as H.264 for phones."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
COL_L, COL_R, COL_MID, COL_HIT = (255, 170, 60), (60, 140, 255), (230, 230, 230), (40, 220, 120)  # BGR


def draw_frame(img: np.ndarray, kp: np.ndarray, label: str | None, contact: bool = False, blank: bool = False) -> np.ndarray:
    """Return a copy of `img` (or a black frame with `blank`) with the (17,3) skeleton and an optional label."""
    out = np.zeros_like(img) if blank else img.copy()
    scale = max(1, round(img.shape[0] / 360))
    ok = np.isfinite(kp[:, 0]) & (kp[:, 2] > 0)
    pts = {j: (int(kp[j, 0]), int(kp[j, 1])) for j in range(17) if ok[j]}
    for a, b in EDGES:
        if a in pts and b in pts:
            col = COL_L if a in LEFT and b in LEFT else COL_R if a not in LEFT and b not in LEFT else COL_MID
            cv2.line(out, pts[a], pts[b], col, 2 * scale, cv2.LINE_AA)
    for j, p in pts.items():
        cv2.circle(out, p, 3 * scale, COL_L if j in LEFT else COL_R, -1, cv2.LINE_AA)
    if contact:
        for w in (9, 10):
            if w in pts:
                cv2.circle(out, pts[w], 14 * scale, COL_HIT, 2 * scale, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (12 * scale, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0), 4 * scale, cv2.LINE_AA)
        cv2.putText(out, label, (12 * scale, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, COL_HIT, 2 * scale, cv2.LINE_AA)
    return out


def render_overlay(video: str | Path, kp: np.ndarray, result: dict, out_path: str | Path, blank: bool = False) -> Path:
    """Draw the skeleton on every frame of `video`, label each scored swing, write `out_path` (H.264 if ffmpeg exists)."""
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    labels = {}
    for s in result.get("strokes", []):
        text = f"{result['stroke_type'].capitalize()}  {s['score']:.0f}/100"
        for f in range(s["start_frame"], s["end_frame"] + 1):
            labels[f] = (text, abs(f - s["contact_frame"]) <= 2)
    raw = out_path.with_suffix(".raw.mp4")
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        text, hit = labels.get(i, (None, False))
        pose = kp[i] if i < len(kp) else np.zeros((17, 3))
        writer.write(draw_frame(frame, pose, text, contact=hit, blank=blank))
        i += 1
    cap.release()
    if writer is None:
        raise ValueError(f"no frames in {video}")
    writer.release()
    if shutil.which("ffmpeg") is None:
        raw.replace(out_path)  # phones may not play mp4v; install ffmpeg for H.264
        return out_path
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out_path)], check=True)
    raw.unlink()
    return out_path
