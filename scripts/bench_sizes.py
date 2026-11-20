"""Day 17: speed of YOLO11-pose sizes on CPU and (if present) GPU for one clip, plus the rest of the phone pipeline.

Usage: bench_sizes.py clip.mp4 <models dir> [frames]   (models dir holds yolo11{n,s,m}-pose.pt)
Speed only: people found and mean keypoint confidence are a sanity check, not an accuracy measurement.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.analyze import analyze_keypoints  # noqa: E402
from padel.overlay import render_overlay  # noqa: E402
from padel.pose import clean_keypoints, extract_keypoints, smooth_keypoints, yolo_detector  # noqa: E402
from padel.scoring import load_reference  # noqa: E402

clip, models = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
cap = cv2.VideoCapture(str(clip))
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
h, w = frames[0].shape[:2]
import torch  # noqa: E402

devices = [("cpu", "CPU")] + ([("0", torch.cuda.get_device_name(0))] if torch.cuda.is_available() else [])
print(f"{len(frames)} frames of {w}x{h}; devices: {', '.join(name for _, name in devices)}")

kp = None
for dev, name in devices:
    for size in ("n", "s", "m"):
        detect = yolo_detector(models / f"yolo11{size}-pose.pt", device=dev)
        detect(frames[0])  # warm-up, not timed
        t = time.perf_counter()
        raw, warns = extract_keypoints(frames, detect)
        per = (time.perf_counter() - t) / len(frames)
        conf = raw[:, :, 2][raw[:, :, 2] > 0]
        print(f"{name:22s} yolo11{size}-pose: {per:.3f} s/frame ({1 / per:.1f} fps), 10 s clip at 30 fps = "
              f"{300 * per:.0f} s; mean kp conf {conf.mean():.2f}")
        kp = raw

t = time.perf_counter()
clean = smooth_keypoints(clean_keypoints(kp))
print(f"clean + smooth: {1000 * (time.perf_counter() - t) / len(frames):.2f} ms/frame")
t = time.perf_counter()
analyze_keypoints(clean, 30.0, "right", "forehand", load_reference(Path(__file__).resolve().parents[1] / "reference" / "forehand.npz"))
print(f"analysis (lr fix, features, swings, scoring): {time.perf_counter() - t:.2f} s for {len(frames)} frames")
out = Path(sys.argv[1]).with_name("bench_overlay.mp4")
t = time.perf_counter()
render_overlay(clip, clean, {"stroke_type": "forehand", "strokes": []}, out)
print(f"overlay (draw + H.264): {time.perf_counter() - t:.2f} s for the whole clip")
out.unlink()
