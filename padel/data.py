"""Loaders for the PadelTracker100 pose JSON and shot CSV files."""
from __future__ import annotations

import csv
import json
from itertools import groupby
from pathlib import Path

import numpy as np

# Canonical COCO-17 order; poses are re-ordered into this using each file's own category names.
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
STROKE_TYPES = ("Forehand", "Backhand")


def frame_index(file_name: str) -> int:
    """'frame_000123.PNG' -> 123."""
    return int(Path(file_name).stem.split("_")[1])


def load_poses(path: str | Path, swap_lr: bool = False):
    """Return {frame: array (n_people, 17, 3)} with keypoints in canonical COCO order.

    Columns are reordered by the file's own keypoint names. If `swap_lr` is set the left/right
    pairs are exchanged afterwards (use when the names, not the coordinates, are what is wrong).
    """
    d = json.load(open(path))
    names = d["categories"][0]["keypoints"]
    order = [names.index(n) for n in COCO17]
    fname = {im["id"]: frame_index(im["file_name"]) for im in d["images"]}
    out: dict[int, list] = {}
    for a in d["annotations"]:
        kp = np.asarray(a["keypoints"], dtype=float).reshape(17, 3)[order]
        if swap_lr:
            kp = kp[[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]]
        out.setdefault(fname[a["image_id"]], []).append(kp)
    return {f: np.stack(v) for f, v in out.items()}, names


def load_shots(path: str | Path):
    """Return a list of (frame, category) for rows with a shot; category '0'/empty -> skipped."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["has_shot"].strip() == "1":
                rows.append((frame_index(r["file_name"]), r["category"].strip()))
    n_rows = sum(1 for _ in open(path)) - 1
    return rows, n_rows


def stroke_windows(shots, types=STROKE_TYPES):
    """Group consecutive frames of the same category into windows -> (type, start, end)."""
    wins = []
    run_start = prev = None
    cur = None
    for frame, cat in shots:
        if cur is not None and cat == cur and frame == prev + 1:
            prev = frame
            continue
        if cur in types:
            wins.append((cur, run_start, prev))
        cur, run_start, prev = cat, frame, frame
    if cur in types:
        wins.append((cur, run_start, prev))
    return wins
