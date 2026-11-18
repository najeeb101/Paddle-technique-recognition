"""Day 5: keypoints and features for every near-side stroke window, plus an overlay plot of 20 forehands.

Usage: build_stroke_features.py <v1 labels dir> data/strokes_index_full.csv data/processed/stroke_features.npz

Each stroke covers its window plus a 10-frame margin each side. The .npz holds, per stroke id
(e.g. FinalF_000045), `<id>/kp` (T,17,3) and `<id>/feat` (T,F), plus `meta` rows
(id, match, type, hitter, handed, start, contact, end, first_frame) and `feature_names`.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses  # noqa: E402
from padel.features import FEATURES, fix_lr_flips, stroke_features  # noqa: E402
from padel.strokes import _series  # noqa: E402
from padel.tracking import track_players  # noqa: E402

MARGIN, FPS = 10, 30.0

labels, idx, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
rows = [r for r in csv.DictReader(open(idx)) if r["near_side"] == "1" and r["handed"]]
arrays, meta, n_swapped = {}, [], 0
for match in ("FinalF", "FinalM"):
    mine = [r for r in rows if r["match"] == match]
    poses, _ = load_poses(labels / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["end"]) for r in mine) + MARGIN + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    for r in mine:
        start, end = int(r["start"]), int(r["end"])
        frames = list(range(start - MARGIN, end + MARGIN + 1))
        kp = _series(tracks, int(r["hitter"]), frames)
        kp[np.isnan(kp[..., 2])] = (np.nan, np.nan, 0.0)
        kp, swapped = fix_lr_flips(kp)
        n_swapped += int(swapped.any())
        sid = f"{match}_{start:06d}"
        hand = "right" if r["handed"] == "R" else "left"
        arrays[f"{sid}/kp"] = kp.astype(np.float32)
        arrays[f"{sid}/feat"] = stroke_features(kp, hand, FPS).astype(np.float32)
        meta.append((sid, match, r["type"], r["hitter"], r["handed"], start, int(r["contact"]), end, frames[0]))

out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, meta=np.array(meta, dtype=str), feature_names=np.array(FEATURES), **arrays)
print(f"wrote {len(meta)} strokes to {out}; {n_swapped} had left/right swaps corrected")

allf = np.concatenate([arrays[f"{m[0]}/feat"] for m in meta])
print("missing (NaN) share per feature:")
for name, share in zip(FEATURES, np.isnan(allf).mean(axis=0)):
    print(f"  {name:18s} {100 * share:5.1f}%")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for stroke in ("Forehand", "Backhand"):
    sample = [m for m in meta if m[2] == stroke][:20]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
    for ax, (j, name) in zip(axes.flat, enumerate(FEATURES)):
        for m in sample:
            f = arrays[f"{m[0]}/feat"][:, j]
            t = np.arange(len(f)) + int(m[8]) - int(m[6])
            ax.plot(t, f, lw=1, alpha=0.6, color="tab:red" if m[4] == "L" else "tab:blue")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(name)
    for ax in axes[-1]:
        ax.set_xlabel("frames from contact")
    fig.suptitle(f"{len(sample)} near-side {stroke.lower()}s (blue right-handed, red left-handed mirrored)")
    fig.tight_layout()
    png = out.parent / f"curves_{stroke.lower()}.png"
    fig.savefig(png, dpi=80)
    print("wrote", png)
