"""Test the swing detector on every near-side pro stroke, cut as a 3 s clip like a phone clip.

Usage: eval_swings.py <v1 labels dir> data/strokes_index_full.csv

Each stroke becomes a clip of 90 frames with the labelled contact at frame 45, taken from the dataset keypoints of
the hitter's track and cleaned and smoothed like phone keypoints (padel/pose.py). The detector runs on the
wrist_speed feature, as analyze_clip did, and on swing_speed (wrist relative to the
body centre). A stroke counts as found when a swing's contact lies within 6 frames of the labelled contact; every other swing in the clip is counted as extra (it may still be a real second shot).
Settings are compared on the women's final (FinalF) and the chosen one is checked on the men's final (FinalM).
"""
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses  # noqa: E402
from padel.features import FEATURES, fix_lr_all, stroke_features  # noqa: E402
from padel.pose import clean_keypoints, smooth_keypoints  # noqa: E402
from padel.strokes import _series  # noqa: E402
from padel.swings import detect_swings, swing_speed  # noqa: E402
from padel.tracking import track_players  # noqa: E402

BEFORE, AFTER, TOL, FPS = 45, 44, 6, 30.0
WS = FEATURES.index("wrist_speed")

labels, idx = Path(sys.argv[1]), sys.argv[2]
rows = [r for r in csv.DictReader(open(idx)) if r["near_side"] == "1" and r["handed"]]
speeds = {"FinalF": [], "FinalM": []}
rel = {"FinalF": [], "FinalM": []}
for match in speeds:
    mine = [r for r in rows if r["match"] == match]
    poses, _ = load_poses(labels / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["contact"]) for r in mine) + AFTER + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    for r in mine:
        c = int(r["contact"])
        kp = _series(tracks, int(r["hitter"]), range(c - BEFORE, c + AFTER + 1))
        kp[np.isnan(kp[..., 2])] = (np.nan, np.nan, 0.0)
        kp = fix_lr_all(smooth_keypoints(clean_keypoints(kp)))
        hand = "right" if r["handed"] == "R" else "left"
        speeds[match].append((r["type"], stroke_features(kp, hand, FPS)[:, WS]))
        rel[match].append((r["type"], swing_speed(kp, hand, FPS)))
    print(f"{match}: {len(speeds[match])} clips")


def evaluate(clips, **kw):
    found = extra = 0
    for _, s in clips:
        sw = detect_swings(s, FPS, **kw)
        hit = [abs(c - BEFORE) <= TOL for _, c, _ in sw]
        found += any(hit)
        extra += len(sw) - any(hit)
    return found / len(clips), extra / len(clips)


def report(label, sig):
    print(f"\n===== {label} =====")
    default = dict(peak_factor=3.0, smooth=5, min_peak=2.0)
    print("\ncurrent settings:")
    for match, clips in sig.items():
        for t in ("Forehand", "Backhand"):
            f, e = evaluate([x for x in clips if x[0] == t], **default)
            print(f"  {match} {t:8s} found {100 * f:5.1f}%  extra swings per clip {e:.2f}")

    grid = list(product((1.5, 2.0, 2.5, 3.0), (1, 3, 5), (1.0, 1.5, 2.0)))
    res = []
    for pf, sm, mp in grid:
        f, e = evaluate(sig["FinalF"], peak_factor=pf, smooth=sm, min_peak=mp)
        res.append((f - 0.5 * e, f, e, pf, sm, mp))
    res.sort(reverse=True)
    print("\nFinalF, best 8 (rank by found - 0.5 x extra):")
    print("  peak_factor smooth min_peak   found  extra")
    for _, f, e, pf, sm, mp in res[:8]:
        print(f"  {pf:11.1f} {sm:6d} {mp:8.1f}  {100 * f:5.1f}%  {e:.2f}")
    _, _, _, pf, sm, mp = res[0]
    print(f"\nchosen peak_factor={pf} smooth={sm} min_peak={mp}, checked on FinalM:")
    for t in ("Forehand", "Backhand"):
        for name, kw in (("current", default), ("chosen", dict(peak_factor=pf, smooth=sm, min_peak=mp))):
            f, e = evaluate([x for x in sig["FinalM"] if x[0] == t], **kw)
            print(f"  {t:8s} {name:7s} found {100 * f:5.1f}%  extra swings per clip {e:.2f}")

    all_s = np.concatenate([s for c in sig.values() for _, s in c])
    print(f"\nwrist_speed over all clips: median {np.nanmedian(all_s):.2f}, missing {100 * np.isnan(all_s).mean():.1f}%")


report("wrist_speed", speeds)
report("swing_speed", rel)
