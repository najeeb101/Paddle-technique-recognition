"""Add hitter, racket hand, peak speed and contact frame to strokes_index.csv; print a handedness table.

Usage: infer_hitters.py <labels dir> data/strokes_index.csv data/strokes_hitters.csv [--ball]

With --ball the hitter is the player nearest the labelled ball where it changes direction (padel/strokes.py
analyse_window_ball, checked in docs/results.md 6e); without, the fastest wrist (analyse_window).
"""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players
from padel.strokes import analyse_window, analyse_window_ball, load_ball

root = Path(sys.argv[1]); idx = Path(sys.argv[2]); out = Path(sys.argv[3]); use_ball = "--ball" in sys.argv
rows = list(csv.DictReader(open(idx)))
res = []
for match in ("FinalF", "FinalM"):
    poses, _ = load_poses(root / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["end"]) for r in rows if r["match"] == match) + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    ball = load_ball(root / f"2022_BCN_{match}_1_ball.json") if use_ball else None
    for r in rows:
        if r["match"] != match: continue
        if use_ball:
            a = analyse_window_ball(tracks, ball, int(r["start"]), int(r["end"]))
        else:
            a = analyse_window(tracks, int(r["start"]), int(r["end"]))
        empty = dict(hitter="", hand="", peak="", contact="", **(dict(turn="", rule="") if use_ball else {}))
        res.append({**r, **(a or empty)})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print("windows", len(res), "no hitter found", sum(r["hitter"] == "" for r in res))
hands = defaultdict(Counter)
for r in res:
    if r["hitter"] != "": hands[(r["match"], r["hitter"])][r["hand"]] += 1
for k in sorted(hands): print(k, dict(hands[k]))
