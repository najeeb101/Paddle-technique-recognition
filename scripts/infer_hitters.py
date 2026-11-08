"""Add hitter, racket hand, peak speed and contact frame to strokes_index.csv; print a handedness table."""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players
from padel.strokes import analyse_window

root = Path(sys.argv[1]); idx = Path(sys.argv[2]); out = Path(sys.argv[3])
rows = list(csv.DictReader(open(idx)))
res = []
for match in ("FinalF", "FinalM"):
    poses, _ = load_poses(root / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["end"]) for r in rows if r["match"] == match) + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    for r in rows:
        if r["match"] != match: continue
        a = analyse_window(tracks, int(r["start"]), int(r["end"]))
        res.append({**r, **(a or dict(hitter="", hand="", peak="", contact=""))})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print("windows", len(res), "no hitter found", sum(r["hitter"] == "" for r in res))
hands = defaultdict(Counter)
for r in res:
    if r["hitter"] != "": hands[(r["match"], r["hitter"])][r["hand"]] += 1
for k in sorted(hands): print(k, dict(hands[k]))
