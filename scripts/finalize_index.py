"""Add per-window near_side and handed (R/L, from wrist side vs stroke type) to strokes_hitters.csv -> strokes_index_full.csv."""
import csv, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players, hip_centre

root, inp, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(inp))); res = []
for match in ("FinalF", "FinalM"):
    poses, _ = load_poses(root / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["end"]) for r in rows if r["match"] == match) + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    for r in rows:
        if r["match"] != match: continue
        kp = tracks[int(r["contact"])].get(int(r["hitter"])) if r["hitter"] != "" else None
        if kp is None: res.append({**r, "near_side": "", "handed": ""}); continue
        hx, hy = hip_centre(kp); near = hy >= 600
        h = ""
        if near:
            right = kp[9 if r["hand"] == "left" else 10, 0] > hx
            h = "R" if right == (r["type"] == "Forehand") else "L"
        res.append({**r, "near_side": int(near), "handed": h})
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(Counter((r["match"], r["type"], r["near_side"], r["handed"]) for r in res if r["near_side"] != ""))
print("near-side usable:", Counter(r["type"] for r in res if r["near_side"] == 1))
