"""Handedness vote per near-side track from forehand/backhand geometry (seen from behind: forehand wrist on the racket side)."""
import csv, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players, hip_centre

root, csvp = Path(sys.argv[1]), sys.argv[2]
rows = [r for r in csv.DictReader(open(csvp)) if r["hitter"] != ""]
votes = defaultdict(Counter); ys = defaultdict(list)
for match in ("FinalF", "FinalM"):
    poses, _ = load_poses(root / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    last = max(int(r["end"]) for r in rows if r["match"] == match) + 5
    tracks = track_players({f: v for f, v in poses.items() if f <= last})
    for r in rows:
        if r["match"] != match: continue
        kp = tracks[int(r["contact"])].get(int(r["hitter"]))
        if kp is None: continue
        hy = hip_centre(kp)[1]; ys[(match, r["hitter"])].append(hy)
        if hy < 600: continue  # far side: skip (mirrored view)
        wr = kp[9 if r["hand"] == "left" else 10, 0] - hip_centre(kp)[0]
        right_side = wr > 0
        votes[(match, r["hitter"])][("R" if right_side == (r["type"] == "Forehand") else "L")] += 1
for k in sorted(votes):
    c = votes[k]; n = sum(c.values()); print(k, "near-side strokes", n, dict(c), "-> %s %.0f%%" % (max(c, key=c.get), 100 * max(c.values()) / n),
          "| median hip y %.0f" % np.median(ys[k]))
for k in sorted(set(ys) - set(votes)): print(k, "far side, median hip y %.0f, n %d" % (np.median(ys[k]), len(ys[k])))
