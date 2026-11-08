"""Run the tracker on a frame range and report how stable the four tracks are; save a trajectory plot."""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players, hip_centre

root, match, f0, f1, outpng = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
poses, _ = load_poses(root / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
sub = {f: v for f, v in poses.items() if f0 <= f < f1}
tr = track_players(sub)
ids = sorted({i for v in tr.values() for i in v})
full = sum(len(v) == 4 for v in tr.values())
print(match, "frames", len(sub), "track ids", ids, "frames with all 4 tracks", full)
jumps = []
canvas = np.full((540, 960, 3), 255, np.uint8); col = [(255,0,0),(0,160,0),(0,0,255),(0,150,255)]
prev = {}
for f in sorted(tr):
    for i, kp in tr[f].items():
        c = hip_centre(kp)
        if i in prev and f - prev[i][0] == 1: jumps.append(np.linalg.norm(c - prev[i][1]))
        prev[i] = (f, c)
        cv2.circle(canvas, (int(c[0] / 2), int(c[1] / 2)), 1, col[i % 4], -1)
print("max frame-to-frame hip jump px: %.0f, 99th pct %.0f" % (max(jumps), np.percentile(jumps, 99)))
cv2.imwrite(outpng, canvas)
