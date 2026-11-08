"""Day 1-2: load labels, print counts, write strokes_index.csv (one row per stroke window)."""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import load_poses, load_shots, stroke_windows  # noqa: E402

root = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/strokes_index.csv")
out.parent.mkdir(parents=True, exist_ok=True)
rows = []
for match in ("FinalF", "FinalM"):
    poses, names = load_poses(root / f"2022_BCN_{match}_1_pose.json")
    shots, n_rows = load_shots(root / f"2022_BCN_{match}_1_shots.csv")
    wins = stroke_windows(shots)
    print(match, "csv rows", n_rows, "pose frames", len(poses), "windows", Counter(w[0] for w in wins))
    print("  keypoint names[1:3]:", names[1:3], "people/frame:", Counter(len(v) for v in poses.values()))
    for t, s, e in wins:
        rows.append((match, t, s, e, e - s + 1))
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["match", "type", "start", "end", "n_frames"])
    w.writerows(rows)
print("wrote", out, len(rows), "rows")
