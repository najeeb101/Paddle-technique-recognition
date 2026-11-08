"""Contact sheet: for N men's windows show the inferred hitter's crop at the contact frame (left wrist green, right wrist red)."""
import csv, sys, json
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players

root, video, csvp, out, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = [r for r in csv.DictReader(open(csvp)) if r["match"] == "FinalM" and r["hitter"] != ""]
rows = rows[:: max(1, len(rows) // n)][:n]
poses, _ = load_poses(root / "2022_BCN_FinalM_1_pose.json", NAMES_TRUSTED["FinalM"])
last = max(int(r["end"]) for r in rows) + 5
tracks = track_players({f: v for f, v in poses.items() if f <= last})
cap = cv2.VideoCapture(video); tiles = []
for r in rows:
    c = int(r["contact"]); kp = tracks[c].get(int(r["hitter"]))
    if kp is None: continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, c); ok, img = cap.read()
    for i, col in ((9, (0, 255, 0)), (10, (0, 0, 255))): cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 6, col, -1)
    x, y = kp[[11, 12], :2].mean(0); m = 130
    crop = img[max(int(y - m * 1.3), 0):int(y + m), max(int(x - m), 0):int(x + m)]
    crop = cv2.resize(crop, (240, 240))
    cv2.putText(crop, f"{r['type'][:2]} {r['hand'][0]} f{c}", (4, 18), 0, 0.55, (255, 255, 255), 2)
    tiles.append(crop)
while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
