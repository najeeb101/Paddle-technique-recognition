"""Draw each file's keypoints, coloured by the file's own 'right_*' / 'left_*' names, on crops of near-side players.
Right = red, left = green. Near-side players face away from the camera, so a player's right side should be on the image right."""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
root, match, video, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
frames = [int(x) for x in sys.argv[5:]]
d = json.load(open(root / f"2022_BCN_{match}_1_pose.json"))
names = d["categories"][0]["keypoints"]
fid = {im["file_name"]: im["id"] for im in d["images"]}
cap = cv2.VideoCapture(video); tiles = []
for fr in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, img = cap.read()
    for a in d["annotations"]:
        if a["image_id"] != fid[f"frame_{fr:06d}.PNG"]: continue
        x, y, w, h = a["bbox"]
        if y + h < 650: continue  # keep near-side players only
        kp = np.array(a["keypoints"]).reshape(17, 3)
        g = img.copy()
        for n, (px, py, _) in zip(names, kp):
            if n.startswith(("right_", "left_")):
                cv2.circle(g, (int(px), int(py)), 4, (0, 0, 255) if n.startswith("right") else (0, 255, 0), -1)
        m = 40; x0, y0 = max(int(x - m), 0), max(int(y - m), 0)
        c = g[y0:int(y + h + m), x0:int(x + w + m)]
        tiles.append(cv2.resize(c, (240, int(240 * c.shape[0] / c.shape[1]))))
H = max(t.shape[0] for t in tiles)
cv2.imwrite(out, np.hstack([cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]))
