"""Contact sheets for labelling the racket hand of near-side strokes by eye, plus a CSV to fill in.

Usage: make_hand_sheets.py <v1 extracted dir> data/strokes_index_full.csv <out dir> [per_sheet]

Each window shows the hitter at contact -6, -3, 0 and +3, cropped around the pose, with no keypoints drawn
so the label comes from the racket alone. Seen from behind, the player's right is image right.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses  # noqa: E402
from padel.tracking import track_players  # noqa: E402

ext, idx, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
per = int(sys.argv[4]) if len(sys.argv) > 4 else 8
outdir.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(idx)) if r["near_side"] == "1"]
S, COLS, OFFSETS = 200, 2, (-6, -3, 0, 3)


def crop(img, kp):
    ok = kp[:, 2] > 0
    if not ok.any():
        return np.zeros((S, S, 3), np.uint8)
    x0, y0 = kp[ok, :2].min(0)
    x1, y1 = kp[ok, :2].max(0)
    h = max(y1 - y0, x1 - x0, 80) * 0.9
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = int(max(cy - h, 0)), int(min(cy + h, img.shape[0]))
    c, d = int(max(cx - h, 0)), int(min(cx + h, img.shape[1]))
    return cv2.resize(img[a:b, c:d], (S, S))


tiles = []
for match in ("FinalF", "FinalM"):
    mine = [r for r in rows if r["match"] == match]
    poses, _ = load_poses(ext / "labels" / f"2022_BCN_{match}_1_pose.json", NAMES_TRUSTED[match])
    tracks = track_players({f: v for f, v in poses.items() if f <= max(int(r["end"]) for r in mine) + 5})
    cap = cv2.VideoCapture(str(ext / f"2022_BCN_{match}_1.mp4"))
    for r in sorted(mine, key=lambda r: int(r["contact"])):
        c, tid = int(r["contact"]), int(r["hitter"])
        pair = []
        for f in (c + o for o in OFFSETS):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            kp = tracks.get(f, {}).get(tid)
            pair.append(crop(img, kp) if ok and kp is not None else np.zeros((S, S, 3), np.uint8))
        tiles.append((r, np.hstack(pair)))

out_rows = []
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    cells = []
    for k, (r, t) in enumerate(chunk):
        n = s + k + 1
        cv2.rectangle(t, (0, 0), (len(OFFSETS) * S - 1, S - 1), (255, 255, 255), 2)
        cv2.putText(t, f"#{n} {r['type'][:2]}", (6, 22), 0, 0.7, (0, 255, 255), 2)
        cells.append(t)
        out_rows.append(dict(id=n, match=r["match"], start=r["start"], type=r["type"], hand=r["hand"],
                             handed=r["handed"], racket_hand=""))
    while len(cells) % COLS:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)])
    cv2.imwrite(str(outdir / f"hand_{s // per + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
with open(outdir / "hand_labels.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print("sheets:", (len(tiles) + per - 1) // per, "windows:", len(tiles))
