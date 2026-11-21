"""Larger strips for the 40-window hitter check (data/check/hitter_check.csv): per window, frames at contact -8, -4,
0, +4, +8 of a wide crop (half the frame height) around the inferred hitter, with every tracked player boxed (the
inferred hitter yellow, others grey). Five windows per image, so ball, racket and the other players stay visible.

Usage: make_hitter_strips.py <extracted dir> data/check/hitter_check.csv data/check/strips
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses  # noqa: E402
from padel.tracking import track_players  # noqa: E402

ext, ans, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(ans)))
OFFS, H, TW = (-8, -4, 0, 4, 8), 540, 360
tr, vid = {}, {}
for m in ("FinalF", "FinalM"):
    last = max(int(r["contact"]) for r in rows if r["match"] == m) + 20
    poses, _ = load_poses(ext / "labels" / f"2022_BCN_{m}_1_pose.json", NAMES_TRUSTED[m])
    tr[m] = track_players({f: v for f, v in poses.items() if f <= last})
    vid[m] = cv2.VideoCapture(str(ext / f"2022_BCN_{m}_1.mp4"))


def box(kp):
    ok = kp[:, 2] > 0
    xy = kp[ok, :2]
    return xy.min(0).astype(int), xy.max(0).astype(int)


def tile(r, o):
    m, f, tid = r["match"], int(r["contact"]) + o, int(r["hitter"])
    vid[m].set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = vid[m].read()
    people = tr[m].get(f, {})
    centre = people.get(tid)
    if centre is None:
        centre = tr[m].get(int(r["contact"]), {}).get(tid)
    for t, kp in people.items():
        a, b = box(kp)
        cv2.rectangle(img, tuple(a - 6), tuple(b + 6), (0, 255, 255) if t == tid else (150, 150, 150), 3 if t == tid else 1)
        if t == tid:
            for i, col in ((9, (0, 255, 0)), (10, (0, 0, 255))):
                cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 6, col, -1)
    cx, cy = centre[11:13, :2].mean(0)
    w = int(H * 1.0)
    x0 = int(np.clip(cx - w / 2, 0, img.shape[1] - w))
    y0 = int(np.clip(cy - H * 0.6, 0, img.shape[0] - H))
    c = cv2.resize(img[y0:y0 + H, x0:x0 + w], (TW, TW))
    cv2.putText(c, "contact" if o == 0 else f"{o:+d}", (6, 22), 0, 0.7, (255, 255, 255), 2)
    return c


for s in range(0, len(rows), 5):
    strips = []
    for r in rows[s:s + 5]:
        strip = np.hstack([tile(r, o) for o in OFFS])
        cv2.putText(strip, f"#{r['id']} {r['match']} {r['type']}", (TW * 5 - 330, 22), 0, 0.7, (0, 255, 255), 2)
        strips.append(strip)
    out = outdir / f"strip_{s + 1:02d}-{s + len(strips):02d}.jpg"
    cv2.imwrite(str(out), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)
