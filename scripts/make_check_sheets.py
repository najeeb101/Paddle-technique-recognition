"""Make 2 contact sheets (20 windows each) for hand-checking the hitter and contact frame, plus an answer CSV.
Each window is a row: crops of the inferred hitter at contact-4, contact, contact+4 (left wrist green, right wrist red)."""
import csv, random, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players, hip_centre

ext, idx, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); outdir.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(idx)) if r["hitter"] != ""]
random.seed(7); pick = []
for m in ("FinalF", "FinalM"):
    pick += random.sample([r for r in rows if r["match"] == m], 20)
random.shuffle(pick)
tr, vid = {}, {}
for m in ("FinalF", "FinalM"):
    poses, _ = load_poses(ext / "labels" / f"2022_BCN_{m}_1_pose.json", NAMES_TRUSTED[m])
    tr[m] = track_players({f: v for f, v in poses.items() if f <= 20200}); vid[m] = cv2.VideoCapture(str(ext / f"2022_BCN_{m}_1.mp4"))
S = 190
def tile(m, f, tid, label):
    vid[m].set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = vid[m].read()
    kp = tr[m].get(f, {}).get(tid)
    if not ok or kp is None: return np.zeros((S, S, 3), np.uint8)
    for i, col in ((9, (0, 255, 0)), (10, (0, 0, 255))): cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 5, col, -1)
    x, y = hip_centre(kp); h = 120
    c = img[max(int(y - h * 1.2), 0):int(y + h * 0.8), max(int(x - h), 0):int(x + h)]
    c = cv2.resize(c, (S, S)); cv2.putText(c, label, (4, 16), 0, 0.5, (255, 255, 255), 2); return c
ans = []
for s in range(2):
    rws = []
    for k, r in enumerate(pick[s * 20:(s + 1) * 20]):
        n = s * 20 + k + 1; c = int(r["contact"]); tid = int(r["hitter"])
        row = np.hstack([tile(r["match"], c + o, tid, l) for o, l in ((-4, "-4"), (0, "contact"), (4, "+4"))])
        cv2.putText(row, f"#{n} {r['type']}", (S * 3 - 190, 18), 0, 0.6, (0, 255, 255), 2); rws.append(row)
        ans.append(dict(id=n, match=r["match"], type=r["type"], start=r["start"], end=r["end"], contact=c, hitter=r["hitter"], hand=r["hand"], hitter_ok="", contact_offset_frames=""))
    cols = [np.vstack(rws[i:i + 10]) for i in (0, 10)]
    cv2.imwrite(str(outdir / f"sheet{s + 1}.jpg"), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 85])
with open(outdir / "hitter_check.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ans[0])); w.writeheader(); w.writerows(ans)
print("done", len(ans))
