"""Compare hitter rules on the hand-labelled windows in data/check/hitter_check.csv (hitter_ok y/n; ? skipped).

Usage: compare_hitter_rules.py <extracted dir>

Rules: current (fastest single-frame wrist jump), smoothed and body-relative variants, closest wrist to the labelled
ball, the player nearest the ball where the ball turns ("turn"), and fastest wrist among players near the ball.
Also writes data/check/strips/ball_*.jpg: full frames with the ball (green), current pick (yellow), turn pick (magenta).
"""
import csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.data import NAMES_TRUSTED, load_poses
from padel.tracking import track_players
from padel.strokes import _series, wrist_speeds, L_SH, R_SH, L_HIP, R_HIP
L_WR, R_WR = 9, 10
ext = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open("data/check/hitter_check.csv")) if r["hitter_ok"] in "yn"]
tr = {}
for m in ("FinalF", "FinalM"):
    last = max(int(r["end"]) for r in rows if r["match"] == m) + 10
    poses, _ = load_poses(ext / "labels" / f"2022_BCN_{m}_1_pose.json", NAMES_TRUSTED[m])
    tr[m] = track_players({f: v for f, v in poses.items() if f <= last})

def rel_speeds(kp):
    c = np.nanmean(kp[:, [L_SH, R_SH, L_HIP, R_HIP], :2], 1)
    torso = np.linalg.norm(np.nanmean(kp[:, [L_SH, R_SH], :2], 1) - np.nanmean(kp[:, [L_HIP, R_HIP], :2], 1), axis=1)
    t = np.nanmedian(torso)
    sp = np.full((len(kp), 2), np.nan)
    for i, w in enumerate((L_WR, R_WR)):
        sp[1:, i] = np.linalg.norm(np.diff(kp[:, w, :2] - c, axis=0), axis=1) / t
    return sp

def pick(tracks, start, end, fn, smooth, pad=2):
    frames = list(range(start - pad, end + pad + 1))
    ids = {i for f in frames for i in tracks.get(f, {})}
    best = None
    for tid in ids:
        kp = _series(tracks, tid, frames)
        if np.isfinite(kp[:, 0, 0]).sum() < 6: continue
        sp = fn(kp)
        if smooth > 1:
            sp = np.stack([np.convolve(np.nan_to_num(sp[:, c]), np.ones(smooth) / smooth, "same") for c in (0, 1)], 1)
        sp = np.nan_to_num(sp)
        v = sp.max()
        if best is None or v > best[0]: best = (v, tid)
    return best[1]

for name, fn in (("image", wrist_speeds), ("body-relative", rel_speeds)):
    for sm in (1, 3):
        agree = {"y": 0, "n": 0}; tot = {"y": 0, "n": 0}; changed = []
        for r in rows:
            h = pick(tr[r["match"]], int(r["start"]), int(r["end"]), fn, sm)
            tot[r["hitter_ok"]] += 1
            same = h == int(r["hitter"])
            if (r["hitter_ok"] == "y") == same: agree[r["hitter_ok"]] += 1
        print(f"{name:14s} smooth {sm}: kept {agree['y']}/{tot['y']} right ones, changed {agree['n']}/{tot['n']} wrong ones")
print()
for r in rows:
    if r["hitter_ok"] != "n": continue
    s, e = int(r["start"]) - 2, int(r["end"]) + 2
    t = tr[r["match"]]
    ids = sorted({i for f in range(s, e + 1) for i in t.get(f, {})})
    info = []
    for tid in ids:
        kp = _series(t, tid, range(s, e + 1)); n = int(np.isfinite(kp[:, 0, 0]).sum())
        sp = np.nan_to_num(wrist_speeds(kp)).max() if n >= 6 else float("nan")
        y = np.nanmean(kp[:, 11:13, 1]) if n else float("nan")
        info.append(f"t{tid}:{n}f hipy{y:.0f} pk{sp:.2f}")
    print(r["id"], r["match"], "picked", r["hitter"], "|", "  ".join(info))
print()
import json
from padel.data import frame_index
ball = {}
for m in ("FinalF", "FinalM"):
    d = json.load(open(ext / "labels" / f"2022_BCN_{m}_1_ball.json"))
    fn = {im["id"]: frame_index(im["file_name"]) for im in d["images"]}
    ball[m] = {fn[a["image_id"]]: np.array([a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2])
               for a in d["annotations"] if a["category_id"] == 1}

def ball_pick(m, start, end, pad=2):
    frames = list(range(start - pad, end + pad + 1)); t = tr[m]
    nb = sum(f in ball[m] for f in frames)
    best = None
    for tid in {i for f in frames for i in t.get(f, {})}:
        kp = _series(t, tid, frames)
        torso = np.nanmedian(np.linalg.norm(np.nanmean(kp[:, [L_SH, R_SH], :2], 1) - np.nanmean(kp[:, [L_HIP, R_HIP], :2], 1), axis=1))
        ds = [np.nanmin(np.linalg.norm(kp[i, [L_WR, R_WR], :2] - ball[m][f], axis=1)) / torso
              for i, f in enumerate(frames) if f in ball[m] and np.isfinite(kp[i, 0, 0])]
        if ds and (best is None or min(ds) < best[0]): best = (min(ds), tid)
    return (best[1] if best else None), nb, (best[0] if best else None)

ok = {"y": 0, "n": 0}; tot = {"y": 0, "n": 0}
for r in rows:
    h, nb, d = ball_pick(r["match"], int(r["start"]), int(r["end"]))
    tot[r["hitter_ok"]] += 1
    same = h == int(r["hitter"])
    if (r["hitter_ok"] == "y") == same: ok[r["hitter_ok"]] += 1
    if r["hitter_ok"] == "n" or not same: print("  ", r["id"], r["hitter_ok"], "old", r["hitter"], "ball pick", h, "ball frames", nb, "dist", None if d is None else round(d, 2))
print(f"ball rule: kept {ok['y']}/{tot['y']} right ones, changed {ok['n']}/{tot['n']} wrong ones")
print()
def turn_frame(m, frames):
    fs = [f for f in frames if f in ball[m] and f - 1 in ball[m] and f + 1 in ball[m]]
    best = None
    for f in fs:
        a = ball[m][f] - ball[m][f - 1]; b = ball[m][f + 1] - ball[m][f]
        na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb_ < 1e-6: continue
        ang = np.degrees(np.arccos(np.clip(a @ b / na / nb_, -1, 1)))
        if best is None or ang > best[0]: best = (ang, f)
    return best

def pick2(m, start, end, mode, pad=2, near=1.5):
    frames = list(range(start - pad, end + pad + 1)); t = tr[m]
    tf = turn_frame(m, frames)
    cands = []
    for tid in {i for f in frames for i in t.get(f, {})}:
        kp = _series(t, tid, frames)
        if np.isfinite(kp[:, 0, 0]).sum() < 6: continue
        torso = np.nanmedian(np.linalg.norm(np.nanmean(kp[:, [L_SH, R_SH], :2], 1) - np.nanmean(kp[:, [L_HIP, R_HIP], :2], 1), axis=1))
        if mode == "turn":
            if tf is None: return None
            i = frames.index(tf[1]); seg = kp[max(i - 1, 0):i + 2, [L_WR, R_WR], :2]
            d = np.nanmin(np.linalg.norm(seg - ball[m][tf[1]], axis=2)) / torso
            cands.append((d, tid))
        else:
            ds = [np.nanmin(np.linalg.norm(kp[i, [L_WR, R_WR], :2] - ball[m][f], axis=1)) / torso
                  for i, f in enumerate(frames) if f in ball[m] and np.isfinite(kp[i, 0, 0])]
            sp = np.nan_to_num(wrist_speeds(kp)).max()
            cands.append((-sp if ds and min(ds) < near else np.inf, tid))
    cands = [c for c in cands if np.isfinite(c[0])]
    return min(cands)[1] if cands else None

for mode in ("turn", "speed-near-ball"):
    ok = {"y": 0, "n": 0}; tot = {"y": 0, "n": 0}; bad = []
    for r in rows:
        h = pick2(r["match"], int(r["start"]), int(r["end"]), mode)
        if h is None: h = int(r["hitter"])
        tot[r["hitter_ok"]] += 1
        same = h == int(r["hitter"])
        if (r["hitter_ok"] == "y") == same: ok[r["hitter_ok"]] += 1
        else: bad.append(r["id"])
    print(f"{mode}: kept {ok['y']}/{tot['y']} right, changed {ok['n']}/{tot['n']} wrong; disagreements {bad}")
import cv2
vid = {m: cv2.VideoCapture(str(ext / f"2022_BCN_{m}_1.mp4")) for m in tr}
ids = [x["id"] for x in rows if x["hitter_ok"] == "n"]
def boxof(kp):
    xy = kp[kp[:, 2] > 0, :2]; return xy.min(0).astype(int), xy.max(0).astype(int)
strips = []
for r in [x for x in rows if x["id"] in ids]:
    m = r["match"]; old = int(r["hitter"]); new = pick2(m, int(r["start"]), int(r["end"]), "turn")
    frames = list(range(int(r["start"]) - 2, int(r["end"]) + 3)); tf = turn_frame(m, frames)
    c = tf[1] if tf else int(r["contact"])
    tiles = []
    for o in (-6, -3, 0, 3, 6):
        f = c + o; vid[m].set(cv2.CAP_PROP_POS_FRAMES, f); _, img = vid[m].read()
        for tid, col in ((old, (0, 255, 255)), (new, (255, 0, 255))):
            kp = tr[m].get(f, {}).get(tid)
            if kp is not None:
                a, b = boxof(kp); cv2.rectangle(img, tuple(a - 8 - (tid == new) * 6), tuple(b + 8 + (tid == new) * 6), col, 3)
        if f in ball[m]: cv2.circle(img, tuple(ball[m][f].astype(int)), 14, (0, 255, 0), 3)
        sm = cv2.resize(img, (480, 270)); cv2.putText(sm, f"{o:+d}", (6, 22), 0, 0.7, (255, 255, 255), 2); tiles.append(sm)
    s = np.hstack(tiles); cv2.putText(s, f"#{r['id']} mine={r['hitter_ok']} {r['type']}", (1900, 24), 0, 0.8, (0, 255, 255), 2); strips.append(s)
for k in range(0, len(strips), 6):
    cv2.imwrite(f"data/check/strips/ball_{k // 6 + 1}.jpg", np.vstack(strips[k:k + 6]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("ok")
