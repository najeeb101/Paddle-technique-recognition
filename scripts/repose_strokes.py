"""Re-run the pose model on zoomed-in crops of every near-side stroke, to test whether sharper skeletons shrink the
professionals' spread.

Usage: repose_strokes.py data/processed/stroke_features.npz <videos dir> <yolo pose weights> data/processed/stroke_features_crop.npz

For each stroke, one square crop around the hitter (the union of the dataset keypoints over the stroke, padded) is
cut from every frame and scaled to 640 px, so the player fills the image as on a phone. In each crop the person whose
box overlaps the dataset hitter box most (IoU >= 0.3) is kept and mapped back to frame coordinates. Keypoints then go
through the same cleaning and smoothing as phone clips (padel/pose.py), the left/right fix and the features.
Output has the same layout as build_stroke_features.py, so build_reference.py and the eval scripts take it as is.
Progress is cached per stroke next to the output (<out>.parts/), so a stopped run resumes.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.features import FEATURES, fix_lr_all, stroke_features  # noqa: E402
from padel.pose import box_iou, clean_keypoints, smooth_keypoints, yolo_detector  # noqa: E402

PAD, SIZE, MIN_IOU, FPS = 1.4, 640, 0.3, 30.0

src, videos, weights, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
parts = out.with_suffix(".parts")
parts.mkdir(parents=True, exist_ok=True)
d = np.load(src)
meta = [tuple(m) for m in d["meta"]]
detect = yolo_detector(weights, device="cpu")


def box_of(kp):
    ok = kp[..., 2] > 0.3
    if not ok.any():
        return None
    xy = kp[..., :2][ok]
    return np.array([*xy.min(0), *xy.max(0)], dtype=float)


def crop_box(kp, w=1920, h=1080):
    b = box_of(kp.reshape(-1, 3))
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = min(max(b[2] - b[0], b[3] - b[1]) * PAD, h)
    x0 = float(np.clip(cx - side / 2, 0, w - side))
    y0 = float(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, side


todo = [m for m in meta if not (parts / f"{m[0]}.npz").exists()]
if len(sys.argv) > 5:  # optional stroke limit for a quick trial
    todo = todo[:int(sys.argv[5])]
print(f"{len(meta)} strokes, {len(todo)} to do", flush=True)
t_start, done = time.time(), 0
for match in ("FinalF", "FinalM"):
    mine = sorted([m for m in todo if m[1] == match], key=lambda m: int(m[8]))
    if not mine:
        continue
    jobs = {}
    for m in mine:
        kp = d[f"{m[0]}/kp"]
        first = int(m[8])
        x0, y0, side = crop_box(kp)
        jobs[m[0]] = dict(first=first, n=len(kp), crop=(x0, y0, side), ds=kp, out=np.full((len(kp), 17, 3), np.nan))
        jobs[m[0]]["out"][..., 2] = 0.0
    by_frame = {}
    for sid, j in jobs.items():
        for i in range(j["n"]):
            by_frame.setdefault(j["first"] + i, []).append(sid)
    last = max(by_frame)
    cap = cv2.VideoCapture(str(videos / f"2022_BCN_{match}_1.mp4"))
    f = 0
    remaining = {sid: j["n"] for sid, j in jobs.items()}
    while f <= last:
        ok = cap.grab()
        if not ok:
            break
        if f in by_frame:
            _, frame = cap.retrieve()
            for sid in by_frame[f]:
                j = jobs[sid]
                x0, y0, side = j["crop"]
                s = SIZE / side
                M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
                img = cv2.warpAffine(frame, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
                boxes, kps = detect(img)
                i = f - j["first"]
                ref = box_of(j["ds"][i])
                if len(boxes) and ref is not None:
                    ref_c = np.array([(ref[0] - x0) * s, (ref[1] - y0) * s, (ref[2] - x0) * s, (ref[3] - y0) * s])
                    iou = box_iou(ref_c, boxes)
                    k = int(np.argmax(iou))
                    if iou[k] >= MIN_IOU:
                        kp = kps[k].astype(float).copy()
                        kp[:, 0] = kp[:, 0] / s + x0
                        kp[:, 1] = kp[:, 1] / s + y0
                        j["out"][i] = kp
                remaining[sid] -= 1
                if remaining[sid] == 0:
                    raw = j["out"]
                    raw[raw[..., 2] <= 0, :2] = np.nan
                    np.savez(parts / f"{sid}.npz", raw=raw.astype(np.float32))
                    done += 1
                    if done % 10 == 0:
                        el = time.time() - t_start
                        print(f"{done}/{len(todo)} strokes, {el / 60:.1f} min, about {el / done * (len(todo) - done) / 60:.0f} min left", flush=True)
        f += 1
    cap.release()

arrays, n_found = {}, []
new_meta = []
for m in meta:
    p = parts / f"{m[0]}.npz"
    if not p.exists():
        continue
    raw = np.load(p)["raw"]
    n_found.append(float((raw[..., 2] > 0).any(axis=1).mean()))
    kp = smooth_keypoints(clean_keypoints(raw))
    kp = fix_lr_all(kp)
    hand = "right" if m[4] == "R" else "left"
    arrays[f"{m[0]}/kp"] = kp.astype(np.float32)
    arrays[f"{m[0]}/feat"] = stroke_features(kp, hand, FPS).astype(np.float32)
    new_meta.append(m)
np.savez_compressed(out, meta=np.array(new_meta, dtype=str), feature_names=np.array(FEATURES), **arrays)
print(f"wrote {len(new_meta)} strokes to {out}; hitter found in {100 * np.mean(n_found):.0f}% of frames on average")
allf = np.concatenate([arrays[f"{m[0]}/feat"] for m in new_meta])
for name, share in zip(FEATURES, np.isnan(allf).mean(axis=0)):
    print(f"  {name:18s} missing {100 * share:5.1f}%")
