"""Day 11: turn one phone clip into a keypoint file. Usage: clip_to_keypoints.py clip.mp4 yolo11m-pose.pt out.npz

The .npz holds `keypoints` (cleaned and smoothed, (T,17,3)), `raw` (straight from the model), `fps` and `warnings`.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.pose import clip_keypoints  # noqa: E402

video, weights, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.parent.mkdir(parents=True, exist_ok=True)
t = time.time()
r = clip_keypoints(video, weights)
secs = time.time() - t
np.savez_compressed(out, keypoints=r["keypoints"], raw=r["raw"], fps=r["fps"], warnings=np.array(r["warnings"], dtype=str))
T = len(r["keypoints"])
dom = r["raw"][:, [5, 6, 7, 8, 9, 10], 2]  # shoulders, elbows, wrists
print(f"{video.name}: {T} frames at {r['fps']:.1f} fps, pose took {secs:.1f} s ({secs / max(T, 1):.3f} s/frame)")
print(f"frames where all arm keypoints have conf >= 0.5: {100 * (dom >= 0.5).all(axis=1).mean():.0f}%")
print(f"keypoints still missing after cleaning: {100 * np.isnan(r['keypoints'][:, :, 0]).mean():.1f}%")
for w in r["warnings"]:
    print("warning:", w)
print("wrote", out)
