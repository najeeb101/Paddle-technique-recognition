"""Recompute the features of a stroke file from its keypoints, optionally with the hip-width mask.

Usage: refeature.py <in.npz> <out.npz> [--min-hip-width 0.15]

Keypoints, meta and stroke ids are copied unchanged; only `<id>/feat` is recomputed (padel/features.py
stroke_features), so the eval scripts can compare feature settings on exactly the same strokes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from padel.features import FEATURES, stroke_features  # noqa: E402

FPS = 30.0
src, out = Path(sys.argv[1]), Path(sys.argv[2])
mhw = float(sys.argv[sys.argv.index("--min-hip-width") + 1]) if "--min-hip-width" in sys.argv else None
d = np.load(src)
arrays = {}
for m in d["meta"]:
    kp = d[f"{m[0]}/kp"].astype(float)
    hand = "right" if m[4] == "R" else "left"
    arrays[f"{m[0]}/kp"] = d[f"{m[0]}/kp"]
    arrays[f"{m[0]}/feat"] = stroke_features(kp, hand, FPS, min_hip_width=mhw).astype(np.float32)
np.savez_compressed(out, meta=d["meta"], feature_names=np.array(FEATURES), **arrays)
allf = np.concatenate([arrays[f"{m[0]}/feat"] for m in d["meta"]])
print(f"wrote {len(d['meta'])} strokes to {out} (min_hip_width={mhw})")
for name in ("hip_rotation", "separation"):
    print(f"  {name:13s} missing {100 * np.isnan(allf[:, FEATURES.index(name)]).mean():5.1f}%")
