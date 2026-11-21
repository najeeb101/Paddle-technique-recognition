"""Hold-out check: build references on 80% of strokes, score the other 20% against both stroke types.

Usage: eval_holdout.py data/processed/stroke_features.npz [seed]
Held-out forehands should score higher against the forehand reference than against the backhand one, and the
reverse for backhands. Strokes in docs/label-suspects.md (likely backhand or ambiguous) are left out entirely.
"""
import re
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.reference import build_reference, resample  # noqa: E402
from padel.scoring import score_stroke, timing_stats  # noqa: E402

src = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
text = (root / "docs" / "label-suspects.md").read_text(encoding="utf-8")
rows = [ln for ln in text.splitlines() if ln.startswith("| Likely") or ln.startswith("| Ambiguous")]
excluded = set(re.findall(r"Final[FM]_\d{6}", "\n".join(rows)))

d = np.load(src)
names = [str(n) for n in d["feature_names"]]
rng = np.random.default_rng(seed)
refs, test = {}, {}
for stroke in ("Forehand", "Backhand"):
    metas = [m for m in d["meta"] if m[2] == stroke and m[0] not in excluded]
    order = rng.permutation(len(metas))
    cut = int(0.8 * len(metas))
    train, held = [metas[i] for i in order[:cut]], [metas[i] for i in order[cut:]]
    raw = [(d[f"{m[0]}/feat"], int(m[6]) - int(m[8])) for m in train]
    r = build_reference([resample(*x) for x in raw])
    r.update(timing_stats(raw, names))
    refs[stroke] = {"mean": r["mean"], "std": r["std"], "scale": r["scale"], "names": names,
                    "timing_mean": r["timing_mean"], "timing_std": r["timing_std"]}
    test[stroke] = held
    print(f"{stroke}: train {len(train)}, held-out {len(held)}")

for stroke, held in test.items():
    other = "Backhand" if stroke == "Forehand" else "Forehand"
    own, wrong = [], []
    for m in held:
        feat, c = d[f"{m[0]}/feat"], int(m[6]) - int(m[8])
        own.append(score_stroke(feat, c, refs[stroke]).score)
        wrong.append(score_stroke(feat, c, refs[other]).score)
    own, wrong = np.array(own), np.array(wrong)
    print(f"{stroke}: own-type score mean {own.mean():.1f} (sd {own.std():.1f}), wrong-type {wrong.mean():.1f} (sd {wrong.std():.1f}); "
          f"higher for own type in {100 * (own > wrong).mean():.0f}% of strokes")
    print(f"  own-type score range {own.min():.0f}-{own.max():.0f}; share of strokes below 50: {100 * (own < 50).mean():.0f}%")
