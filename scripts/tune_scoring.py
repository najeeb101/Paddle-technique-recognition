"""Choose the scoring settings (SOFTNESS, WORST_MIX, timing on/off) on training strokes only.

Usage: tune_scoring.py data/processed/stroke_features.npz [--seed 0]

The outer 80/20 split is the same as eval_holdout.py and eval_flawed.py, and its held-out 20% is never used
here. The training 80% is split again 75/25: references are built on the first part and every setting is scored
on the second (validation) part. The objective is the mean of
  - the clean-vs-flawed AUC over the five faults at size 1 (scripts/eval_flawed.py), and
  - the share of validation strokes scoring higher against their own stroke type than the other one.
The faults are synthetic, so a setting chosen this way is tuned towards them; the held-out report shows whether
it carries over to strokes the tuning never saw.
"""
import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.faults import FAULTS, apply_fault, auc  # noqa: E402
from padel.reference import build_reference, resample  # noqa: E402
from padel.scoring import score_stroke, timing_stats  # noqa: E402

SOFTNESS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
WORST_MIX = (0.0, 0.25, 0.5, 0.75, 1.0)
TIMING = (False, True)

ap = argparse.ArgumentParser()
ap.add_argument("features")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

text = (root / "docs" / "label-suspects.md").read_text(encoding="utf-8")
rows = [ln for ln in text.splitlines() if ln.startswith("| Likely") or ln.startswith("| Ambiguous")]
excluded = set(re.findall(r"Final[FM]_\d{6}", "\n".join(rows)))
d = np.load(args.features)
names = [str(n) for n in d["feature_names"]]

rng = np.random.default_rng(args.seed)
inner = np.random.default_rng(args.seed + 1000)
refs, val = {}, {}
for stroke in ("Forehand", "Backhand"):
    metas = [m for m in d["meta"] if m[2] == stroke and m[0] not in excluded]
    order = rng.permutation(len(metas))
    train = [metas[i] for i in order[:int(0.8 * len(metas))]]
    o = inner.permutation(len(train))
    cut = int(0.75 * len(train))
    build, val[stroke] = [train[i] for i in o[:cut]], [train[i] for i in o[cut:]]
    raw = [(d[f"{m[0]}/feat"], int(m[6]) - int(m[8])) for m in build]
    r = build_reference([resample(*x) for x in raw])
    r.update(timing_stats(raw, names))
    refs[stroke] = {"mean": r["mean"], "std": r["std"], "scale": r["scale"], "names": names,
                    "timing_mean": r["timing_mean"], "timing_std": r["timing_std"]}
    print(f"{stroke}: build {len(build)}, validation {len(val[stroke])}, turn peak "
          f"{r['timing_mean']:.3f} +- {r['timing_std']:.3f} s from contact")

results = []
for soft, mix, timing in itertools.product(SOFTNESS, WORST_MIX, TIMING):
    kw = dict(softness=soft, worst_mix=mix, use_timing=timing)
    aucs, order_ok, clean_all = {}, [], []
    for stroke, held in val.items():
        other = "Backhand" if stroke == "Forehand" else "Forehand"
        clean = []
        for m in held:
            feat, c = d[f"{m[0]}/feat"], int(m[6]) - int(m[8])
            s = score_stroke(feat, c, refs[stroke], **kw).score
            clean.append(s)
            order_ok.append(s > score_stroke(feat, c, refs[other], **kw).score)
        clean_all += clean
        for fault in FAULTS:
            fl = [score_stroke(*apply_fault(d[f"{m[0]}/feat"], int(m[6]) - int(m[8]), names, fault, 1.0),
                               refs[stroke], **kw).score for m in held]
            aucs[f"{stroke[:2]} {fault}"] = auc(clean, fl)
    mean_auc, ordering = float(np.mean(list(aucs.values()))), float(np.mean(order_ok))
    results.append(((mean_auc + ordering) / 2, soft, mix, timing, mean_auc, ordering, np.mean(clean_all), np.std(clean_all)))

results.sort(reverse=True)
print("\n| Objective | Softness | Worst mix | Timing | Mean fault AUC | Own type higher | Clean score (sd) |")
print("|---|---|---|---|---|---|---|")
for obj, soft, mix, timing, a, o, cm, cs in results[:10]:
    print(f"| {obj:.3f} | {soft:g} | {mix:g} | {'on' if timing else 'off'} | {a:.3f} | {100 * o:.0f}% | {cm:.1f} ({cs:.1f}) |")
