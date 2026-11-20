"""Flawed-stroke test (plan day 8-9): do held-out pro strokes score higher than flawed copies of themselves?

Usage: eval_flawed.py data/processed/stroke_features.npz [--seed 0] [--train-match FinalF|FinalM]

References are rebuilt from training strokes only (the committed reference/ files saw every stroke).
Default split: 80/20 per stroke type with the same seed logic as eval_holdout.py. --train-match builds from
one final and tests on the other (leave-match-out, since player identity is not tracked).
Strokes in docs/label-suspects.md are left out, as in the other scripts. Prints markdown tables.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.faults import FAULTS, apply_fault, auc  # noqa: E402
from padel.reference import build_reference, resample  # noqa: E402
from padel.scoring import score_stroke  # noqa: E402

SIZES = (0.5, 1.0, 2.0, 4.0)
NAMED = {"rotation": {"shoulder_rotation", "separation"}}  # either name counts for the rotation fault

ap = argparse.ArgumentParser()
ap.add_argument("features")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train-match", choices=("FinalF", "FinalM"))
args = ap.parse_args()

text = (root / "docs" / "label-suspects.md").read_text(encoding="utf-8")
rows = [ln for ln in text.splitlines() if ln.startswith("| Likely") or ln.startswith("| Ambiguous")]
excluded = set(re.findall(r"Final[FM]_\d{6}", "\n".join(rows)))

d = np.load(args.features)
names = [str(n) for n in d["feature_names"]]
rng = np.random.default_rng(args.seed)
split = f"leave-match-out (build {args.train_match})" if args.train_match else f"80/20 per type, seed {args.seed}"
print(f"Split: {split}. Faults at size 1: " + "; ".join(f"{k} = {f.text}" for k, f in FAULTS.items()) + "\n")

summary, sens = [], []
for stroke in ("Forehand", "Backhand"):
    metas = [m for m in d["meta"] if m[2] == stroke and m[0] not in excluded]
    if args.train_match:
        train = [m for m in metas if m[1] == args.train_match]
        held = [m for m in metas if m[1] != args.train_match]
    else:
        order = rng.permutation(len(metas))
        cut = int(0.8 * len(metas))
        train, held = [metas[i] for i in order[:cut]], [metas[i] for i in order[cut:]]
    r = build_reference([resample(d[f"{m[0]}/feat"], int(m[6]) - int(m[8])) for m in train])
    ref = {"mean": r["mean"], "std": r["std"], "scale": r["scale"], "names": names}

    clean = {}
    for m in held:
        clean[m[0]] = score_stroke(d[f"{m[0]}/feat"], int(m[6]) - int(m[8]), ref).score
    for fault in FAULTS:
        flawed, named = [], []
        drops = {s: [] for s in SIZES}
        for m in held:
            feat, c = d[f"{m[0]}/feat"], int(m[6]) - int(m[8])
            for s in SIZES:
                g, gc = apply_fault(feat, c, names, fault, s)
                res = score_stroke(g, gc, ref)
                drops[s].append(clean[m[0]] - res.score)
                if s == 1.0:
                    flawed.append(res.score)
                    want = NAMED.get(fault, {FAULTS[fault].feature})
                    named.append(bool(want & {dv.feature for dv in res.deviations}))
        cl, fl = np.array([clean[m[0]] for m in held]), np.array(flawed)
        named_pct = "n/a" if FAULTS[fault].feature is None else f"{100 * np.mean(named):.0f}%"
        summary.append(f"| {stroke} | {fault} | {len(train)} / {len(held)} | {cl.mean():.1f} | {fl.mean():.1f} | "
                       f"{(cl - fl).mean():.1f} | {100 * (fl < cl).mean():.0f}% | {named_pct} | {auc(cl, fl):.2f} |")
        sens.append(f"| {stroke} | {fault} | " + " | ".join(f"{np.mean(drops[s]):.1f}" for s in SIZES) + " |")

print("| Stroke | Fault | Build / test strokes | Clean score | Flawed score | Mean drop | Score dropped | Fault in top 3 feedback | AUC |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(summary))
print("\nMean score drop by fault size (1 = the size above):\n")
print("| Stroke | Fault | " + " | ".join(f"x{s:g}" for s in SIZES) + " |")
print("|---|---|" + "---|" * len(SIZES))
print("\n".join(sens))
