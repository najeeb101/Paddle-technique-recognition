"""Compare the professionals' spread between two stroke feature files (for example dataset poses vs re-run crops).

Usage: compare_spread.py data/processed/stroke_features.npz data/processed/stroke_features_crop.npz

Only strokes present in both files are used, with the same label-suspect exclusions as the other scripts. For each
stroke type and feature, prints the median per-step std over the contact window (steps 30 to 50) of a reference
built on all those strokes, and the share of missing values.
"""
import re
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.reference import build_reference, resample  # noqa: E402
from padel.scoring import DEFAULT_FEATURES  # noqa: E402

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
text = (root / "docs" / "label-suspects.md").read_text(encoding="utf-8")
rows = [ln for ln in text.splitlines() if ln.startswith("| Likely") or ln.startswith("| Ambiguous")]
excluded = set(re.findall(r"Final[FM]_\d{6}", "\n".join(rows)))
names = [str(n) for n in a["feature_names"]]
common = {m[0] for m in b["meta"]} & {m[0] for m in a["meta"]}

print(f"| Stroke | Feature | Spread, {Path(sys.argv[1]).stem} | Spread, {Path(sys.argv[2]).stem} | Change | Missing (a / b) |")
print("|---|---|---|---|---|---|")
for stroke in ("Forehand", "Backhand"):
    metas = [m for m in a["meta"] if m[2] == stroke and m[0] in common and m[0] not in excluded]
    out = []
    for d in (a, b):
        seqs = [resample(d[f"{m[0]}/feat"], int(m[6]) - int(m[8])) for m in metas]
        r = build_reference(seqs)
        miss = np.isnan(np.concatenate([d[f"{m[0]}/feat"] for m in metas])).mean(0)
        out.append((np.median(r["std"][30:51], axis=0), miss))
    for f in DEFAULT_FEATURES:
        j = names.index(f)
        s0, s1 = out[0][0][j], out[1][0][j]
        unit = "" if f == "wrist_height" else " deg"
        fmt = "{:.2f}" if f == "wrist_height" else "{:.1f}"
        print(f"| {stroke} ({len(metas)}) | {f} | {fmt.format(s0)}{unit} | {fmt.format(s1)}{unit} | "
              f"{100 * (s1 / s0 - 1):+.0f}% | {100 * out[0][1][j]:.0f}% / {100 * out[1][1][j]:.0f}% |")
