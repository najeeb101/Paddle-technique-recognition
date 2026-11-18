"""Build forehand and backhand references from the near-side stroke features.

Usage: build_reference.py data/processed/stroke_features.npz reference/
Strokes listed in docs/label-suspects.md (likely backhand or ambiguous) are excluded from the forehand build.
"""
import re
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from padel.reference import CONTACT_STEP, N_STEPS, build_reference, resample  # noqa: E402

src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
text = (root / "docs" / "label-suspects.md").read_text(encoding="utf-8")
rows = [ln for ln in text.splitlines() if ln.startswith("| Likely") or ln.startswith("| Ambiguous")]
excluded = set(re.findall(r"Final[FM]_\d{6}", "\n".join(rows)))

d = np.load(src)
names = [str(n) for n in d["feature_names"]]
for stroke in ("Forehand", "Backhand"):
    seqs, ids, matches = [], [], set()
    for m in d["meta"]:
        sid = m[0]
        if m[2] != stroke or sid in excluded:
            continue
        seqs.append(resample(d[f"{sid}/feat"], int(m[6]) - int(m[8])))
        ids.append(sid)
        matches.add(str(m[1]))
    ref = build_reference(seqs)
    out = outdir / f"{stroke.lower()}.npz"
    np.savez_compressed(
        out, mean=ref["mean"].astype(np.float32), std=ref["std"].astype(np.float32),
        scale=ref["scale"].astype(np.float32), feature_names=np.array(names), n_strokes=ref["n_strokes"],
        medoid=ids[ref["medoid"]], matches=np.array(sorted(matches)), contact_step=CONTACT_STEP,
        n_steps=N_STEPS, stroke_ids=np.array(ids),
    )
    print(f"{stroke}: {len(ids)} strokes, medoid {ids[ref['medoid']]} -> {out}")
    nan = {n: round(float(np.isnan(ref["mean"][:, j]).mean()), 2) for j, n in enumerate(names) if np.isnan(ref["mean"][:, j]).any()}
    print("  features with NaN steps in the mean:", nan or "none")
