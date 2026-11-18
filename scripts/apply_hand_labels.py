"""Replace the rule-based `handed` column with the eye-checked racket hand from make_hand_sheets.py.

Usage: apply_hand_labels.py data/strokes_index_full.csv data/check/hand/hand_labels.csv

The old value is kept as `handed_rule`. Windows labelled '?' get an empty `handed`, so later steps skip them.
"""
import csv
import sys
from collections import Counter

idx, labels = sys.argv[1], sys.argv[2]
hand = {(r["match"], r["start"]): r["racket_hand"] for r in csv.DictReader(open(labels))}
rows = list(csv.DictReader(open(idx)))
for r in rows:
    if "handed_rule" not in r:
        r["handed_rule"] = r["handed"]
    lab = hand.get((r["match"], r["start"]))
    if lab is not None:
        r["handed"] = lab if lab in ("R", "L") else ""
with open(idx, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(Counter((r["match"], r["type"], r["handed"]) for r in rows if r["near_side"] == "1"))
