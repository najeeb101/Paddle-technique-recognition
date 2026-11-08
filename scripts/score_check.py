"""Read data/check/hitter_check.csv after hand-marking and print hitter accuracy and contact-frame error."""
import csv, sys, statistics as st
rows = [r for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "data/check/hitter_check.csv")) if r["hitter_ok"].strip()]
ok = [r for r in rows if r["hitter_ok"].strip().lower() in ("y", "yes", "1")]
print(f"hitter correct: {len(ok)}/{len(rows)} = {100 * len(ok) / max(len(rows), 1):.0f}%")
off = [abs(int(r["contact_offset_frames"])) for r in ok if r["contact_offset_frames"].strip().lstrip("-").isdigit()]
if off: print(f"contact error (correct hitters only, n={len(off)}): median {st.median(off)} frames, max {max(off)}")
