"""Run the team's own phone clips through the app and summarise what works (docs/filming-guide.md).

Usage: eval_own_clips.py <clips dir with clips.csv> [--weights yolo11m-pose.pt] [--out data/own_clips] [--overlay]

Pose runs once per clip and is cached in <out>/keypoints/<clip>.npz, so a re-run after a code or threshold change
takes seconds (delete the cache after changing pose.py). Writes <out>/results.json (the app's result per clip) and
<out>/summary.md with:
- swing detection: clips with a swing found, failures by code, clips with more than one swing
- good vs fault: scores per stroke type and the AUC between them (0.5 = no separation)
- own vs other reference: does a good forehand score higher against forehand than against backhand
- named faults: is the matching feature among the three feedback lines, against the chance of naming it at random
- a swing-threshold sweep on the cached keypoints: each clip holds one stroke, so the ideal setting finds exactly one

With several swings in a clip, the one with the fastest racket wrist is taken as the stroke. No contact frame is
labelled, so a found swing is not proof it is the stroke: check the overlay videos (--overlay) by eye.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from itertools import product
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from padel.analyze import MAX_CLIP_S, analyze_keypoints, clip_seconds  # noqa: E402
from padel.faults import auc  # noqa: E402
from padel.features import fix_lr_all  # noqa: E402
from padel.scoring import DEFAULT_FEATURES, load_reference  # noqa: E402
from padel.swings import detect_swings, swing_speed  # noqa: E402

STROKES = ("forehand", "backhand")
FAULT_FEATURES = {
    "elbow_low": {"shoulder_angle", "elbow_angle"},
    "no_shoulder_turn": {"shoulder_rotation", "separation"},
    "late_contact": {"timing"},
    "no_knee_bend": {"knee_angle_dom", "knee_angle_off"},
    "straight_arm": {"elbow_angle"},
}
SWEEP = dict(peak_factor=(2.0, 2.5, 3.0), smooth=(3, 5), min_peak=(1.5, 2.0, 3.0))
DEFAULT_SWING = dict(peak_factor=3.0, smooth=5, min_peak=2.0)


def read_log(clips_dir: Path) -> list[dict]:
    """Rows of clips.csv whose video exists; prints a line for each problem row or unlogged video."""
    log = clips_dir / "clips.csv"
    if not log.exists():
        raise SystemExit(f"{log} not found; see docs/filming-guide.md for the template")
    rows, seen = [], set()
    for r in csv.DictReader(open(log, newline="", encoding="utf-8-sig")):
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        seen.add(r["clip"])
        if not (clips_dir / r["clip"]).exists():
            print(f"skip {r['clip']}: video not found")
        elif r["hand"] not in ("right", "left") or r["stroke"] not in STROKES or r["form"] not in ("good", "fault"):
            print(f"skip {r['clip']}: hand, stroke or form not filled in correctly")
        else:
            rows.append(r)
    for v in sorted(clips_dir.iterdir()):
        if v.suffix.lower() in {".mp4", ".mov", ".m4v"} and v.name not in seen:
            print(f"skip {v.name}: no row in clips.csv")
    return rows


def yolo_pose(weights: Path):
    """Return pose(video) -> dict(keypoints, fps, warnings), loading the model once for all clips."""
    from padel.pose import clean_keypoints, extract_keypoints, smooth_keypoints, video_frames, yolo_detector

    detect = yolo_detector(weights)

    def pose(video):
        fps, frames = video_frames(video)
        raw, warnings = extract_keypoints(frames, detect)
        return dict(keypoints=smooth_keypoints(clean_keypoints(raw)).astype(np.float32), fps=fps, warnings=warnings)
    return pose


def cached_pose(video: Path, cache: Path, pose) -> dict:
    f = cache / f"{video.stem}.npz"
    if f.exists():
        d = np.load(f)
        return dict(keypoints=d["keypoints"], fps=float(d["fps"]), warnings=[str(w) for w in d["warnings"]])
    p = pose(video)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, keypoints=p["keypoints"], fps=p["fps"], warnings=np.array(p["warnings"], dtype=str))
    return p


def main_stroke(result: dict, kp: np.ndarray, hand: str, fps: float) -> dict | None:
    """The scored swing with the fastest racket wrist at contact (one stroke per clip), or None."""
    if not result["strokes"]:
        return None
    s = swing_speed(fix_lr_all(kp), hand, fps)
    return max(result["strokes"], key=lambda x: np.nan_to_num(s[x["contact_frame"]], nan=-1))


def analyse(rows: list[dict], clips_dir: Path, out: Path, pose, refs: dict, overlay: bool = False) -> list[dict]:
    """Analyse every logged clip. Returns one record per clip: the log row, result, other-type score, keypoints."""
    records = []
    for i, r in enumerate(rows, 1):
        video = clips_dir / r["clip"]
        print(f"[{i}/{len(rows)}] {r['clip']}", flush=True)
        other = STROKES[1 - STROKES.index(r["stroke"])]
        if clip_seconds(video) > MAX_CLIP_S:
            res = dict(clip=r["clip"], strokes=[], n_swings_found=0,
                       failure={"code": "clip_too_long", "message": "over 20 s"})
            records.append(dict(row=r, result=res, main=None, other_score=None, kp=None, fps=None))
            continue
        p = cached_pose(video, out / "keypoints", pose)
        kp, fps = p["keypoints"], p["fps"]
        res = analyze_keypoints(kp, fps, r["hand"], r["stroke"], refs[r["stroke"]], clip=r["clip"], warnings=p["warnings"])
        main = main_stroke(res, kp, r["hand"], fps)
        other_score = None
        if main is not None:
            alt = analyze_keypoints(kp, fps, r["hand"], other, refs[other])
            match = [s for s in alt["strokes"] if s["contact_frame"] == main["contact_frame"]]
            other_score = match[0]["score"] if match else None
        if overlay and res["strokes"]:
            from padel.overlay import render_overlay

            (out / "overlays").mkdir(parents=True, exist_ok=True)
            render_overlay(video, kp, res, out / "overlays" / f"{video.stem}.mp4")
        records.append(dict(row=r, result=res, main=main, other_score=other_score, kp=kp, fps=fps))
    return records


def chance_named(k: int, n: int = len(DEFAULT_FEATURES), shown: int = 3) -> float:
    """Chance that at least one of k matching features is among `shown` features picked at random from n."""
    k = min(k, n)
    return 1 - comb(n - k, shown) / comb(n, shown)


def sweep(records: list[dict]) -> list[tuple]:
    """(found share, mean extra swings, setting) for each swing setting, best first (found - 0.5 x extra)."""
    clips = [(swing_speed(fix_lr_all(x["kp"]), x["row"]["hand"], x["fps"]), x["fps"]) for x in records if x["kp"] is not None]
    if not clips:
        return []
    res = []
    for pf, sm, mp in product(*SWEEP.values()):
        n = [len(detect_swings(s, fps, peak_factor=pf, smooth=sm, min_peak=mp)) for s, fps in clips]
        found, extra = np.mean([k > 0 for k in n]), np.mean([max(k - 1, 0) for k in n])
        res.append((found, extra, dict(peak_factor=pf, smooth=sm, min_peak=mp)))
    return sorted(res, key=lambda t: -(t[0] - 0.5 * t[1]))


def summarise(records: list[dict]) -> str:
    L = [f"# Own clips: results ({len(records)} clips)", ""]
    if not records:
        return "\n".join(L + ["No clips analysed."])

    L += ["## Swing detection", "", "| Stroke | Form | Clips | Scored | More than one swing | Failures |", "|---|---|---|---|---|---|"]
    for t in STROKES:
        for form in ("good", "fault"):
            g = [x for x in records if x["row"]["stroke"] == t and x["row"]["form"] == form]
            if not g:
                continue
            fails = Counter(x["result"]["failure"]["code"] for x in g if x["result"]["failure"])
            multi = sum(x["result"]["n_swings_found"] > 1 for x in g)
            fail_txt = ", ".join(f"{c} {n}" for c, n in fails.most_common()) or "none"
            L.append(f"| {t} | {form} | {len(g)} | {sum(x['main'] is not None for x in g)} | {multi} | {fail_txt} |")

    L += ["", "## Scores", "", "Score of the main swing against its own stroke type's reference.", "",
          "| Stroke | Good mean (n) | Fault mean (n) | AUC good vs fault | Good: own type higher than other type |",
          "|---|---|---|---|---|"]
    for t in STROKES:
        g = [x for x in records if x["row"]["stroke"] == t and x["main"] is not None]
        good = [x["main"]["score"] for x in g if x["row"]["form"] == "good"]
        bad = [x["main"]["score"] for x in g if x["row"]["form"] == "fault"]
        pairs = [(x["main"]["score"], x["other_score"]) for x in g if x["row"]["form"] == "good" and x["other_score"] is not None]
        m = lambda v: f"{np.mean(v):.1f} ({len(v)})" if v else "- (0)"  # noqa: E731
        a = f"{auc(good, bad):.2f}" if good and bad else "-"
        own = f"{100 * np.mean([o > w for o, w in pairs]):.0f}% of {len(pairs)}" if pairs else "-"
        L.append(f"| {t} | {m(good)} | {m(bad)} | {a} | {own} |")

    faults = [x for x in records if x["row"]["form"] == "fault" and x["main"] is not None]
    if faults:
        L += ["", "## Named faults", "", "Is a matching feature among the three feedback lines? "
              "Chance = picking three of the nine scored features at random.", "",
              "| Fault | Clips | Named | Chance | Matching features |", "|---|---|---|---|---|"]
        for f, feats in FAULT_FEATURES.items():
            g = [x for x in faults if x["row"]["fault"] == f]
            if not g:
                continue
            named = sum(any(d["feature"] in feats for d in x["main"]["feedback"]) for x in g)
            k = len(feats & set(DEFAULT_FEATURES))
            chance = f"{100 * chance_named(k):.0f}%" if k else "0% (timing is off in the scorer)"
            L.append(f"| {f} | {len(g)} | {named} | {chance} | {', '.join(sorted(feats))} |")
        unknown = sorted({x["row"]["fault"] for x in faults} - set(FAULT_FEATURES))
        if unknown:
            L.append(f"\nUnknown fault names in clips.csv (not counted): {', '.join(unknown)}")

    sw = sweep(records)
    if sw:
        cur = next(t for t in sw if t[2] == DEFAULT_SWING)
        L += ["", "## Swing thresholds", "", "Each clip holds one stroke: found = at least one swing, "
              "extra = swings beyond the first. Ranked by found - 0.5 x extra.", "",
              "| peak_factor | smooth | min_peak | Found | Extra per clip |", "|---|---|---|---|---|"]
        for found, extra, s in sw[:6]:
            tag = " (current)" if s == DEFAULT_SWING else ""
            L.append(f"| {s['peak_factor']} | {s['smooth']} | {s['min_peak']}{tag} | {100 * found:.0f}% | {extra:.2f} |")
        if cur not in sw[:6]:
            L.append(f"| 3.0 | 5 | 2.0 (current) | {100 * cur[0]:.0f}% | {cur[1]:.2f} |")

    L += ["", "## Per clip", "", "| Clip | Stroke | Form | Fault | Score | Other type | Feedback |", "|---|---|---|---|---|---|---|"]
    for x in records:
        r, mn = x["row"], x["main"]
        if mn is None:
            res = f"failed: {x['result']['failure']['code']}" if x["result"]["failure"] else "-"
            L.append(f"| {r['clip']} | {r['stroke']} | {r['form']} | {r['fault']} | {res} | | |")
        else:
            fb = "; ".join(d["feature"] for d in mn["feedback"])
            L.append(f"| {r['clip']} | {r['stroke']} | {r['form']} | {r['fault']} | {mn['score']:.0f} | "
                     f"{x['other_score'] if x['other_score'] is not None else '-'} | {fb} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clips", type=Path)
    ap.add_argument("--weights", type=Path, default=Path(os.environ.get("PADEL_WEIGHTS", ROOT.parent / "models" / "yolo11m-pose.pt")))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "own_clips")
    ap.add_argument("--overlay", action="store_true", help="also write skeleton overlay videos")
    a = ap.parse_args()
    rows = read_log(a.clips)
    if not rows:
        raise SystemExit("no usable clips")
    if not a.weights.exists() and any(not (a.out / "keypoints" / f"{Path(r['clip']).stem}.npz").exists() for r in rows):
        raise SystemExit(f"pose weights not found at {a.weights}; pass --weights or set PADEL_WEIGHTS")
    refs = {t: load_reference(ROOT / "reference" / f"{t}.npz") for t in STROKES}
    records = analyse(rows, a.clips, a.out, yolo_pose(a.weights) if a.weights.exists() else None, refs, a.overlay)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump([dict(row=x["row"], result=x["result"]) for x in records], open(a.out / "results.json", "w"), indent=1)
    text = summarise(records)
    (a.out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {a.out / 'summary.md'} and results.json")


if __name__ == "__main__":
    main()
