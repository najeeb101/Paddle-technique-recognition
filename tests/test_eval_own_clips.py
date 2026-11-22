import importlib.util
from pathlib import Path

import cv2
import numpy as np
from padel.scoring import load_reference
from test_analyze import player

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_own_clips", ROOT / "scripts" / "eval_own_clips.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
REFS = {t: load_reference(ROOT / "reference" / f"{t}.npz") for t in ev.STROKES}


def write_clip(path, frames=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for _ in range(frames):
        vw.write(np.zeros((64, 64, 3), np.uint8))
    vw.release()


def make_set(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = ["clip,date,player,hand,stroke,form,fault,camera_height_m,camera_distance_m,notes",
            "a.mp4,2026-11-24,P1,right,forehand,good,,1.6,6,",
            "b.mp4,2026-11-24,P1,right,forehand,fault,no_shoulder_turn,1.6,6,",
            "c.mp4,2026-11-24,P1,right,backhand,good,,1.6,6,",
            "missing.mp4,2026-11-24,P1,right,forehand,good,,1.6,6,",
            "d.mp4,2026-11-24,P1,sideways,forehand,good,,1.6,6,"]
    (clips / "clips.csv").write_text("\n".join(rows) + "\n")
    for c in ("a", "b", "c", "d", "unlogged"):
        write_clip(clips / f"{c}.mp4")
    return clips


def fake_pose(calls):
    def pose(video):
        calls.append(video.name)
        still = video.stem == "c"
        return dict(keypoints=player(150, swing_at=None if still else 75), fps=30.0, warnings=[])
    return pose


def test_bad_rows_and_unlogged_videos_are_skipped(tmp_path, capsys):
    rows = ev.read_log(make_set(tmp_path))
    assert [r["clip"] for r in rows] == ["a.mp4", "b.mp4", "c.mp4"]
    out = capsys.readouterr().out
    assert "missing.mp4: video not found" in out and "d.mp4" in out and "unlogged.mp4: no row" in out


def test_summary_counts_swings_failures_and_faults(tmp_path):
    clips = make_set(tmp_path)
    calls = []
    recs = ev.analyse(ev.read_log(clips), clips, tmp_path / "out", fake_pose(calls), REFS)
    assert [x["main"] is not None for x in recs] == [True, True, False]
    assert recs[0]["other_score"] is not None
    text = ev.summarise(recs)
    assert "| forehand | good | 1 | 1 | 0 | none |" in text
    assert "| backhand | good | 1 | 0 | 0 | no_swing 1 |" in text
    assert "| no_shoulder_turn | 1 |" in text and "## Swing thresholds" in text


def test_pose_is_cached_between_runs(tmp_path):
    clips = make_set(tmp_path)
    calls = []
    rows = ev.read_log(clips)
    ev.analyse(rows, clips, tmp_path / "out", fake_pose(calls), REFS)
    ev.analyse(rows, clips, tmp_path / "out", fake_pose(calls), REFS)
    assert calls == ["a.mp4", "b.mp4", "c.mp4"]


def test_chance_of_naming_a_feature():
    assert abs(ev.chance_named(1) - 3 / 9) < 1e-9
    assert ev.chance_named(9) == 1.0
