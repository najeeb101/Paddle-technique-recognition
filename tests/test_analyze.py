import json
from pathlib import Path

import numpy as np
from padel.analyze import analyze_keypoints, feedback_text
from padel.scoring import Deviation, load_reference

REF = load_reference(Path(__file__).resolve().parents[1] / "reference" / "forehand.npz")
FPS = 30.0
STANDING = {0: (500, 180), 1: (490, 170), 2: (510, 170), 3: (480, 175), 4: (520, 175),
            5: (450, 250), 6: (550, 250), 7: (430, 330), 8: (580, 330), 9: (430, 400), 10: (600, 400),
            11: (465, 450), 12: (535, 450), 13: (465, 560), 14: (535, 560), 15: (465, 670), 16: (535, 670)}


def player(T, swing_at=None):
    """Right-handed stick figure seen from behind; with `swing_at`, the right wrist sweeps a fast arc there."""
    kp = np.zeros((T, 17, 3))
    kp[..., 2] = 0.9
    for j, (x, y) in STANDING.items():
        kp[:, j, :2] = (x, y)
    if swing_at is not None:
        kp[:swing_at - 6, 10, :2] = (580, 210)                            # wrist waits at the start of the arc
        for i, f in enumerate(range(swing_at - 6, swing_at + 7)):
            a = np.radians(-90 + 180 * (1 - np.cos(np.pi * i / 12)) / 2)   # eased sweep, fastest mid-swing
            kp[f, 10, :2] = (580 + 120 * np.cos(a), 330 + 120 * np.sin(a))
        kp[swing_at + 7:, 10, :2] = kp[swing_at + 6, 10, :2]
    return kp


def test_a_swing_gets_a_score_and_three_feedback_lines():
    r = analyze_keypoints(player(150, swing_at=75), FPS, "right", "forehand", REF, clip="demo.mp4")
    assert r["failure"] is None and r["n_swings_found"] == 1
    s = r["strokes"][0]
    assert s["start_frame"] < s["contact_frame"] < s["end_frame"]
    assert abs(s["contact_frame"] - 75) <= 1                            # contact = fastest point of the sweep
    assert 0 <= s["score"] <= 100 and 0 <= s["confidence"] <= 1
    assert len(s["feedback"]) == 3 and all(f["text"] for f in s["feedback"])
    json.dumps(r)                                                       # must be plain JSON for the server


def test_no_person_is_a_failure_not_a_score():
    kp = np.full((90, 17, 3), np.nan)
    kp[..., 2] = 0
    r = analyze_keypoints(kp, FPS, "right", "forehand", REF)
    assert r["failure"]["code"] == "no_person" and r["strokes"] == []


def test_standing_still_finds_no_swing():
    r = analyze_keypoints(player(90), FPS, "right", "forehand", REF)
    assert r["failure"]["code"] == "no_swing"


def test_long_clips_are_refused():
    r = analyze_keypoints(player(21 * 30), FPS, "right", "forehand", REF)
    assert r["failure"]["code"] == "clip_too_long"


def test_missing_racket_arm_means_low_pose_quality():
    kp = player(150, swing_at=75)
    kp[:, 8] = (np.nan, np.nan, 0)                                      # right elbow never seen
    r = analyze_keypoints(kp, FPS, "right", "forehand", REF)
    assert r["failure"]["code"] == "low_pose_quality"


def test_analyze_clip_reports_stages_and_their_timings(tmp_path):
    import cv2
    from padel.analyze import analyze_clip

    video = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for _ in range(10):
        vw.write(np.zeros((64, 64, 3), np.uint8))
    vw.release()
    fake_pose = lambda v, w: dict(keypoints=player(150, swing_at=75), fps=30.0, warnings=[])
    stages = []
    result, kp = analyze_clip(video, "right", "forehand", "unused.pt", progress=stages.append, pose=fake_pose)
    assert stages == ["pose", "analysis"]
    assert result["failure"] is None and kp.shape == (150, 17, 3)
    assert set(result["timings_s"]) == {"pose", "analysis"} and all(t >= 0 for t in result["timings_s"].values())


def test_feedback_text_says_bent_or_straight_for_joints():
    assert "more bent" in feedback_text(Deviation("elbow_angle", "at", "lower", 18.4, -2.5))
    assert "straighter" in feedback_text(Deviation("knee_angle_dom", "before", "higher", 12.0, 2.2))
    assert "18 degrees" in feedback_text(Deviation("elbow_angle", "at", "lower", 18.4, -2.5))
