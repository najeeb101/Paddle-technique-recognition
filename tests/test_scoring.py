import numpy as np
import pytest

from padel.features import FEATURES
from padel.reference import N_STEPS
from padel.scoring import DEFAULT_FEATURES, load_reference, phase_of, score_stroke, step_weights


def make_ref():
    F = len(FEATURES)
    t = np.linspace(0, np.pi, N_STEPS)[:, None]
    mean = np.sin(t) * np.arange(1, F + 1)
    return {"mean": mean, "std": np.full((N_STEPS, F), 5.0), "scale": np.full(F, 10.0), "names": list(FEATURES)}


def stroke_from(ref, T=45, contact=30, offset=0.0, col=None):
    src = np.linspace(0, N_STEPS - 1, T)
    feat = np.stack([np.interp(src, np.arange(N_STEPS), ref["mean"][:, j]) for j in range(ref["mean"].shape[1])], 1)
    if col is not None:
        feat[:, FEATURES.index(col)] += offset
    return feat, contact


def test_weights_and_phases():
    w = step_weights()
    assert w[29] == 1 and w[30] == 2 and w[50] == 2 and w[51] == 1
    assert [phase_of(s) for s in (10, 40, 55)] == ["before", "at", "after"]


def test_reference_like_stroke_scores_high():
    ref = make_ref()
    feat, c = stroke_from(ref)
    r = score_stroke(feat, c, ref)
    assert r.score > 90
    assert set(r.feature_scores) == set(DEFAULT_FEATURES)


def test_offset_feature_is_named_with_direction_and_size():
    ref = make_ref()
    feat, c = stroke_from(ref, offset=-30.0, col="elbow_angle")
    r = score_stroke(feat, c, ref)
    assert r.score < score_stroke(*stroke_from(ref), ref).score
    assert r.deviations[0].feature == "elbow_angle"
    assert r.deviations[0].direction == "lower"
    assert r.deviations[0].size == pytest.approx(30.0, abs=3.0)


def test_wrist_speed_not_scored_by_default():
    ref = make_ref()
    feat, c = stroke_from(ref, offset=500.0, col="wrist_speed")
    assert "wrist_speed" not in score_stroke(feat, c, ref).feature_scores


def test_missing_feature_is_skipped():
    ref = make_ref()
    feat, c = stroke_from(ref)
    feat[:, FEATURES.index("elbow_angle")] = np.nan
    r = score_stroke(feat, c, ref)
    assert "elbow_angle" not in r.feature_scores and r.coverage < 1.0


def test_small_offset_lowers_score_continuously():
    ref = make_ref()
    base = score_stroke(*stroke_from(ref), ref).score
    small = score_stroke(*stroke_from(ref, offset=-3.0, col="elbow_angle"), ref).score
    big = score_stroke(*stroke_from(ref, offset=-6.0, col="elbow_angle"), ref).score
    assert base > small > big


def test_worst_mix_weights_the_worst_feature():
    ref = make_ref()
    feat, c = stroke_from(ref, offset=-8.0, col="elbow_angle")
    assert score_stroke(feat, c, ref, worst_mix=1.0).score < score_stroke(feat, c, ref, worst_mix=0.0).score


def test_turn_lead_finds_the_fastest_turn():
    from padel.scoring import timing_stats, turn_lead
    names = list(FEATURES)
    feat = np.zeros((40, len(names)))
    feat[:, names.index("shoulder_rotation")] = 90 * np.tanh(np.arange(40) - 17.0)  # fastest turn at frame 17
    assert turn_lead(feat, 20, names, fps=30.0) == pytest.approx(-3 / 30, abs=1 / 30)
    feat[:, names.index("shoulder_rotation")] = np.nan
    assert np.isnan(turn_lead(feat, 20, names))
    stats = timing_stats([(feat, 20)] * 2 + [(stroke_from(make_ref())[0], 30)], names)
    assert set(stats) == {"timing_mean", "timing_std"}


def test_timing_term_used_only_when_asked():
    ref = make_ref() | {"timing_mean": -0.1, "timing_std": 0.15}
    feat, c = stroke_from(ref)
    assert "timing" not in score_stroke(feat, c, ref).feature_scores
    assert "timing" in score_stroke(feat, c, ref, use_timing=True).feature_scores
