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
