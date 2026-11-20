import numpy as np
import pytest
from padel.faults import FAULTS, apply_fault, auc
from padel.features import FEATURES
from padel.scoring import score_stroke

COL = {n: i for i, n in enumerate(FEATURES)}


def stroke(T=60):
    rng = np.random.default_rng(1)
    return np.cumsum(rng.normal(0, 1, size=(T, len(FEATURES))), axis=0) + 50.0


def test_elbow_fault_bends_the_elbow_by_20_degrees_and_nothing_else():
    f = stroke()
    g, c = apply_fault(f, 40, FEATURES, "elbow")
    assert np.allclose(g[:, COL["elbow_angle"]], f[:, COL["elbow_angle"]] - 20)
    others = [i for i in range(len(FEATURES)) if i != COL["elbow_angle"]]
    assert np.allclose(g[:, others], f[:, others]) and c == 40


def test_size_multiplies_the_fault():
    f = stroke()
    g, _ = apply_fault(f, 40, FEATURES, "wrist_height", size=2.0)
    assert np.allclose(g[:, COL["wrist_height"]], f[:, COL["wrist_height"]] - 0.6)


def test_rotation_fault_keeps_separation_consistent():
    f = stroke()
    g, _ = apply_fault(f, 40, FEATURES, "rotation")
    assert np.allclose(g[:, COL["shoulder_rotation"]], f[:, COL["shoulder_rotation"]] - 20)
    assert np.allclose(g[:, COL["hip_rotation"]], f[:, COL["hip_rotation"]])
    sep = (f[:, COL["separation"]] - 20 + 180) % 360 - 180
    assert np.allclose(g[:, COL["separation"]], sep)


def test_timing_fault_moves_contact_and_stays_inside_the_stroke():
    f = stroke()
    g, c = apply_fault(f, 40, FEATURES, "timing")
    assert c == 37 and np.array_equal(g, f)
    _, c = apply_fault(f, 2, FEATURES, "timing")
    assert c == 1


def test_original_stroke_is_not_changed():
    f = stroke()
    before = f.copy()
    apply_fault(f, 40, FEATURES, "knee")
    assert np.array_equal(f, before)


def test_unknown_fault_is_an_error():
    with pytest.raises(ValueError):
        apply_fault(stroke(), 40, FEATURES, "racket_thrown")


def test_auc_counts_how_often_clean_beats_flawed():
    assert auc([90, 95], [10, 20]) == 1.0
    assert auc([10, 20], [90, 95]) == 0.0
    assert auc([50, 50], [50, 50]) == 0.5
    assert auc([3, 1], [2]) == 0.5


def test_a_flawed_copy_scores_lower_and_its_feature_is_named():
    mean = stroke()
    ref = {"mean": mean, "std": np.ones_like(mean), "scale": np.ones(len(FEATURES)), "names": list(FEATURES)}
    clean = score_stroke(mean, 40, ref)
    g, c = apply_fault(mean, 40, FEATURES, "elbow")
    flawed = score_stroke(g, c, ref)
    assert flawed.score < clean.score
    assert flawed.deviations[0].feature == FAULTS["elbow"].feature
