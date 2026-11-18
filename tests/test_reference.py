import numpy as np
import pytest

from padel.reference import CONTACT_STEP, N_STEPS, align_to, build_reference, dtw, fill_nan, medoid, resample


def test_dtw_worked_example():
    a = np.array([[0.0], [1.0], [2.0], [2.0]])
    b = np.array([[0.0], [1.0], [1.0], [2.0]])
    cost, path = dtw(a, b, band=None)
    assert cost == pytest.approx(0.0)
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    assert all((i2 - i1, j2 - j1) in {(1, 0), (0, 1), (1, 1)} for (i1, j1), (i2, j2) in zip(path, path[1:]))


def test_dtw_identical_and_shifted():
    a = np.sin(np.linspace(0, 6, 40))[:, None]
    assert dtw(a, a)[0] == pytest.approx(0.0)
    shifted = np.sin(np.linspace(0.4, 6.4, 40))[:, None]
    assert 0 < dtw(a, shifted)[0] < np.abs(a - shifted).sum()


def test_resample_keeps_contact_at_step():
    f = np.zeros((50, 1))
    f[17] = 1.0
    r = resample(f, 17)
    assert r.shape == (N_STEPS, 1)
    assert r[:, 0].argmax() == CONTACT_STEP


def test_fill_nan_interpolates_and_keeps_empty():
    x = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]])
    out = fill_nan(x)
    assert out[1, 0] == pytest.approx(2.0)
    assert np.isnan(out[:, 1]).all()


def test_medoid_picks_central_sequence():
    base = np.linspace(0, 1, 30)[:, None]
    assert medoid([base - 0.5, base, base + 0.5]) == 1


def test_build_reference_recovers_mean_and_floors_std():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, N_STEPS)
    seqs = [np.stack([np.sin(t) + rng.normal(0, 0.05, N_STEPS), np.full(N_STEPS, 3.0)], 1) for _ in range(12)]
    ref = build_reference(seqs)
    assert ref["mean"].shape == (N_STEPS, 2)
    assert np.abs(ref["mean"][:, 0] - np.sin(t)).max() < 0.3
    assert ref["mean"][:, 1] == pytest.approx(3.0)
    assert (ref["std"] > 0).all()


def test_align_to_shape():
    x = np.linspace(0, 3, 30)
    assert align_to(np.sin(x ** 1.1)[:, None], np.sin(x)[:, None]).shape == (30, 1)
