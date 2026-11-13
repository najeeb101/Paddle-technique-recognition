import numpy as np
from padel.pose import clean_keypoints, extract_keypoints, largest_person, smooth_keypoints


def walk(T, conf=0.9):
    """(T,17,3) keypoints where every joint moves right by 10 px per frame."""
    kp = np.zeros((T, 17, 3))
    kp[:, :, 0] = np.arange(T)[:, None] * 10.0
    kp[:, :, 1] = 500.0
    kp[:, :, 2] = conf
    return kp


def test_low_confidence_points_become_missing():
    kp = walk(10)
    kp[4:9, 9, 2] = 0.2                                   # left wrist unsure for 5 frames (too long to fill)
    out = clean_keypoints(kp)
    assert np.isnan(out[4:9, 9, :2]).all() and (out[4:9, 9, 2] == 0).all()
    assert np.isfinite(out[:, 10, :2]).all()             # other joints untouched


def test_short_gaps_are_filled_linearly():
    kp = walk(10)
    kp[3:6, 9, 2] = 0.1                                   # 3 missing frames
    out = clean_keypoints(kp)
    assert np.allclose(out[3:6, 9, 0], [30, 40, 50])


def test_long_gaps_and_edges_stay_missing():
    kp = walk(12)
    kp[3:7, 9, 2] = 0.1                                   # 4 missing frames: too long
    kp[0, 10, 2] = 0.1                                    # missing at the start: nothing to interpolate from
    out = clean_keypoints(kp)
    assert np.isnan(out[3:7, 9, 0]).all()
    assert np.isnan(out[0, 10, 0])


def test_smoothing_keeps_a_steady_motion():
    kp = walk(20)
    assert np.allclose(smooth_keypoints(kp)[:, :, :2], kp[:, :, :2])


def test_smoothing_reduces_jitter():
    kp = walk(30)
    rng = np.random.default_rng(0)
    noisy = kp.copy()
    noisy[:, :, :2] += rng.normal(0, 3, size=(30, 17, 2))
    before = np.abs(noisy[:, :, :2] - kp[:, :, :2]).mean()
    after = np.abs(smooth_keypoints(noisy)[:, :, :2] - kp[:, :, :2]).mean()
    assert after < 0.75 * before


def test_smoothing_leaves_missing_frames_missing():
    kp = walk(20)
    kp[8:12, 9, :2] = np.nan
    out = smooth_keypoints(kp)
    assert np.isnan(out[8:12, 9, 0]).all() and np.isfinite(out[:8, 9, 0]).all()


def test_largest_person_is_picked_by_box_area():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 50, 40], [0, 0, 20, 20]], float)
    assert largest_person(boxes) == 1
    assert largest_person(np.zeros((0, 4))) is None


def test_extract_keeps_the_largest_person_and_warns():
    big, small = walk(1, 0.9)[0], walk(1, 0.8)[0] + 1000
    detections = [
        (np.array([[0, 0, 100, 200]], float), big[None]),                                  # one person
        (np.array([[0, 0, 20, 40], [0, 0, 100, 200]], float), np.stack([small, big])),     # two people
        (np.zeros((0, 4)), np.zeros((0, 17, 3))),                                          # nobody
    ]
    kp, warnings = extract_keypoints(range(3), lambda i: detections[i])
    assert kp.shape == (3, 17, 3) and kp.dtype == np.float32
    assert np.allclose(kp[1], big)
    assert np.isnan(kp[2, :, :2]).all() and (kp[2, :, 2] == 0).all()
    assert any("more than one person" in w for w in warnings)
    assert any("no person" in w for w in warnings)
