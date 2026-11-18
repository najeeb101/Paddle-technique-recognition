import numpy as np
import pytest

from padel.features import FEATURES, as_right_handed, fix_lr_flips, swap_lr, joint_angle, stroke_features, wrap


def col(name):
    return FEATURES.index(name)


def standing(T=3):
    """Upright right-hander seen from behind, right arm straight out to the side. Image y points down."""
    kp = np.zeros((T, 17, 3))
    pts = {
        5: (-1, 0), 6: (1, 0),        # shoulders
        8: (2, 0), 10: (3, 0),        # right elbow, wrist (arm horizontal)
        7: (-1, 1), 9: (-1, 2),       # left arm hanging
        11: (-1, 2), 12: (1, 2),      # hips: torso length 2
        13: (-1, 3), 14: (1, 3),      # knees
        15: (-1, 4), 16: (1, 4),      # ankles: straight legs
        0: (0, -1), 1: (0, -1), 2: (0, -1), 3: (0, -1), 4: (0, -1),
    }
    for j, (x, y) in pts.items():
        kp[:, j] = (x, y, 1.0)
    return kp


def test_joint_angles_straight_and_right():
    b = np.zeros((1, 2))
    assert joint_angle(np.array([[-1.0, 0]]), b, np.array([[1.0, 0]]))[0] == pytest.approx(180)
    assert joint_angle(np.array([[1.0, 0]]), b, np.array([[0, 1.0]]))[0] == pytest.approx(90)


def test_standing_pose():
    f = stroke_features(standing(), "right", fps=30)[0]
    assert f[col("elbow_angle")] == pytest.approx(180)
    assert f[col("shoulder_angle")] == pytest.approx(90)
    assert f[col("knee_angle_dom")] == pytest.approx(180)
    assert f[col("knee_angle_off")] == pytest.approx(180)
    assert f[col("shoulder_rotation")] == pytest.approx(0)
    assert f[col("separation")] == pytest.approx(0)
    assert f[col("torso_lean")] == pytest.approx(0)
    assert f[col("wrist_height")] == pytest.approx(0)


def test_bent_elbow_and_raised_wrist():
    kp = standing()
    kp[:, 10, :2] = (2, -1)  # forearm points straight up from the elbow
    f = stroke_features(kp, "right", fps=30)[0]
    assert f[col("elbow_angle")] == pytest.approx(90)
    assert f[col("wrist_height")] == pytest.approx(0.5)  # 1 unit above the shoulder / torso 2


def test_rotation_and_separation():
    kp = standing()
    kp[:, 6, :2] = (1, -2)  # right shoulder raised: shoulder line at 45 degrees
    kp[:, 5, :2] = (-1, 0)
    f = stroke_features(kp, "right", fps=30)[0]
    assert f[col("shoulder_rotation")] == pytest.approx(45)
    assert f[col("hip_rotation")] == pytest.approx(0)
    assert f[col("separation")] == pytest.approx(45)


def test_wrist_speed_scales_by_torso_and_fps():
    kp = standing(T=3)
    kp[1, 10, 0] += 1.0  # wrist moves 1 unit = half a torso in one frame
    f = stroke_features(kp, "right", fps=30)
    assert np.isnan(f[0, col("wrist_speed")])
    assert f[1, col("wrist_speed")] == pytest.approx(0.5 * 30)


def test_left_hander_mirrors_to_right_hander():
    right = standing()
    left = right.copy()
    left[..., 0] *= -1  # mirror image of the right-hander
    left[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = left[:, [6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]]
    assert np.allclose(stroke_features(left, "left", 30), stroke_features(right, "right", 30), equal_nan=True)


def test_missing_points_give_nan():
    kp = standing()
    kp[:, 10, 2] = 0
    f = stroke_features(kp, "right", fps=30)[0]
    assert np.isnan(f[col("elbow_angle")]) and np.isnan(f[col("wrist_height")])
    assert f[col("knee_angle_dom")] == pytest.approx(180)


def test_bad_hand_and_wrap():
    with pytest.raises(ValueError):
        as_right_handed(standing(), "both")
    assert np.allclose(wrap(np.array([190.0, -190.0, 0.0])), [-170, 170, 0])


def test_fix_lr_flips_restores_swapped_frames():
    kp = standing(8)
    kp[:, 8] += (0, 0, 0)
    flipped = kp.copy()
    flipped[3:6] = swap_lr(kp[3:6])
    fixed, mask = fix_lr_flips(flipped)
    assert mask.tolist() == [False, False, False, True, True, True, False, False]
    assert np.allclose(fixed, kp)


def test_fix_lr_flips_leaves_clean_stroke_alone():
    kp = standing(8)
    fixed, mask = fix_lr_flips(kp)
    assert not mask.any() and np.allclose(fixed, kp)
