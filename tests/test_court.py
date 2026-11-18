import numpy as np

from padel.court import feet_point, is_near_side, to_court

# FinalF matrix from the v2 homography file.
H = np.array([
    [0.026098095342145942, 0.014904213484550795, -19.946805713490495],
    [0.0004925894191014476, 0.12221633966441776, -39.13767794560289],
    [2.573098798160646e-05, 0.002983857630570546, 1.0],
])


def test_identity_maps_points_to_themselves():
    pts = np.array([[1.0, 2.0], [3.5, -4.0]])
    assert np.allclose(to_court(np.eye(3), pts), pts)


def test_known_court_points():
    # Reference pairs stored next to H in the file: pixel -> metres.
    px = np.array([[806.67, 351.66], [676.07, 837.38]])
    assert np.allclose(to_court(H, px), [[3.07, 2.05], [2.89, 18.07]], atol=0.3)


def _person(ankle_y, hip_y, ankle_conf=1.0):
    kp = np.zeros((17, 3))
    kp[[11, 12]] = [[900, hip_y, 1], [960, hip_y, 1]]
    kp[[15, 16]] = [[900, ankle_y, ankle_conf], [960, ankle_y, ankle_conf]]
    return kp


def test_feet_falls_back_to_hips():
    assert np.allclose(feet_point(_person(800, 700)), [930, 800])
    assert np.allclose(feet_point(_person(800, 700, ankle_conf=0)), [930, 700])


def test_near_and_far_side():
    assert is_near_side(H, _person(850, 750))
    assert not is_near_side(H, _person(380, 330))
