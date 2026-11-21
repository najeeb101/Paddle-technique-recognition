import numpy as np
from padel.strokes import analyse_window_ball, ball_turn


def player(x, T=20, swing_wrist=None):
    kp = np.zeros((T, 17, 3))
    kp[..., 2] = 1.0
    kp[:, 5, :2], kp[:, 6, :2] = (x - 10, 100), (x + 10, 100)
    kp[:, 11, :2], kp[:, 12, :2] = (x - 10, 200), (x + 10, 200)
    kp[:, 9, :2], kp[:, 10, :2] = (x - 20, 150), (x + 20, 150)
    if swing_wrist is not None:
        kp[:, swing_wrist, 0] += np.sin(np.linspace(0, np.pi, T)) * 40
    return kp


def tracks_of(players):
    return {f: {tid: kp[f] for tid, kp in players.items()} for f in range(20)}


def test_ball_turn_finds_the_bounce():
    ball = {f: np.array([f * 10.0, 100.0]) for f in range(10)}
    ball.update({f: np.array([90.0 - (f - 9) * 10.0, 100.0]) for f in range(10, 20)})
    assert ball_turn(ball, range(20)) == 9


def test_no_turn_without_neighbours():
    assert ball_turn({5: np.zeros(2)}, range(20)) is None


def test_hitter_is_nearest_to_ball_not_fastest():
    still = player(100, swing_wrist=10)   # fast wrist, far from the ball
    still[:, 10, 0] += np.r_[np.zeros(10), [300], np.zeros(9)]  # one keypoint jump
    hitter = player(500)
    ball = {f: np.array([300.0 + f * 18, 150.0]) for f in range(11)}  # reaches x=480 at frame 10
    ball.update({f: np.array([480.0 - (f - 10) * 18, 150.0]) for f in range(11, 20)})  # and turns back there
    a = analyse_window_ball(tracks_of({0: still, 1: hitter}), ball, 2, 17)
    assert a["hitter"] == 1 and a["hand"] == "left" and a["rule"] == "ball" and a["turn"] == 10


def test_falls_back_to_speed_without_ball():
    tr = tracks_of({0: player(100), 1: player(500, swing_wrist=10)})
    a = analyse_window_ball(tr, {}, 2, 17)
    assert a["hitter"] == 1 and a["rule"] == "speed"
