import numpy as np
from padel.tracking import track_players


def person(x, y):
    kp = np.zeros((17, 3)); kp[:, 2] = 2
    kp[11] = (x - 10, y, 2); kp[12] = (x + 10, y, 2)
    return kp


def test_ids_stay_stable_and_survive_a_missed_detection():
    poses = {0: np.stack([person(100, 500), person(900, 500)]),
             1: np.stack([person(905, 502), person(105, 501)]),   # order swapped in the file
             2: np.stack([person(110, 503)]),                      # second player missed
             3: np.stack([person(115, 505), person(915, 505)])}
    tr = track_players(poses, n_players=2)
    assert all(tr[f][0][11, 0] < 200 for f in (0, 1, 2, 3))      # track 0 is always the left player
    assert 1 not in tr[2] and tr[3][1][11, 0] > 800
