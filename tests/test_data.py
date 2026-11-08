from padel.data import stroke_windows


def test_windows_group_consecutive_same_type():
    shots = [(0, "Serve"), (1, "Serve"), (5, "Forehand"), (6, "Forehand"), (7, "Backhand"), (9, "Backhand")]
    assert stroke_windows(shots) == [("Forehand", 5, 6), ("Backhand", 7, 7), ("Backhand", 9, 9)]
