import numpy as np
from padel.swings import detect_swings

FPS = 30.0


def bumps(T, peaks, height=10.0, width=4.0, base=0.5):
    """Wrist speed that idles at `base` with a Gaussian bump at each frame in `peaks`."""
    t = np.arange(T)
    s = np.full(T, base)
    for p in peaks:
        s += height * np.exp(-0.5 * ((t - p) / width) ** 2)
    return s


def test_finds_each_swing_at_its_peak():
    swings = detect_swings(bumps(200, [50, 140]), FPS)
    assert [c for _, c, _ in swings] == [50, 140]


def test_window_surrounds_contact():
    (start, contact, end), = detect_swings(bumps(200, [100]), FPS)
    assert start < contact < end
    assert contact - start <= round(0.7 * FPS) and end - contact <= round(0.7 * FPS)


def test_window_is_capped_at_07_seconds_each_side():
    (start, contact, end), = detect_swings(bumps(300, [150], width=40.0), FPS)   # a very slow bump
    assert contact - start == round(0.7 * FPS) and end - contact == round(0.7 * FPS)


def test_no_swing_in_a_still_clip():
    assert detect_swings(np.full(150, 0.5), FPS) == []


def test_peaks_closer_than_08_seconds_count_once():
    s = bumps(200, [80, 95])          # 15 frames = 0.5 s apart
    assert len(detect_swings(s, FPS)) == 1


def test_tiny_movements_are_not_swings():
    # 0.6 torso lengths per second is far above the median here, but too slow to be a racket swing
    assert detect_swings(bumps(150, [75], height=0.6, base=0.05), FPS) == []


def test_missing_speed_values_are_tolerated():
    s = bumps(200, [100])
    s[20:25] = np.nan
    assert [c for _, c, _ in detect_swings(s, FPS)] == [100]
