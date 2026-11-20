import shutil

import cv2
import numpy as np
import pytest
from padel.overlay import draw_frame, render_overlay


def person():
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    kp[:, 0] = np.linspace(40, 120, 17)
    kp[:, 1] = np.linspace(20, 140, 17)
    return kp


def test_skeleton_is_drawn_where_the_joints_are():
    img = np.zeros((160, 160, 3), np.uint8)
    out = draw_frame(img, person(), label=None)
    x, y = person()[9, :2].astype(int)
    assert out[y, x].any() and not img.any()          # joint pixel coloured; input left untouched


def test_missing_joints_are_skipped():
    kp = person()
    kp[:, :2] = np.nan
    kp[:, 2] = 0
    out = draw_frame(np.zeros((160, 160, 3), np.uint8), kp, label=None)
    assert not out.any()


def test_blank_mode_hides_the_video():
    img = np.full((160, 160, 3), 200, np.uint8)
    out = draw_frame(img, person(), label=None, blank=True)
    assert out[5, 5].sum() == 0                        # background is black, not the video


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_overlay_video_is_h264_with_every_frame(tmp_path):
    src = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 160))
    for _ in range(12):
        vw.write(np.full((160, 160, 3), 90, np.uint8))
    vw.release()
    kp = np.repeat(person()[None], 12, axis=0)
    result = {"stroke_type": "forehand", "strokes": [{"start_frame": 3, "contact_frame": 6, "end_frame": 9, "score": 81.2}]}
    out = render_overlay(src, kp, result, tmp_path / "overlay.mp4")
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12 and fourcc in ("avc1", "h264", "H264")
