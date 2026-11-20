import time

from fastapi.testclient import TestClient
from padel.server.app import create_app

RESULT = {"clip": "c.mp4", "hand": "right", "stroke_type": "forehand", "strokes": [], "warnings": [], "failure": None}


def fake_analyzer(video, hand, stroke_type, job_dir, progress, skeleton_only=False):
    progress("pose")
    out = job_dir / "overlay.mp4"
    out.write_bytes(b"fake mp4")
    return {**RESULT, "hand": hand, "stroke_type": stroke_type, "skeleton_only": skeleton_only}, out


def broken_analyzer(video, hand, stroke_type, job_dir, progress, skeleton_only=False):
    raise RuntimeError(f"cannot open video {video}")


def client(tmp_path, analyzer=fake_analyzer):
    return TestClient(create_app(analyzer=analyzer, workdir=tmp_path))


def upload(c, hand="right", stroke_type="forehand", name="clip.mp4", ctype="video/mp4"):
    return c.post("/analyze", data={"hand": hand, "stroke_type": stroke_type}, files={"video": (name, b"\x00" * 100, ctype)})


def wait_until_finished(c, job):
    for _ in range(200):
        s = c.get(f"/status/{job}").json()
        if s["state"] in ("done", "error"):
            return s
        time.sleep(0.02)
    raise AssertionError("job never finished")


def test_upload_runs_a_job_and_serves_result_and_video(tmp_path):
    with client(tmp_path) as c:
        r = upload(c, hand="left", stroke_type="backhand")
        assert r.status_code == 200
        job = r.json()["job_id"]
        assert wait_until_finished(c, job)["state"] == "done"
        res = c.get(f"/result/{job}").json()
        assert res["hand"] == "left" and res["stroke_type"] == "backhand"
        assert c.get(f"/video/{job}").content == b"fake mp4"


def test_skeleton_only_choice_reaches_the_analyzer(tmp_path):
    with client(tmp_path) as c:
        on = c.post("/analyze", data={"hand": "right", "stroke_type": "forehand", "skeleton_only": "true"},
                    files={"video": ("clip.mp4", b"\x00" * 100, "video/mp4")}).json()["job_id"]
        off = upload(c).json()["job_id"]
        wait_until_finished(c, on), wait_until_finished(c, off)
        assert c.get(f"/result/{on}").json()["skeleton_only"] is True
        assert c.get(f"/result/{off}").json()["skeleton_only"] is False


def test_uploaded_video_is_deleted_after_processing(tmp_path):
    with client(tmp_path) as c:
        job = upload(c).json()["job_id"]
        wait_until_finished(c, job)
        assert not list(tmp_path.rglob("upload*"))


def test_analyzer_crash_becomes_a_readable_error(tmp_path):
    with client(tmp_path, broken_analyzer) as c:
        job = upload(c).json()["job_id"]
        s = wait_until_finished(c, job)
        assert s["state"] == "error" and "could not read or process" in s["message"]
        assert str(tmp_path) not in s["message"]                          # no laptop paths shown to the player


def test_bad_inputs_are_refused(tmp_path):
    with client(tmp_path) as c:
        assert upload(c, hand="middle").status_code == 400
        assert upload(c, stroke_type="smash").status_code == 400
        assert upload(c, name="notes.txt", ctype="text/plain").status_code == 400


def test_unknown_job_is_404(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/status/nope").status_code == 404
        assert c.get("/result/nope").status_code == 404


def test_phone_page_is_served(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/")
        assert r.status_code == 200 and "<html" in r.text.lower()
