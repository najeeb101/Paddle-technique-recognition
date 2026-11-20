"""FastAPI server for the phone app (design.md sections 3.5 and 8). Start it with `py -m padel.server`.

Jobs run one at a time in a background thread, so the pose model is loaded once. Uploaded videos are
deleted after processing (design.md 8.3) unless `keep_uploads` is set.
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
HANDS, STROKES = ("right", "left"), ("forehand", "backhand")
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".3gp"}


def default_analyzer(weights):
    """The real analysis: pose, swings, scores (analyze_clip), then the overlay video."""
    def run(video, hand, stroke_type, job_dir, progress, skeleton_only=False):
        from padel.analyze import analyze_clip
        from padel.overlay import render_overlay

        result, kp = analyze_clip(video, hand, stroke_type, weights, progress=progress)
        if kp is None:
            return result, None
        progress("overlay")
        t0 = time.perf_counter()
        overlay = render_overlay(video, kp, result, job_dir / "overlay.mp4", blank=skeleton_only)
        result.setdefault("timings_s", {})["overlay"] = round(time.perf_counter() - t0, 2)
        return result, overlay
    return run


def create_app(analyzer=None, workdir=None, keep_uploads: bool = False) -> FastAPI:
    """`analyzer(video, hand, stroke_type, job_dir, progress, skeleton_only) -> (result dict, overlay path or None)`."""
    workdir = Path(workdir or ROOT / "data" / "jobs")
    if analyzer is None:
        analyzer = default_analyzer(os.environ.get("PADEL_WEIGHTS", ROOT.parent / "models" / "yolo11m-pose.pt"))
    jobs: dict[str, dict] = {}
    todo: queue.Queue = queue.Queue()
    lock = threading.Lock()

    def update(job, **fields):
        with lock:
            jobs[job].update(fields)

    def worker():
        while (item := todo.get()) is not None:
            job, video, hand, stroke_type, skeleton_only = item
            update(job, state="running", stage="starting")
            try:
                result, overlay = analyzer(video, hand, stroke_type, workdir / job, lambda s: update(job, stage=s),
                                           skeleton_only=skeleton_only)
                outcome = dict(state="done", stage=None, result=result, video=str(overlay) if overlay else None)
            except Exception:  # any crash becomes a readable job error, never a dead server
                traceback.print_exc()  # full details stay in the server window
                outcome = dict(state="error", stage=None, message="The laptop could not read or process this video. "
                               "Try recording it again, or check the server window for details.")
            if not keep_uploads:
                video.unlink(missing_ok=True)
            update(job, **outcome)

    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(target=worker, daemon=True).start()
        yield
        todo.put(None)

    app = FastAPI(title="Padel technique", lifespan=lifespan)

    def get(job: str) -> dict:
        with lock:
            if job not in jobs:
                raise HTTPException(404, "Unknown job")
            return dict(jobs[job])

    @app.get("/")
    def page():
        return FileResponse(STATIC / "index.html")

    @app.post("/analyze")
    def analyze(video: UploadFile = File(...), hand: str = Form(...), stroke_type: str = Form(...),
                skeleton_only: bool = Form(False)):
        if hand not in HANDS:
            raise HTTPException(400, "hand must be right or left")
        if stroke_type not in STROKES:
            raise HTTPException(400, "stroke_type must be forehand or backhand")
        ext = Path(video.filename or "").suffix.lower()
        if not ((video.content_type or "").startswith("video/") or ext in VIDEO_EXT):
            raise HTTPException(400, "Please upload a video file.")
        job = uuid.uuid4().hex[:12]
        (workdir / job).mkdir(parents=True, exist_ok=True)
        path = workdir / job / f"upload{ext or '.mp4'}"
        with open(path, "wb") as fh:
            shutil.copyfileobj(video.file, fh)
        with lock:
            jobs[job] = dict(state="queued", stage=None)
        todo.put((job, path, hand, stroke_type, skeleton_only))
        return {"job_id": job}

    @app.get("/status/{job}")
    def status(job: str):
        j = get(job)
        return {"state": j["state"], "stage": j.get("stage"), "message": j.get("message")}

    @app.get("/result/{job}")
    def result(job: str):
        j = get(job)
        if j["state"] != "done":
            raise HTTPException(409, "Result not ready")
        return j["result"]

    @app.get("/video/{job}")
    def overlay(job: str):
        j = get(job)
        if j["state"] != "done" or not j.get("video"):
            raise HTTPException(404, "No overlay video for this job")
        return FileResponse(j["video"], media_type="video/mp4")

    return app
