# Why YOLO11-pose and COCO-17

Why the pose model is YOLO11-pose with COCO-17 keypoints, what else was considered, and what is
measured versus not yet proven. Written 2026-11-20 for the instructor's "explain your model choice" request
and the report. Numbers are in [measurements.md](measurements.md).

## 1. What the pose model has to do here

| Need | Why |
|---|---|
| Output COCO-17 keypoints | The professional reference (PadelTracker100) is in COCO-17, so phone clips must be too, or every format difference would look like a technique difference |
| Find several people per frame | Broadcast frames have four players; phone clips can have a partner or a coach in view |
| Work on small, distant players | Players are about 250 px tall in the broadcast view |
| Run on a laptop in about a minute per 10 s clip | Plan target: result within 60 s of upload |
| Python, simple install, instructor-permitted | The instructor advised MediaPipe or something similar |

## 2. Why COCO-17 (the keypoint format)

COCO-17 is the 17-point body layout of the public COCO Keypoints dataset (nose, eyes, ears, shoulders,
elbows, wrists, hips, knees, ankles). Most pose models are trained on it, PadelTracker100 uses it, and YOLO11-pose
outputs it, so the same feature code runs on professional and phone data with no conversion. It has no
racket, hand or foot detail, which limits what can be measured (for example wrist angle).

## 3. Models considered

| Model | How it works | Published accuracy | For this project |
|---|---|---|---|
| **YOLO11-pose** (Ultralytics) | One stage: finds every person and their 17 keypoints in a single pass | COCO Keypoints val2017 mAP50-95: n 50.0, s 58.9, m 64.9 | COCO-17, multi-person, `pip install ultralytics`, found the players in our benchmark (below). **Chosen** |
| MediaPipe Pose Landmarker (Google, BlazePose) | Detect a person, then 33 landmarks; `num_poses` defaults to 1 | Not reported on COCO in the same way | 33 landmarks need mapping to COCO-17; found almost nobody in the broadcast benchmark. Still worth testing on close-up phone clips |
| ViTPose | Top-down: a separate person detector, then a vision transformer per person | 80.9 AP on COCO test-dev (largest model, single model) | Most accurate, and the dataset authors used it to make the PadelTracker100 poses. Heavy, needs a separate detector and a more complex install |
| RTMPose (MMPose) | Top-down, built for real-time use | 75.8 AP on COCO (RTMPose-m), 90+ FPS on an i7-11700 CPU per its paper | Strong speed and accuracy, but needs MMPose plus a separate detector; not tried |
| OpenPose | Bottom-up: finds all joints, then groups them into people | Older (2017) | A study on real athletics footage judged it unsuitable (see [literature-review.md](literature-review.md)); not tried |

Accuracy numbers come from each project's own paper or page and use different test setups (top-down
models are usually scored with given person boxes), so they are not directly comparable. They show the
general picture: top-down models are more accurate per person, while one-stage YOLO is simpler and handles
many people at once.

## 4. What we measured

**YOLO11m-pose vs MediaPipe heavy** (broadcast sample clip, 60 frames, CPU; `scripts/bench_pose.py`):
YOLO found 2.02 people per frame (both near-side players), MediaPipe 0.30. YOLO 0.227 s per frame, MediaPipe 0.140.

**YOLO11 sizes on CPU and GPU** (1920x1080 test clip, 60 frames; `scripts/bench_sizes.py`):

| Device | n | s | m |
|---|---|---|---|
| CPU (laptop, AMD64 Family 23 Model 96, 12 threads) | 0.064 s/frame, 19 s per 10 s clip | 0.114 s/frame, 34 s | 0.224 s/frame, 67 s |
| GPU (NVIDIA GTX 1650 Ti) | 0.034 s/frame, 10 s | 0.037 s/frame, 11 s | 0.030 s/frame, 9 s |

The rest of the phone pipeline is small: cleaning and smoothing under 0.1 ms per frame, analysis 0.07 to
0.09 s per 300-frame clip, overlay video about 2 s.

## 5. Decision

- **YOLO11-pose**, because it is the option that met every need in section 1 in practice: COCO-17, several
  people at once, found the small broadcast players, one simple install, and fast enough on a laptop.
- **Medium (m) size by default.** It is the most accurate of the three sizes (64.9 vs 58.9 vs 50.0). On a
  GPU all three sizes take about the same time, so there is no reason to give up accuracy. On a CPU-only
  laptop medium misses the 60 s target (67 s for a 10 s clip); small (34 s) or nano (19 s) fit, at lower
  published accuracy. The weights file is chosen with the `PADEL_WEIGHTS` environment variable.
- **Not ViTPose**, even though the dataset poses came from it: more setup and much heavier for a phone-app
  demo. The price is a possible mismatch between dataset poses (ViTPose) and phone poses (YOLO); plan day 9
  measures it.

## 6. Not proven yet

- No accuracy check on our own data: nobody has compared YOLO keypoints with hand-marked or dataset
  keypoints on padel footage (plan day 9).
- The size and device timings use a still image repeated as video; a real swing clip may differ slightly.
  GPU timings varied between runs (an earlier run gave 0.019, 0.020 and 0.030 s per frame).
- Lower published accuracy for nano and small has not been checked on padel strokes.
- MediaPipe is untested on close-up phone clips, where it may do better than on broadcast footage.
- Ultralytics has released a newer YOLO26; it has not been tried.
- YOLO11 is AGPL-3.0 licensed (or an Enterprise licence), which is fine for a course project but matters
  for any product.

## Sources

- Ultralytics YOLO11 model page (licence: "YOLO11 models are provided under AGPL-3.0 and Enterprise licenses"): https://docs.ultralytics.com/models/yolo11/
- YOLO11 pose accuracy table (Ultralytics Hugging Face README): https://huggingface.co/Ultralytics/YOLO11
- ViTPose paper: https://arxiv.org/abs/2204.12484
- RTMPose paper: https://arxiv.org/abs/2303.07399
- MediaPipe Pose Landmarker guide: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- OpenPose (part affinity fields) paper: https://arxiv.org/abs/1611.08050
