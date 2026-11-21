# Measurements

## Pose model comparison (2026-11-08)

Script: `scripts/bench_pose.py`. Clip: PadelTracker100 `FinalF` sample (1920x1080, 30 fps), first 60 frames.
Machine: Windows laptop, CPU only (no CUDA), Python 3.14, ultralytics 8.4.208, mediapipe 1.2.2.

| Model | s/frame | fps | People found per frame (mean, min-max) | Mean keypoint confidence |
|---|---|---|---|---|
| YOLO11m-pose (COCO-17) | 0.227 | 4.4 | 2.02 (2-3) | 0.72 |
| MediaPipe Pose Landmarker heavy (33 landmarks, num_poses=4) | 0.140 | 7.2 | 0.30 (0-2) | 0.69 |

Seen on frame 20: YOLO skeletonised both near-side players and missed the far-side pair. MediaPipe
found nobody on that frame. The players are small in this broadcast view (about 250 px tall).

Caveats: 60 frames only; broadcast footage, not a phone clip (a phone clip should show the player much
larger, which may favour MediaPipe, untested); no accuracy comparison against the dataset's ViTPose labels yet.

Decision: use YOLO11-pose as the main model for the pro footage. Re-test MediaPipe on the team's phone clips.

## Pipeline numbers (2026-11-18)

| Item | Value |
|---|---|
| Stroke windows | 575 (293 forehand, 282 backhand) |
| Near-side by court homography | 269 with the ball hitter rule; 355 (194 FH, 161 BH) with the old wrist rule; old hip-y rule 231 |
| Handedness labels | 311 R, 31 L, 13 unclear; rules scored 60% and 64% against them |
| Strokes with features | 259 (131 forehand, 128 backhand); 342 with the old hitter rule |
| Strokes with a left/right swap corrected | 127 of 342; 9 strokes (10 frames) still show a hip flip |
| Missing share per feature | elbow angle 11.8%, wrist height 10.5%, wrist speed 17.4%, shoulder angle 4.0%, others under 2% |
| Reference | 174 forehand and 155 backhand strokes, 60 steps, contact at step 40; build time about 50 s |
| Mean std per feature, forehand | elbow 38, shoulder angle 33, knee 26-27, shoulder rotation 25, hip rotation 18, separation 21, torso lean 21 degrees; wrist height 0.4 torso lengths |
| Hold-out (seed 0, 80/20) | forehand 94.0 vs 88.6 on the wrong type (80% correct ordering); backhand 94.9 vs 92.7 (61%) |
| Tests | 37 passing |

Not yet measured: hitter accuracy, contact error in frames, keypoint availability on phone clips, pose-to-score latency.

## YOLO11 sizes and phone pipeline stages (2026-11-20)

Script: `scripts/bench_sizes.py`. Clip: the Ultralytics sample photo `zidane.jpg` scaled to 1920x1080 and repeated
for 60 frames (no real padel clip was available; YOLO's work per frame barely depends on the content, so this
measures speed, not accuracy). Machine: Windows laptop, CPU AMD64 Family 23 Model 96 (12 threads), GPU NVIDIA
GeForce GTX 1650 Ti (4 GB); Python 3.13, ultralytics 8.3.177, torch 2.14.1+cu126. One warm-up frame not timed.

| Device | YOLO11n-pose | YOLO11s-pose | YOLO11m-pose |
|---|---|---|---|
| CPU | 0.064 s/frame (15.7 fps) | 0.114 s/frame (8.8 fps) | 0.224 s/frame (4.5 fps) |
| GPU | 0.034 s/frame (29.3 fps) | 0.037 s/frame (26.8 fps) | 0.030 s/frame (33.6 fps) |
| 10 s clip at 30 fps, CPU | 19 s | 34 s | 67 s |
| 10 s clip at 30 fps, GPU | 10 s | 11 s | 9 s |

- The CPU medium figure (0.224) matches the 2026-11-08 benchmark on a different laptop (0.227).
- GPU timings vary between runs: an earlier run on the same machine gave 0.019, 0.020 and 0.030 s per frame. On the GPU the three sizes take about the same time, so model size is not the bottleneck there.
- Ultralytics uses a CUDA GPU automatically when PyTorch sees one; `yolo_detector(weights, device="cpu")` forces the CPU.
- Other stages: cleaning and smoothing 0.05 ms per frame; analysis (left/right fix, features, swings, scoring) 0.07 to 0.09 s for a 300-frame synthetic clip with one swing; overlay drawing plus H.264 encoding 1.7 to 1.9 s for 60 frames of 1080p.
- Reading: pose is almost all of the latency. On CPU only, YOLO11m misses the 60 s target for a 10 s clip; s or n fit, at lower published accuracy (COCO pose mAP50-95: n 50.0, s 58.9, m 64.9). See [model-choice.md](model-choice.md).
- Not measured yet: the same numbers on a real phone clip, and upload time over Wi-Fi.
