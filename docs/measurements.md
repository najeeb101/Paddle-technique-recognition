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
| Near-side by court homography | 355 (194 forehand, 161 backhand); old hip-y rule 231 |
| Handedness labels | 311 R, 31 L, 13 unclear; rules scored 60% and 64% against them |
| Strokes with features | 342 (187 forehand, 155 backhand) |
| Strokes with a left/right swap corrected | 127 of 342; 9 strokes (10 frames) still show a hip flip |
| Missing share per feature | elbow angle 11.8%, wrist height 10.5%, wrist speed 17.4%, shoulder angle 4.0%, others under 2% |
| Reference | 174 forehand and 155 backhand strokes, 60 steps, contact at step 40; build time about 50 s |
| Mean std per feature, forehand | elbow 38, shoulder angle 33, knee 26-27, shoulder rotation 25, hip rotation 18, separation 21, torso lean 21 degrees; wrist height 0.4 torso lengths |
| Hold-out (seed 0, 80/20) | forehand 94.0 vs 88.6 on the wrong type (80% correct ordering); backhand 94.9 vs 92.7 (61%) |
| Tests | 37 passing |

Not yet measured: hitter accuracy, contact error in frames, keypoint availability on phone clips, pose-to-score latency.
