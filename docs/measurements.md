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
