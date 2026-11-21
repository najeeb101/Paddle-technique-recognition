# Pipeline guide

How the code turns the PadelTracker100 labels into a reference and a score, step by step. Status as of 2026-11-18. Design intent is in [design.md](design.md); numbers are in [measurements.md](measurements.md) and [results.md](results.md).

Setup (Windows, Python 3.14 via the `py` launcher): `py -m pip install numpy opencv-python pytest scipy ultralytics mediapipe matplotlib fastapi uvicorn python-multipart`, plus ffmpeg on the PATH for the H.264 overlay. The dataset lives outside the repo (`C:\Users\ACER\datasets\padeltracker100\extracted`, labels in `labels/`); `data/` is git-ignored. Tests: `py -m pytest -q` (62 pass).

## Steps in run order

| # | Script | Input | Output | What it does |
|---|---|---|---|---|
| 1 | `build_strokes_index.py` | pose JSONs and shot CSVs | `strokes_index.csv` | Prints row and frame counts, groups consecutive labelled frames of one type into stroke windows (575: 293 forehand, 282 backhand). |
| 2 | `infer_hitters.py` | index, poses | `strokes_hitters.csv` | Tracks the four players, picks the hitter per window (the tracked player whose faster wrist has the highest torso-normalised peak speed) and the contact frame (that peak). |
| 3 | `finalize_index.py` | hitters csv, v2 homography dir | `data/strokes_index_full.csv` | Adds `near_side` from the court homography (355 near-side). It also writes a rule-based `handed`, which step 5 replaces. |
| 4 | `make_hand_sheets.py` | video, index | contact sheets plus a CSV to fill in | Four frames per window (contact -6, -3, 0, +3), no keypoints drawn, so the racket hand is read from the racket alone. |
| 5 | `apply_hand_labels.py` | index, filled CSV | updated index | Replaces `handed` with the eye label; the old value stays as `handed_rule`. `?` windows get an empty `handed` and are skipped later. |
| 6 | `build_stroke_features.py` | labels dir, index | `data/processed/stroke_features.npz`, curve plots | Per stroke: window plus 10 frames each side, NaN masking, left/right fix (`fix_lr_all`), features. Prints how many strokes had swaps corrected. 342 strokes. |
| 7 | `build_reference.py` | features npz | `reference/forehand.npz`, `backhand.npz` | Resamples, runs DTW to the medoid, averages along the paths. Excludes the 13 strokes in `docs/label-suspects.md`. |
| 8 | `eval_holdout.py` | features npz, optional seed | printed scores | 80/20 split per type, own-type against wrong-type reference. |
| 9b | `tune_scoring.py` | features npz, `--seed` | markdown table | Grid search of the scoring settings on a validation split of the training strokes (the hold-out is never used). |
| 9 | `eval_flawed.py` | features npz, `--seed`, optional `--train-match FinalF` or `FinalM` | markdown tables | Flawed-stroke test: rebuilds references from training strokes only, then scores each held-out stroke and five flawed copies (elbow, knee, rotation, wrist height, timing; `padel/faults.py`) at sizes x0.5 to x4. Reports score drop, AUC and whether the feedback names the changed feature. `--train-match` gives a leave-match-out split. |

Other scripts: `bench_pose.py` (YOLO11-pose vs MediaPipe), `check_lr.py` (draws keypoints by their names to settle left/right), `check_tracking.py`, `check_hitters.py`, `make_check_sheets.py` (hand-check sheets), `handedness.py` (old rule-based vote, not used), `clip_to_keypoints.py` (phone clip to cleaned keypoints), `bench_sizes.py` (YOLO11 n/s/m speed on CPU and GPU, plus the other pipeline stages), `score_check.py`. Each script's docstring gives its arguments.

Gotchas: use `py`, not `python`; Python needs `C:/...` style paths where Git Bash shows `/c/...`; step 6 and 7 take tens of seconds to minutes, so run them in the background.

## Data model

- Keypoints: `(T,17,3)` COCO-17, columns x, y, confidence. Confidence 0 or NaN means missing.
- Stroke window: start, contact, end frames, match, type (Forehand or Backhand), hitter track, handedness.
- Features file: per stroke id (for example `FinalF_000045`), `<id>/kp` and `<id>/feat` `(T,10)`, plus `meta` rows (id, match, type, hitter, handed, start, contact, end, first_frame) and `feature_names`.
- Reference file: `mean` and `std` `(60,10)`, `scale` `(10,)`, `feature_names`, `n_strokes`, `medoid`, `matches`, `contact_step` (40), `n_steps` (60), `stroke_ids`.

## Features (padel/features.py)

Computed per frame after the stroke is made right-handed: a left-handed stroke is mirrored (x to -x, left and right joints swapped) so one reference serves both hands. Angles are 2D image-plane angles in degrees.

| Feature | Definition |
|---|---|
| elbow_angle | angle at the racket elbow between shoulder and wrist |
| shoulder_angle | angle at the racket shoulder between hip and elbow |
| knee_angle_dom / knee_angle_off | angle at each knee between hip and ankle (racket side, other side) |
| shoulder_rotation | image-plane direction of the left-to-right shoulder line, unwrapped over time |
| hip_rotation | same for the hip line |
| separation | shoulder rotation minus hip rotation, wrapped to [-180, 180) |
| torso_lean | angle of the hip-to-shoulder midline from vertical |
| wrist_height | (shoulder y - wrist y) divided by torso length, racket side |
| wrist_speed | racket wrist displacement per frame divided by torso length, times fps |

Normalising by torso length makes distances scale-free. 2D angles depend on camera position, so a very different camera view is not comparable (known limit).

## Left/right swap fix

Found 2026-11-18. The pose model swaps left and right labels for stretches inside a stroke. At first this looked like about 18 strokes with flipped shoulders; counting properly gave 112 of 342, and a second pass found that the hips and legs flip independently of the shoulders (94 strokes, 374 frames).

Method (`fix_lr_flips`, `fix_lr_all`): a two-state Viterbi over the frames of a stroke. State 1 means "this frame's sides are swapped". The cost of a frame is how far the paired joints move from the last valid frame, in torso lengths, under the chosen labelling, plus a weak prior (weight 0.5) that the shoulder line points image-right, as it does for a player seen from behind. Pass 1 uses the shoulders as the anchor and swaps all joints; pass 2 keys on the hips and swaps only hips, knees and ankles. Result: 127 of 342 strokes were changed; 9 strokes (10 frames) still show a hip flip and one stroke (`FinalF_015744`) a residual shoulder flip. The hip rotation band of the reference fell from about plus or minus 75 degrees to about plus or minus 17. Unit tests cover: swapped frames restored, a clean stroke left alone, hips flipped alone repaired.

## Reference building (padel/reference.py)

1. Fill NaNs by linear interpolation per feature (a feature missing for the whole stroke stays NaN).
2. Resample each stroke to 60 steps with contact at step 40. The part before and after contact are stretched separately.
3. DTW between strokes on features divided by each feature's overall scale, Sakoe-Chiba band 12 (20% of 60), written by hand in numpy.
4. Medoid: the stroke with the smallest total DTW distance to the others.
5. Align every stroke to the medoid, average along the warping paths per step: mean and std.
6. Std floor: 0.25 times each feature's overall scale, so a step where strokes happen to agree does not get a near-zero band.

Counts: 174 forehand strokes (187 minus the 13 suspects) and 155 backhand.

## Scoring (padel/scoring.py)

1. Resample the stroke the same way and DTW-align it to the reference mean.
2. z = (value - mean) / std per step and feature.
3. Feature score: contact-weighted mean over steps (30 to 50 count double) of exp(-0.5 (z / 2)^2), so any shift lowers it a little.
4. Overall score: 100 x (0.5 x mean of the feature scores + 0.5 x mean of the two lowest). Softness 2 and mix 0.5 were chosen by `scripts/tune_scoring.py` on training strokes only. An optional timing term (off by default) compares the time of the fastest shoulder turn relative to contact (`turn_lead`) with the professionals' spread, stored in the reference files.
5. Feedback: the three lowest-scoring features, each with the phase (before, at, after contact), direction (higher or lower than the reference) and size in the feature's unit, taken at the step of largest |z|.
6. `coverage`: the share of values that were measured rather than missing. A feature with no data is skipped.

`wrist_speed` is excluded from scoring by default because contact is defined from its peak (circular). It is a function parameter.

## Phone clip path (built 2026-11-20, not yet tried on a real padel clip)

Start the app with `py -m padel.server` and open the printed address on a phone on the same Wi-Fi (Windows may ask to allow Python through the firewall). The weights default to `../models/yolo11m-pose.pt` next to the repo; set `PADEL_WEIGHTS` to use another file. Jobs and overlays go to `data/jobs/` (git-ignored); uploads are deleted after processing.

| Stage | Code | What it does |
|---|---|---|
| Pose | `padel/pose.py` (`clip_keypoints`) | Decode, YOLO11-pose, keep the largest person (warning if more than one), mask confidence below 0.3, fill gaps up to 3 frames, Savitzky-Golay smoothing (window 7, order 2). Missing points are NaN with conf 0 |
| Left/right fix and features | `fix_lr_all`, `stroke_features` | Same code as the pro strokes, so reference and player match |
| Swings | `padel/swings.py` (`detect_swings`) | Racket wrist speed, 5-frame moving average, peaks above 3x the clip median and 2 torso lengths per second, at least 0.8 s apart; window to where speed drops below 1.5x the median, at most 0.7 s each side |
| Scoring | `padel/analyze.py` (`analyze_keypoints`) | Each swing with the racket arm seen in at least half its frames goes to `score_stroke`; feedback becomes plain sentences. Failures instead of scores: `clip_too_long` (over 20 s), `no_person`, `no_swing`, `low_pose_quality` |
| Overlay | `padel/overlay.py` | Skeleton on every frame, score label during each swing, a ring on the wrist at contact; optional skeleton-only (no footage); H.264 via ffmpeg |
| Server and page | `padel/server/` | API of design 3.5, one job at a time in a background thread, timings per stage in the result (`timings_s`), phone page in `static/index.html` |
