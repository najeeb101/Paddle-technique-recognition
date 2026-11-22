# Oral exam guide

Every member is examined on all the code and may be asked to change it live (project-plan.md, day 19). This guide is for practice: the pipeline in one minute, then each module with what it does, the numbers behind it, likely questions with short answers, and one live change to practise. Code details are in [pipeline.md](pipeline.md), all results in [results.md](results.md). Numbers here are as of 2026-11-22.

How to practise: one person asks the questions of a module the other did not write; answer without opening the code, then open it and check. Do the live change with the tests running (`py -m pytest -q`, 87 pass).

## The pipeline in one minute

A player films one forehand or backhand from behind with a phone. YOLO11-pose finds 17 body points in every frame. We clean and smooth them, fix frames where the model swapped left and right, and compute 10 features per frame (joint angles, body turn, wrist height and speed). The swing is found from the racket wrist's speed. The stroke is stretched to 60 steps with contact at step 40, lined up in time with a professional reference by DTW, and every feature is compared with the professionals' mean and spread at each step. The score is how similar the stroke is to the pros, from 0 to 100, plus the three features that differ most as plain-language tips.

The reference comes from PadelTracker100: two 2022 Barcelona finals with pose, shot and ball labels. We keep strokes by near-side players (seen from behind, like the phone), pick the hitter from the ball, check handedness by eye, and average 125 forehands and 128 backhands.

## Terms

| Term | Meaning |
|---|---|
| COCO-17 | The standard list of 17 body points (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with x, y and a confidence |
| Homography | A 3x3 matrix that maps image pixels to positions on a flat surface, here the court in metres |
| Hungarian algorithm | Finds the cheapest one-to-one matching between two sets; used to match detections to tracks (`linear_sum_assignment`) |
| Viterbi | Dynamic programming that finds the cheapest sequence of hidden states over time; here state = "this frame's left and right are swapped" |
| Savitzky-Golay filter | Smoothing that fits a small polynomial in a sliding window, so it keeps peaks better than a plain average |
| DTW (dynamic time warping) | Lines up two sequences that move at different speeds by finding the cheapest path through a cost matrix |
| Sakoe-Chiba band | Limits DTW so a step can only match steps within a fixed distance (12 of 60), which stops extreme warps and saves time |
| Medoid | The real example with the smallest total distance to all others (unlike a mean, it is an actual stroke) |
| z-score | (value - mean) / std: how many spreads a value is from the professionals' mean |
| AUC | The chance a random clean stroke scores higher than a random flawed one. 0.5 = no better than guessing, 1 = perfect |
| Hold-out | Strokes kept out of building the reference, used only for testing |
| IoU | Intersection over union of two boxes: overlap area divided by combined area |

## A. Data (padel/data.py, court.py, tracking.py, strokes.py)

**What it does.** Loads the PadelTracker100 pose and shot files, groups consecutive shot frames into stroke windows (575: 293 forehand, 282 backhand), tracks the four players, decides who hit and when, and keeps near-side strokes.

**Numbers.** 575 windows. Near side with the ball hitter rule: 259 strokes (131 FH, 128 BH). The old fastest-wrist rule was wrong in about a third of 40 checked windows (23 right, 10 wrong, 7 unclear). Handedness by eye: 1 left-hander among the 8 pros; the old rules were only 60-64% right.

**Likely questions.**
- *Why not take the hitter from the fastest wrist?* Checked 40 windows by eye: wrong about a third of the time, often a player in ready stance. The ball changes direction where it is hit, so the player whose wrist is nearest the ball at that frame is the hitter. It matched all 33 clear cases.
- *How do you know a player is on the near side?* The dataset gives a homography per match (the same for every frame, which the code checks). We map the feet (ankles, or hips if no ankles) to court metres; y > 10 m is past the net on the camera side.
- *The pose files have no player IDs. How do you track?* Hip centre per detection, matched to the four tracks with the Hungarian algorithm. A detection can join a track only within 120 px plus 30 px per frame the track was missed. Tracks are never deleted, since the same four players stay on court.
- *What is odd about the men's file?* Its keypoint names list left and right swapped, but the coordinates are in standard order. Settled by drawing points by name (`scripts/check_lr.py`), so for FinalM the names are ignored.
- *Is contact the true ball contact?* No. Contact is the racket wrist's speed peak, a proxy. The ball turn frame is only used to pick the hitter.

**Live change.** Make the near-side test need the feet at least 1 m past the net. In `court.py`, `is_near_side` compares with `NET_Y`; change it to `NET_Y + 1.0` and add a test in `tests/test_court.py`.

## B. Pose (padel/pose.py)

**What it does.** Runs YOLO11m-pose on each frame of a phone clip, follows one player, drops unsure points, fills short gaps and smooths.

**Numbers.** Points with confidence below 0.3 become missing. Gaps up to 3 frames are filled linearly; longer gaps or gaps at the clip ends stay missing. Savitzky-Golay window 7, order 2. Player follow: IoU at least 0.1 with the last box; after 15 frames lost, take the largest person again. YOLO11m-pose: 4.4 fps on CPU; a GTX 1650 Ti did a 10 s clip in 9-11 s.

**Likely questions.**
- *Why YOLO11-pose and not MediaPipe?* Benchmarked on the broadcast video: YOLO found 2.0 near-side people per frame, MediaPipe heavy 0.3. MediaPipe is built for one large person; YOLO handles small and several people. The instructor allowed "MediaPipe or something similar".
- *Why not just take the largest person each frame?* On real video the analysis jumped to the partner when she came nearer. Now the first player is followed by box overlap.
- *Why not fill every gap?* Guessing over long gaps invents motion. Missing stays NaN and lowers the confidence instead.
- *Why Savitzky-Golay and not a moving average?* It smooths jitter but keeps the shape of fast peaks like the swing better.

**Live change.** Make the confidence threshold 0.5. `clean_keypoints(kp, min_conf=0.3)`: change the default, or pass it in `clip_keypoints`. Run `tests/test_pose.py`.

## C1. Features (padel/features.py)

**What it does.** 10 per-frame features from the keypoints, after making every stroke right-handed (a left-handed stroke is mirrored and its sides swapped, so one reference serves both).

**The features.** elbow_angle, shoulder_angle, knee_angle_dom, knee_angle_off (joint angles), shoulder_rotation and hip_rotation (direction of the shoulder and hip lines, unwrapped over time), separation (shoulders minus hips), torso_lean, wrist_height (shoulder y minus wrist y, in torso lengths), wrist_speed (torso lengths per second).

**Left/right fix.** The pose model swaps left and right labels mid-stroke: 112 of 342 strokes had shoulder swaps, and hips and legs flip on their own (94 strokes). `fix_lr_flips` is a two-state Viterbi: the cost of a frame is how far the joints jump from the last valid frame under that labelling, plus a weak prior that the shoulder line points image-right (true for a player seen from behind). `fix_lr_all` runs it on the whole body, then again on hips and legs only. Hip rotation spread fell from about 75 to about 17 degrees.

**Hip mask.** When the hips look almost on top of each other (width under 0.15 torso lengths), the hip angle is noise and jumps by up to 100 degrees. Those frames are set to NaN for hip_rotation and separation.

**Likely questions.**
- *Why divide by torso length?* Makes distances independent of how far the player is from the camera.
- *Why are these 2D angles a weakness?* They are measured in the image plane, so they change with camera position. That is why the filming guide fixes the camera behind the player and the reference uses only near-side players.
- *Why unwrap rotation?* An angle can jump from 179 to -179 degrees while the body only turned 2 degrees. Unwrapping removes those 360-degree jumps.
- *Why not just swap frames where the shoulders point the wrong way?* Rotation is exactly what we measure, so a rule on direction alone would undo real turns. Motion continuity is the main cost; the direction is only a weak prior.

**Live change.** Add a feature "wrist_lateral": (wrist x minus shoulder x) / torso. Add the name to `FEATURES` and a line in `cols`, then rebuild features and references (pipeline.md steps 6c and 7). Append it at the end of `FEATURES`: scoring looks columns up by the reference's own feature names, so old reference files still line up until rebuilt. The tests use `len(FEATURES)` and adapt.

## C2. Reference (padel/reference.py)

**What it does.** Turns many strokes into one mean curve and spread per feature.

**Steps.** Fill short NaN runs; resample to 60 steps with contact at step 40 (before and after contact stretched separately); scale each feature by its overall spread; hand-written DTW with a Sakoe-Chiba band of 12; pick the medoid; align every stroke to the medoid and average along the paths; std floor of 0.25 x the feature's scale. 13 suspect forehand labels are excluded (docs/label-suspects.md).

**Likely questions.**
- *Why DTW and not compare step by step?* Players swing at different speeds and with different preparation; DTW lines up the same phase of the motion.
- *Why a medoid and not the mean as the alignment target?* The mean of misaligned strokes is blurred; the medoid is a real stroke with sharp timing.
- *Why scale features before DTW?* Angles are in tens of degrees, wrist height in fractions; without scaling the angles would decide the whole path.
- *Why the std floor?* Where strokes happen to agree, std gets near zero and any tiny difference would look huge.
- *Why the band?* Stops absurd warps (start matched to end) and cuts the work from 60 x 60 to about 60 x 25 cells.
- *Walk me through DTW.* Cost matrix of distances between every step pair; accumulate D[i,j] = cost[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1]); backtrack from the end along the cheapest neighbour to get the path. NaN features are skipped in the distance and the sum is rescaled.

**Live change.** Change the band to 20% of the sequence length instead of a fixed 12: in `dtw`, compute `band` from `max(n, m)` when it is None. Run `tests/test_reference.py` (it has a hand-worked DTW example).

## C3. Scoring (padel/scoring.py)

**What it does.** Resample the stroke, DTW-align it to the reference mean, z-score every step, turn z into a similarity, combine.

**Formula.** Per feature: contact-weighted mean over steps (steps 30 to 50 count double) of exp(-0.5 (z / 2)^2). Overall: 100 x (0.5 x mean of all features + 0.5 x mean of the two worst). Feedback: the three lowest features, with phase (before, at, after contact), direction and size at the step of largest |z|.

**Numbers.** Hold-out (stroke scored higher against its own type's reference than the other type's): about 96% forehand, about 89% backhand (seeds 0-4). Clean pro strokes score about 85. Flawed-stroke AUC at realistic fault sizes: 0.49 to 0.67.

**Likely questions.**
- *Why similarity and not good/bad?* No public padel dataset labels technique quality, only stroke type. So the honest claim is "how close to the pros", not "correct".
- *Why is wrist_speed not scored?* Contact is defined as the wrist speed peak, so scoring it would partly score our own definition (circular).
- *Why mix in the two worst features?* One clearly wrong feature would otherwise be hidden by eight normal ones.
- *How were softness 2 and mix 0.5 chosen?* Grid search on a validation split of the training strokes only (`scripts/tune_scoring.py`); the hold-out was never used for tuning. Softness 1.5 and 2 tied; 2 gives clean pros about 84 instead of 72.
- *Why does it barely catch faults?* The pros' own spread is as big as the faults: elbow std at contact is about 39-46 degrees, the test fault is 20. Even an oracle that knows which feature was faulted reaches only 0.47-0.78 AUC. The limit is the reference (broadcast pose noise, eight players mixed), not the formula.
- *Why is timing off?* DTW absorbs a 0.1 s shift against a blurred mean. The separate turn_lead measure has a pro spread of about 0.14-0.16 s, bigger than the 0.1 s fault, and it did not help on validation.

**Live change.** Show the four worst features instead of three: in `score_stroke`, `[:3]` on `worst`. Then make contact count triple: `w[...] = 3.0` in `step_weights`. Check `tests/test_scoring.py`.

## D. Phone app (padel/swings.py, analyze.py, overlay.py, server/)

**What it does.** Finds the swing in a clip, scores it, returns JSON and a skeleton overlay video to a phone page served by FastAPI.

**Numbers.** Swing detector: wrist speed relative to the body centre, 5-frame moving average, peak above 3x the clip median and at least 2 torso lengths per second, peaks at least 0.8 s apart, window out to 1.5x the median and at most 0.7 s each side. Found 44-61% of 342 pro strokes cut as 3 s clips (image wrist speed found 26-44%). A swing is scored only if the racket arm is seen in at least 50% of its frames; under 80% adds a warning. Clips over 20 s are refused.

**Likely questions.**
- *Why wrist speed relative to the body?* On match video the player runs and the camera pans, so plain image speed is high all the time and the swing peak does not stand out. Relative to the body the peak is about 4.5x the median instead of 2.6x.
- *Why failures instead of a low score?* A score from a bad clip would look like a real result. The four failure codes (clip_too_long, no_person, no_swing, low_pose_quality) each come with a message telling the player what to fix.
- *How does the server handle several uploads?* One background worker thread with a queue, so the model loads once and jobs run one at a time. Uploads are deleted after processing (privacy).
- *Why H.264 through ffmpeg?* OpenCV writes mp4v, which many phones will not play.

**Live change.** Lower the swing threshold to 2.5x the median: `peak_factor` in `detect_swings`. Run `tests/test_swings.py` and `tests/test_analyze.py`. With own clips, check the effect with `scripts/eval_own_clips.py` (it sweeps these settings).

## E. Evaluation (scripts/eval_holdout.py, eval_flawed.py, eval_swings.py, eval_own_clips.py, padel/faults.py)

- **Hold-out:** 80/20 split per stroke type; does a stroke score higher against its own type? Tests that the reference captures the stroke, not that it judges quality.
- **Flawed strokes:** five synthetic faults on real strokes (elbow 20 degrees more bent, knee 20 degrees straighter, shoulder line 20 degrees, wrist 0.3 torso lengths lower, contact 0.1 s late). Measures whether the score drops (AUC) and whether the tips name the changed feature.
- **Swing detection:** every pro stroke as a 3 s clip; found if a swing lands within 6 frames of the labelled contact.
- **Own clips:** `eval_own_clips.py` runs the team's filmed clips: swings found, good vs fault scores, whether the named fault shows up in the tips, compared with chance (one feature among three tips from nine = 33%).

*Why synthetic faults?* There are no labelled bad strokes. Editing one feature by a known amount shows whether the score reacts to that feature. It does not show that the score agrees with a coach.

## Honest weaknesses (expect these)

| Weakness | Answer |
|---|---|
| The score barely separates flawed strokes (AUC 0.49-0.67) | Reference spread from broadcast pose noise; an oracle does not do much better. Own phone clips are the real test |
| Timing faults are not detected | Pros' timing spread is larger than the fault; measure kept but off |
| Swing detector finds about half of pro strokes | Broadcast players are small and keep moving; phone clips with one large player should be easier, thresholds to be set on own clips |
| Contact is a speed peak, not the true ball contact | No ball tracking on phone clips (out of scope); stated as a proxy |
| 2D angles depend on the camera | Fixed filming setup and near-side-only reference |
| Only two matches, eight players | It is the only public padel set with pose and shot labels found; licensed footage is future work |

## Rapid-fire numbers

575 windows; 259 near-side strokes; reference 125 FH / 128 BH; 60 steps, contact at 40; DTW band 12; std floor 0.25; softness 2; worst mix 0.5 (two worst); contact steps 30-50 double weight; confidence cut 0.3; gap fill 3 frames; smoothing 7/2; swing 3x median and 2 torso lengths/s; arm seen 50% (warn under 80%); clip max 20 s; 87 tests.
