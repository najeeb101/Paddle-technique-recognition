# Results and findings log

Everything measured or found so far, with the evidence and what is still unverified. Updated 2026-11-21. No claim here is stronger than the check behind it.

## 1. Data

| Item | Value | Check |
|---|---|---|
| Stroke windows | 575 (293 forehand, 282 backhand) | matches the dataset counts |
| Near-side windows (court homography) | 355 (194 FH, 161 BH) | reproduced exactly after the first implementation was lost and rebuilt |
| Older near-side rule (hip y >= 600 px) | 231 (136 FH, 95 BH), a strict subset | replaced |
| Hand labels | 311 right, 31 left, 13 unclear | by eye from the racket on contact sheets; the left-hander is the dark-haired player in the dark kit (most likely Paula Josemaria, not confirmed by a source) |
| Strokes with features | 342 (187 FH, 155 BH) | the 13 unclear windows dropped |
| Players | Triay/Salazar beat Josemaria/Sanchez (women's final); Lebron/Galan beat Chingotto/Di Nenno (men's final) | from the web; the web gave no reliable racket hands, hence the eye labelling |

## 2. Automatic rules that failed

- Handedness from the fastest wrist: 60% against the eye labels. From wrist side at contact: 64%. Neither is used.
- Handedness per track ID: 54 to 84% agreement. Track IDs are not players: players cross the court and the tracker is position-based, so one ID spans both sides over 20k frames (median hip y 305 to 684 px).
- Hitter rule (fastest wrist peak) is only eyeballed on a 24-window sheet; nearly all tiles showed the inferred player mid-swing. No accuracy count, and contact error in frames is not measured.

## 3. Pose label noise

- Men's file keypoint names are mislabelled but the column order is standard COCO; names are ignored (`NAMES_TRUSTED`). The women's file is correct. Settled by drawing keypoints on near-side players (`scripts/check_lr.py`).
- Mid-stroke left/right swaps: 127 of 342 strokes touched by the fix (112 at the shoulders, plus more at the hips). After the fix: 9 strokes (10 frames) with a hip flip, 1 stroke with a shoulder flip. Hip rotation spread in the reference: from about 75 to about 17 degrees.
- Missing values per feature before gap filling: elbow angle 11.8%, wrist height 10.5%, wrist speed 17.4%, shoulder angle 4.0%, others under 2%.

## 4. Label check on forehands

At contact a right-hander seen from behind should have the racket wrist right of the shoulder midline for a forehand. 88% of the 187 forehands pass; 15 have the wrist clearly left. By eye: 6 likely backhands, 7 ambiguous, 2 probably correct. The 13 likely or ambiguous strokes are excluded from the reference; labels are not changed. Backhands cannot be checked this way (73% also land right of the midline), so backhand labels are unchecked. No padel player has confirmed the verdicts. Details: [label-suspects.md](label-suspects.md).

## 5. Reference

174 forehand and 155 backhand strokes, 60 steps, contact at step 40. Mean std per feature (forehand): elbow 38, shoulder angle 33, knee 26 to 27, shoulder rotation 25, hip rotation 18, separation 21, torso lean 21 degrees; wrist height 0.4 torso lengths. These bands are wide; they mix eight players and pose noise.

## 6. Scoring hold-out (seed 0, 80/20 per type)

### Current scorer (2026-11-21: continuous similarity, softness 2, worst mix 0.5)

| Held-out | n | Own-type score (mean, sd) | Wrong-type score (mean, sd) | Own higher |
|---|---|---|---|---|
| Forehand | 35 | 84.6 (7.5) | 77.3 (9.2) | 86% |
| Backhand | 31 | 85.2 (5.0) | 81.2 (7.6) | 65% |

Own-type range: forehand 57 to 94, backhand 74 to 95. Better than the first scorer below (80% / 61%), but backhand vs forehand is still weak.

### First scorer (2026-11-18: share of steps with |z| <= 2)

| Held-out | n | Own-type score (mean, sd) | Wrong-type score (mean, sd) | Own higher |
|---|---|---|---|---|
| Forehand | 35 | 94.0 (7.7) | 88.6 (9.6) | 80% |
| Backhand | 31 | 94.9 (4.3) | 92.7 (6.3) | 61% |

Own-type range: forehand 63 to 100, backhand 83 to 100; no stroke scores below 50. Reading: the score saturates near 94, so it cannot yet tell a good stroke from a poor one, and only weakly tells forehand from backhand. Likely causes: wide std bands (almost any plausible stroke is within 2 std), DTW absorbing timing and shape differences, and the 2 std limit being loose. Not yet tried: a z limit of 1 or less, a smaller std floor, scoring without whole-stroke DTW alignment. This hold-out has been looked at once; tune on folds of the build set only (plan day 15) and take the final numbers from a fresh split or from the team's own clips.

## 6b. Flawed-stroke test (`scripts/eval_flawed.py`, 2026-11-21)

Each held-out stroke is scored clean and with one synthetic fault (elbow 20 degrees more bent, racket knee 20 degrees straighter, shoulder line rotated 20 degrees, wrist 0.3 torso lengths lower, contact 0.1 s late). AUC = chance a clean stroke beats a flawed one across strokes (0.5 = no separation). "Dropped" = share of strokes whose own flawed copy scores lower; for the continuous score about 50% is chance (a fixed shift moves about half the strokes towards the mean), so read AUC first.

| Fault (size x1) | First scorer AUC (FH / BH) | Current AUC (FH / BH) | Current dropped (FH / BH) | Single-feature oracle AUC (FH / BH) |
|---|---|---|---|---|
| Elbow | 0.57 / 0.53 | 0.54 / 0.53 | 60% / 61% | 0.56 / 0.50 |
| Knee | 0.58 / 0.51 | 0.55 / 0.58 | 51% / 74% | 0.47 / 0.65 |
| Rotation | 0.59 / 0.58 | 0.64 / 0.64 | 80% / 100% | 0.68 / 0.78 |
| Wrist height | 0.50 / 0.56 | 0.54 / 0.59 | 69% / 87% | 0.72 / 0.75 |
| Timing | 0.49 / 0.50 | 0.50 / 0.51 | 51% / 71% | not measured |

The first scorer often left the score unchanged (ties), so its "dropped" values (31% to 71%) are not comparable. At x4 the current score drops 8 to 37 points (rotation most); timing barely moves it. Leave-match-out (build FinalM, test FinalF) gives AUC 0.49 to 0.69, the same picture.

Single-feature oracle: an informed baseline that already knows the faulted feature and reads its mean |z| over the contact window (a scratch check, not a committed script). It is not a strict bound (the scorer beats it for forehand knee and backhand elbow), but it reaches only 0.47 to 0.78, so the plan target of AUC >= 0.8 at these fault sizes looks out of reach on this reference. A 20 degree elbow change is about half the professionals' own spread (elbow std about 35 to 43 degrees). The main limit looks like the reference (eight players mixed, broadcast pose noise) rather than the scoring formula.

How the settings were chosen: `scripts/tune_scoring.py` splits the training 80% again (75/25) and grid-searches softness, worst mix and timing on that validation part only. Objective: mean of the fault AUCs and the own-type ordering. Softness 1.5 and 2 tied; 2 was picked because clean pro strokes then score about 84 instead of 72. The objective uses the same synthetic faults, so the held-out numbers above are tuned towards those fault types.

Timing: DTW against the averaged reference does not see a 0.1 s shift (the reference curves are too blurred; against the stroke's own clean copy DTW does find it). A second measure, the time of the fastest shoulder turn relative to contact (`turn_lead`), has a professional spread of about 0.15 s around -0.13 s, so a 0.1 s fault is under one spread. Adding it did not raise the validation objective, so it is off by default (`use_timing=True` turns it on).

## 6c. Phone app on real padel video (2026-11-21)

No own clips yet, so 8 near-side pro strokes (2 per match and type, right-handed, seed 7) were cut from the broadcast video as 3 s clips (true contact at frame 45), cropped square around the hitter and scaled to 720 px, then run through `analyze_clip`. These strokes are in the reference, so scores are optimistic.

| Result | First run | After the person-tracking fix |
|---|---|---|
| Swing found within 6 frames of contact | 3 | 3 (scores 60, 79, 88) |
| Swing found at the wrong moment | 1 (the partner's later shot) | 0 |
| No swing found | 4 | 5 |

- Person switching: `extract_keypoints` took the largest person in each frame, so it jumped to the partner when she came nearer. It now follows the first player by box overlap (`padel/pose.py`, two new tests).
- Swing detection is too strict for fast pro swings: the 5-frame smoothing flattens the peak, so it falls under 3x the clip median. Smoothing 3 and factor 2.5 found 6 of 8, but 8 cropped broadcast clips are too few to retune on; decide with own phone clips.
- Tips are not reliable yet: one clip got "hip turn about 110 degrees lower" (skeleton noise on small, upscaled players).
- Pose time per 3 s clip on CPU (yolo11m): 31 to 162 s.
- A reference from one match or one track position is barely tighter (elbow spread at contact 33 to 44 degrees vs 41 for both), so the spread is mostly pose noise, not player mix.

## 6d. Swing detection on all 342 pro strokes (2026-11-21)

`scripts/eval_swings.py` cuts every near-side stroke as a 3 s clip (contact at frame 45) from the dataset keypoints, cleans and smooths them like phone keypoints, and counts a stroke as found when a detected swing lies within 6 frames of the labelled contact. Thresholds were compared on FinalF and checked on FinalM.

| Signal | Found FH / BH (FinalF) | Found FH / BH (FinalM) | Extra swings per clip |
|---|---|---|---|
| wrist_speed (image motion, as before) | 31.5% / 43.8% | 26.3% / 40.7% | 0.20 to 0.29 |
| swing_speed (wrist relative to body centre) | 51.1% / 60.9% | 44.2% / 59.3% | 0.36 to 0.60 |

- Cause of the misses: pros keep running and the camera pans, so the clip median of image wrist speed is high and the stroke peak sits only about 2.6x above it (median), under the 3x threshold. Measuring the wrist relative to the body centre lifts that ratio to about 4.5x.
- `analyze_clip` now uses `swing_speed` (padel/swings.py); the `wrist_speed` feature and the references are unchanged.
- Retuning the thresholds on FinalF did not beat the current ones on FinalM (best FinalF setting is the current one), so they stay.
- Ceiling on this test: even the single highest peak in the clip is at contact only 55% of the time (top two peaks 68%). Pose noise on small broadcast players and other fast arm movements in the 3 s compete with the stroke, so this test understates phone clips with one player filling the frame. The extra swings may partly be real recoveries or shots. Decide final thresholds on own phone clips (docs/filming-guide.md).

## 6d2. Zoomed re-pose of the pro strokes (2026-11-21)

`scripts/repose_strokes.py` re-ran YOLO11m-pose on a square crop around the hitter of every near-side stroke, scaled to 640 px like a phone clip (342 strokes, about 1.5 h on CPU), then cleaned and smoothed like phone keypoints. `scripts/compare_spread.py` (median per-step std over the contact window):

- Arm and knee spreads shrink: forehand elbow 41.9 -> 32.6 deg, knees 12 to 20% lower; backhand shoulder angle 36.1 -> 21.8, shoulder rotation 44.2 -> 25.8 deg. Missing values drop (elbow 10-13% -> 8-9%).
- Hip rotation spreads grow (FH 20.0 -> 26.7, BH 18.0 -> 25.4 deg), so separation does not improve. Cause not checked yet (hip L/R flips in the new poses are the first suspect).
- Elbow spread is still above the 20 deg fault size, so sharper poses alone do not make faults detectable. These strokes still use the old hitter picks (section 6e), so some are the wrong player.

Why hip spread grows (`scripts/check_hip_spread.py`, old-hitter files, 342 strokes):

- The hip line angle jumps by more than 60 deg between frames in 141 strokes on the crops vs 82 on the dataset poses (median jump about 100 deg).
- The jumps happen when the two hip points almost coincide: median hip width at a jump is 0.03 torso lengths on the crops. YOLO puts the hips that close in 15.6% of crop frames vs 3.9% of dataset frames (hip width median 0.34 vs 0.40 torso). The angle of a near-zero line is noise, so these are not left/right flips and the L/R fix cannot catch them.
- The typical stroke barely changes (median per-stroke difference 4.6 deg); the extra spread is these outliers.
- Masking hip_rotation and separation where hip width is under a threshold (spread over strokes at contact +-3 frames, crop poses):

| Mask below | Hip rotation | Separation | Missing |
|---|---|---|---|
| none | 28.2 deg | 30.5 deg | 0% |
| 0.10 torso | 24.5 deg | 25.9 deg | 10% |
| 0.15 torso | 21.3 deg | 20.1 deg | 20% |

With the 0.15 mask the crop hip spread matches the dataset poses (20.0 deg) and separation is tighter than on them (24.5). Cost: 20% of contact frames lose these two features. Not yet applied in `padel/features.py`; to be tested with the hold-out and flawed-stroke checks on the new crops.

## 6e. Hitter check on 40 windows (2026-11-21)

The 40 random windows of `scripts/make_check_sheets.py` (20 per match, all court sides) were labelled by eye from larger strips (`scripts/make_hitter_strips.py`, 5 frames around contact) and then from full frames with the dataset ball labels drawn (`scripts/compare_hitter_rules.py`). Answers are in data/check/hitter_check.csv.

- Current rule (track with the fastest single-frame wrist jump): right in 23, wrong in 10, unclear in 7. So hitter accuracy is about 58 to 75%, not the "nearly every tile" of the first eyeball check.
- Wrong picks are mostly a player standing in the ready position while another player hits: one keypoint jump on a still player beats the real swing. In the 23 sampled windows that feed the near-side reference, at least 5 are the wrong player, so the reference holds strokes that are not strokes, which adds to the pros' spread.
- Ball rule ("turn"): find the frame in the window where the labelled ball changes direction most, pick the player whose wrist is nearest the ball there. It agrees with all 33 clear labels (23 right kept, 10 wrong changed). Caveat: 4 of the 10 "wrong" labels were first marked right from tight crops and corrected after seeing the ball on full frames, so this is not a fully independent test.
- Smoothing or body-relative wrist speed alone fixed 3 to 7 of the 10.
- Phone clips are not affected (one player is followed).

### Rebuild with the ball rule (2026-11-21)

`infer_hitters.py --ball` (padel/strokes.py `analyse_window_ball`): 570 of 575 windows get a ball turn, 5 fall back to wrist speed or find no hitter. The hitter changes in 191 of 575 windows (33%). Contact stays the racket wrist's speed peak; it is within 4 frames of the ball turn at the median, 10 frames at the 90th percentile.

- Racket hand from the nearest wrist to the ball agrees with the eye labels in only 154 of 209 windows (74%), so it is not used. Eye labels are kept where the hitter did not change; the 26 changed near-side women's windows were labelled again by eye (6 L, all dark kit; 5 unclear, dropped); the 29 changed men's windows are R (all 186 men's eye labels are R).
- Near side: 269 windows (was 355); 259 usable with a known hand (131 FH, 128 BH; was 342). References: 125 FH (13 label suspects excluded as before), 128 BH. Old files kept as data/*_wristrule.csv and data/processed/stroke_features_wristrule.npz.

| Test | Old hitter rule | Ball hitter rule |
|---|---|---|
| Hold-out, own type scores higher (FH / BH) | 86% / 65% | 100% / 85% |
| Hold-out own-type mean (FH / BH) | about 85 | 87.3 / 84.6 (wrong type 70.9 / 75.0) |
| Flawed-stroke AUC at x1 | 0.53 to 0.64 | 0.50 to 0.65 |

- Clean hitters fix stroke-type separation but not fault detection: a realistic fault still moves the score by 0.3 to 2.9 points. The limit there is pose noise (sections 6b, 6d2), not wrong players.
- The zoomed re-pose (6d2) used the old hitters; a re-run on the new 259 strokes would combine both gains.

## 7. Pose model

YOLO11m-pose: 0.227 s per frame (4.4 fps, CPU), 2.02 people per frame, mean keypoint confidence 0.72. MediaPipe heavy: 0.140 s per frame, 0.30 people per frame. 60 frames of one broadcast clip only; MediaPipe is untested on phone footage. See [measurements.md](measurements.md).

## 8. Process notes (what went wrong and how it was fixed)

- The first court-filter implementation was never committed and was lost; rebuilt from the homography files and checked to give the same 355.
- Counting L/R flips by eye underestimated them by a factor of six (18 versus 112).
- A hip-only second pass was needed because the global pass left hip and leg flips.
- Windows quirks: `py` not `python`; `C:/` paths for Python; long commands must run in the background.

## 9. What is left

1. Score separation now looks limited by the reference spread (section 6b). Options: a tighter reference (one player or one match), or judge the score on the team's own clips instead.
2. Flawed-stroke plot for the report (the table is in 6b).
3. Leave-players-out split. Needs a hand-labelled player map, or a leave-match-out split.
4. Compare dataset poses with re-run YOLO poses on about 50 windows.
5. Try the phone app (built 2026-11-20) on a real padel clip and from a phone over Wi-Fi.
6. Film consented clips (20 per stroke, 2 players), failure log, own-clip evaluation, coach ratings if possible.
7. Count hitter accuracy on 40 windows and contact error in frames; ask a padel player to settle the 7 ambiguous forehands and check handedness.
8. Latency measurement and a speed pass.
9. Report, demo script, backup video, oral exam dry run.

Open outside the code: written instructor approval, whether a phone app satisfies the "camera sensor connected to MATLAB" requirement, demo timing, rights to the pro match video, and a fifth team member (the team has four).
