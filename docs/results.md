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

Each held-out stroke is scored clean and with one synthetic fault (elbow 20 degrees more bent, racket knee 20 degrees straighter, shoulder line rotated 20 degrees, wrist 0.3 torso lengths lower, contact 0.1 s late). AUC = chance a clean stroke beats a flawed one across strokes (0.5 = no separation). "Dropped" = share of strokes whose own flawed copy scores lower.

| Fault (size x1) | First scorer AUC (FH / BH) | Current AUC (FH / BH) | Current dropped (FH / BH) | Ceiling AUC (FH / BH) |
|---|---|---|---|---|
| Elbow | 0.57 / 0.53 | 0.54 / 0.53 | 60% / 61% | 0.56 / 0.50 |
| Knee | 0.58 / 0.51 | 0.55 / 0.58 | 51% / 74% | 0.47 / 0.65 |
| Rotation | 0.59 / 0.58 | 0.64 / 0.64 | 80% / 100% | 0.68 / 0.78 |
| Wrist height | 0.50 / 0.56 | 0.54 / 0.59 | 69% / 87% | 0.72 / 0.75 |
| Timing | 0.49 / 0.50 | 0.50 / 0.51 | 51% / 71% | not measured |

With the first scorer "dropped" was 31% to 71%. At x4 the current score drops 8 to 37 points (rotation most); timing barely moves it. Leave-match-out (build FinalM, test FinalF) gives AUC 0.49 to 0.69, the same picture.

Ceiling: an oracle that already knows the faulted feature and reads its mean z over the contact window (a scratch check, not a committed script). No scorer can do better on this reference, so the plan target of AUC >= 0.8 is not reachable at these fault sizes. A 20 degree elbow change is about half the professionals' own spread (elbow std about 35 to 43 degrees). The limit is the reference (eight players mixed, broadcast pose noise), not the scoring formula.

How the settings were chosen: `scripts/tune_scoring.py` splits the training 80% again (75/25) and grid-searches softness, worst mix and timing on that validation part only. Objective: mean of the fault AUCs and the own-type ordering. Softness 1.5 and 2 tied; 2 was picked because clean pro strokes then score about 84 instead of 72. The objective uses the same synthetic faults, so the held-out numbers above are tuned towards those fault types.

Timing: DTW against the averaged reference does not see a 0.1 s shift (the reference curves are too blurred; against the stroke's own clean copy DTW does find it). A second measure, the time of the fastest shoulder turn relative to contact (`turn_lead`), has a professional spread of about 0.15 s around -0.13 s, so a 0.1 s fault is under one spread. Adding it did not raise the validation objective, so it is off by default (`use_timing=True` turns it on).

## 7. Pose model

YOLO11m-pose: 0.227 s per frame (4.4 fps, CPU), 2.02 people per frame, mean keypoint confidence 0.72. MediaPipe heavy: 0.140 s per frame, 0.30 people per frame. 60 frames of one broadcast clip only; MediaPipe is untested on phone footage. See [measurements.md](measurements.md).

## 8. Process notes (what went wrong and how it was fixed)

- The first court-filter implementation was never committed and was lost; rebuilt from the homography files and checked to give the same 355.
- Counting L/R flips by eye underestimated them by a factor of six (18 versus 112).
- A hip-only second pass was needed because the global pass left hip and leg flips.
- Windows quirks: `py` not `python`; `C:/` paths for Python; long commands must run in the background.

## 9. What is left

1. Score separation is now limited by the reference spread (section 6b). Options: a tighter reference (one player or one match), or judge the score on the team's own clips instead.
2. Flawed-stroke plot for the report (the table is in 6b).
3. Leave-players-out split. Needs a hand-labelled player map, or a leave-match-out split.
4. Compare dataset poses with re-run YOLO poses on about 50 windows.
5. Try the phone app (built 2026-11-20) on a real padel clip and from a phone over Wi-Fi.
6. Film consented clips (20 per stroke, 2 players), failure log, own-clip evaluation, coach ratings if possible.
7. Count hitter accuracy on 40 windows and contact error in frames; ask a padel player to settle the 7 ambiguous forehands and check handedness.
8. Latency measurement and a speed pass.
9. Report, demo script, backup video, oral exam dry run.

Open outside the code: written instructor approval, whether a phone app satisfies the "camera sensor connected to MATLAB" requirement, demo timing, rights to the pro match video, and a fifth team member (the team has four).
