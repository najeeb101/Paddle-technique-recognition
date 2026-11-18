# Results and findings log

Everything measured or found so far, with the evidence and what is still unverified. Updated 2026-11-18. No claim here is stronger than the check behind it.

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

| Held-out | n | Own-type score (mean, sd) | Wrong-type score (mean, sd) | Own higher |
|---|---|---|---|---|
| Forehand | 35 | 94.0 (7.7) | 88.6 (9.6) | 80% |
| Backhand | 31 | 94.9 (4.3) | 92.7 (6.3) | 61% |

Own-type range: forehand 63 to 100, backhand 83 to 100; no stroke scores below 50. Reading: the score saturates near 94, so it cannot yet tell a good stroke from a poor one, and only weakly tells forehand from backhand. Likely causes: wide std bands (almost any plausible stroke is within 2 std), DTW absorbing timing and shape differences, and the 2 std limit being loose. Not yet tried: a z limit of 1 or less, a smaller std floor, scoring without whole-stroke DTW alignment. This hold-out has been looked at once; tune on folds of the build set only (plan day 15) and take the final numbers from a fresh split or from the team's own clips.

## 7. Pose model

YOLO11m-pose: 0.227 s per frame (4.4 fps, CPU), 2.02 people per frame, mean keypoint confidence 0.72. MediaPipe heavy: 0.140 s per frame, 0.30 people per frame. 60 frames of one broadcast clip only; MediaPipe is untested on phone footage. See [measurements.md](measurements.md).

## 8. Process notes (what went wrong and how it was fixed)

- The first court-filter implementation was never committed and was lost; rebuilt from the homography files and checked to give the same 355.
- Counting L/R flips by eye underestimated them by a factor of six (18 versus 112).
- A hip-only second pass was needed because the global pass left hip and leg flips.
- Windows quirks: `py` not `python`; `C:/` paths for Python; long commands must run in the background.

## 9. What is left

1. Make the score discriminate (tune the z limit and std floor on build-set folds), then re-run.
2. Flawed-stroke test: offset one feature of real strokes by a known amount and check the score drops and the right feature is named. Put a table and plot in this file.
3. Leave-players-out split. Needs a hand-labelled player map, or a leave-match-out split.
4. Compare dataset poses with re-run YOLO poses on about 50 windows.
5. Swing detection for phone clips, `analyze_clip`, result JSON, overlay video, FastAPI server and phone page, readable error messages.
6. Film consented clips (20 per stroke, 2 players), failure log, own-clip evaluation, coach ratings if possible.
7. Count hitter accuracy on 40 windows and contact error in frames; ask a padel player to settle the 7 ambiguous forehands and check handedness.
8. Latency measurement and a speed pass.
9. Report, demo script, backup video, oral exam dry run.

Open outside the code: written instructor approval, whether a phone app satisfies the "camera sensor connected to MATLAB" requirement, demo timing, rights to the pro match video, and a fifth team member (the team has four).
