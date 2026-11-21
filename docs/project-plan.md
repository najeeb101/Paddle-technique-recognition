# Project plan: 4 weeks

Goal: a phone-friendly web app where a player records one forehand or backhand on a single fixed camera
and gets back an explainable similarity score against a carefully filtered professional reference, with
confidence indicators, failure reporting, a skeleton overlay, and the two or three biggest deviations.

This plan replaces the earlier 15-week MATLAB plan. The technical details are in
[design.md](design.md); data facts are in [datasets.md](datasets.md); related work is in
[literature-review.md](literature-review.md).

## 1. Read this first: course compliance

`CMAI 480 - Projects.pdf` requires MATLAB as the primary environment, a camera connected to MATLAB, at
least one from-scratch vision step, only instructor-permitted pretrained models, and individual oral
examination of all code. It bans AI assistance (produce, complete, rewrite, debug, document) unless the
instructor authorises it explicitly. This plan is **Python-first with a phone web app**, which the PDF
does not allow by itself. The team reports that the instructor allows Python and AI use as long as each
member understands the content and it stays within the project requirements.

**Before week 1 ends, get this in writing from the instructor:**

- [x] Python may be the main language for pose extraction, analysis and the app (instructor approved, reported verbally by Najeeb, 2026-11-08; get it in writing).
- [x] AI assistance is allowed under the stated condition (same verbal approval).
- [x] Which pretrained pose model is permitted: the instructor advised MediaPipe or something similar; YOLO11-pose chosen after a benchmark (`docs/measurements.md`).
- [ ] A phone web app satisfies the "camera sensor connected to MATLAB" requirement, or what substitutes for it.
- [ ] Whether the 4-week schedule replaces the weeks 14–15 demo timing.

**If MATLAB is still required for part of the work**, the cheapest way to satisfy it is a small
MATLAB track (section 10) that implements DTW from scratch, runs the simulation study and produces the
evaluation plots from the keypoint files the Python side writes. It must be staffed from week 1; it
cannot be added at the end.

## Progress (updated 2026-11-20)

Python pipeline from pro data to a score exists and is tested (62 tests, `py -m pytest -q`). The phone app runs end to end (`py -m padel.server`) but has not yet been tried on a real padel clip. The score does not yet separate strokes well enough to ship.

### Done

| Plan item | State | Where |
|---|---|---|
| Day 1-2 loaders, men's keypoint order, stroke windows (575: 293 FH, 282 BH) | Done | `padel/data.py`, `docs/datasets.md` |
| Day 2 tracker | Done, but track IDs are not players | `padel/tracking.py` |
| Day 3-4 hitter, contact frame | Done, eyeballed only; hitter accuracy and contact error in frames not counted | `padel/strokes.py`, `scripts/infer_hitters.py` |
| Day 3 handedness | Done by eye on contact sheets (311 R, 31 L, 13 unclear), replacing rules that scored 60-64% | `scripts/make_hand_sheets.py`, `apply_hand_labels.py` |
| Day 4 near-side filter | Done with the v2 court homography: 269 of 575 windows with the ball hitter rule (355 with the old rule) | `padel/court.py` |
| Pose model | YOLO11m-pose chosen from a benchmark | `docs/measurements.md` |
| Day 5 features and curves | Done: 10 features, curves coherent after the left/right fix, 259 strokes (342 before the ball hitter rule) | `padel/features.py`, `scripts/build_stroke_features.py` |
| Left/right pose swaps | Fixed: the model swaps sides mid-stroke, shoulders and hips separately (127 of 342 strokes touched) | `fix_lr_all` in `padel/features.py` |
| Forehand label check | Done: 13 of 187 forehands flagged, excluded from the reference | `docs/label-suspects.md` |
| Day 6-7 DTW, resampling, medoid reference | Done: 174 forehand and 155 backhand strokes, 60 steps, contact at step 40 | `padel/reference.py`, `reference/*.npz` |
| Day 8 scoring | Built per design 7.3, `wrist_speed` left out (circular) | `padel/scoring.py` |
| Day 9 first hold-out check | Run; weak (below) | `scripts/eval_holdout.py` |
| Day 11 phone clip pose pipeline | Done: decode, pose, pick person, clean, fill gaps, smooth | `padel/pose.py`, `scripts/clip_to_keypoints.py` |
| Day 11 swing detection on player clips | Built (design 6.1); thresholds unchecked on real clips | `padel/swings.py` |
| Day 12 `analyze_clip` | Built: result JSON per design 3.4 with failure codes; checked on synthetic and static clips only | `padel/analyze.py` |
| Days 10, 12-13, 16 overlay, server, phone page, error messages | Built: H.264 overlay, FastAPI with one background worker, phone page, skeleton-only option | `padel/overlay.py`, `padel/server/` |
| Day 17 latency (first pass) | Measured per stage; pose dominates (CPU: 67 s per 10 s clip with YOLO11m, 34 s with s, 19 s with n) | `docs/measurements.md`, `scripts/bench_sizes.py` |
| Model choice write-up | Done: why YOLO11-pose and COCO-17, alternatives, evidence, open questions | `docs/model-choice.md` |

First hold-out result (80/20 stroke split, seed 0): held-out forehands score 94.0 against the forehand reference and 88.6 against the backhand one (higher for the right type in 80% of strokes); backhands 94.9 against 92.7 (61%). Scores saturate near 94, so the score is not usable yet.

### Left

| Priority | Item | Plan day |
|---|---|---|
| 1 | Score made continuous and tuned on a validation split (2026-11-21): ordering 86% / 65%, fault AUC 0.53 to 0.64, single-feature oracle only 0.47 to 0.78 at these fault sizes (`docs/results.md` 6b). Next: a tighter reference, or judge on own clips | 8-9, 15 |
| 2 | Flawed-stroke test run on real data (table in `docs/results.md` 6b); plot for the report still to do | 8-9 |
| 3 | Decide leave-players-out splits. Player identity is not tracked, so a leave-match-out split (women vs men) or a hand-labelled player map is needed | 7 |
| 4 | Compare dataset poses with re-run YOLO poses on about 50 windows | 9-10 |
| 5 | Try the phone app on real padel clips from a phone over Wi-Fi; check the swing thresholds and the "low reference confidence" warning (not built) | 11-13 |
| 6 | Film consented clips (20 per stroke, 2 players), failure log, own-clip evaluation, coach ratings if possible | 13-14, 16 |
| 7 | Hitter accuracy count on 40 windows, contact error in frames, handedness check by a padel player | 3-4 |
| 8 | Speed pass: decide the YOLO11 size for CPU-only laptops (m misses 60 s on CPU) after an accuracy check | 17 |
| 9 | Report, demo script and backup video, oral exam dry run | 18-20 |

Open outside the code: written instructor approval, whether the phone app meets the "camera sensor connected to MATLAB" requirement, demo timing, footage rights for the pro video, and a fifth member (the team has four). A padel player should settle the 7 ambiguous forehand labels.

## 2. What "done" means

A teammate who has never seen the code can:

1. Open the web page on a phone on the same Wi-Fi as the laptop running the server.
2. Pick hand (right/left) and stroke (forehand/backhand), record or choose a clip of one stroke.
3. Within 60 seconds (target, not yet measured) see an overlay video with skeleton, a similarity score
   from 0 to 100, confidence indicators, and the top three feature deviations in words.
4. If analysis is unreliable, see an explicit failure reason instead of a misleading score.
5. Repeat for five clips in a row without restarting the server.

Plus: an evaluation on held-out professionals and deliberately flawed swings, a written failure-case
list, and a report and demo that every member can explain.

## 3. Scope

| In | Out |
|---|---|
| User-selected forehand and backhand | Smash, serve, dropshot, bandeja, víbora |
| One fixed camera behind the baseline, one player | Automatic stroke classification |
| One fixed camera behind the baseline, one player | Multiple players, multiple cameras |
| Record, upload, get a result | Live scoring, ball tracking, shot speed |
| Score against a carefully filtered near-side reference from PadelTracker100 | Coach-validated "correct/incorrect" claims |
| Laptop server plus phone browser | Native iOS or Android app |

## 4. Team and ownership

Four members: Najeeb, Ashen, Moaz and Ahmed. The course asks for groups of 5, so confirm with the
instructor how a group of 4 is handled. Every member owns one area and reviews another, because every
member is examined on all code. **This split is a first draft and can change.**

| Area | Owner | Reviewer | Output |
|---|---|---|---|
| A. Data preparation | Najeeb | Ashen | Stroke database from PadelTracker100 |
| B. Pose and tracking | Ashen | Moaz | Keypoints from any video, smoothed |
| C. Features, reference, scoring | Moaz | Ahmed | Reference files and the `score_stroke` function |
| D. Web app and server | Ahmed | Najeeb | Phone UI, API, overlay video |
| E. Evaluation, recording, report | Shared (below) | The owner of the neighbouring area | Test clips, metrics, failure cases, report, demo |

Area E is split by task so no one person carries the largest grading weight:

| E task | Person |
|---|---|
| Filming, consent, clip log | Ahmed |
| Evaluation protocol | Najeeb |
| Running the evaluation and results | Moaz |
| Report drafting and failure-case log | Ashen |

The schedule tables below still show a letter (A to E) as the owner. Read it with the table above. The
conditional MATLAB track (section 10) was written for owners C and E, now Moaz plus the evaluation tasks.

Review chain: Najeeb reviews Ashen, Ashen reviews Moaz, Moaz reviews Ahmed, Ahmed reviews Najeeb.

Review rule: nothing merges until its reviewer can explain it to the owner without reading the code
aloud. This is also the oral-exam preparation.

## 5. Four-week schedule

Days are working days. "Done when" is a check someone else can run.

### Week 1: data, poses and the first features

Milestone M1 (end of week 1): the stroke database exists and the features look sensible on a sample.

| Day | Owner | Task | Done when |
|---|---|---|---|
| 1 | A | Extract the downloaded zip into its own folder. Load both pose JSONs and shot CSVs. Print counts and compare with `datasets.md`. | Counts match: 20,100 and 20,035 CSV rows; 45,934 and 53,888 pose frames |
| 1 | B | Install the chosen pose model; run it on the 30 MB sample clip. Time it. | Skeleton video produced; seconds-per-frame recorded in `docs/measurements.md` |
| 1 | E | Write the filming protocol (section 9). Book the court and tripod. Draft the consent form. | Protocol and consent form committed |
| 1 | C, D | Agree the file schemas in design.md section 3 and create the repo skeleton and empty tests. | Schemas merged; `pytest` runs (zero tests passing is fine) |
| 2 | A | Resolve the men's file left/right keypoint question: draw keypoints on 10 video frames for each match and look. | Written answer in `datasets.md`; loader reads the correct order per match |
| 2 | A | Group consecutive labelled frames of the same type into stroke windows. | `strokes_index.csv` with one row per window; count matches 293 forehand and 282 backhand windows |
| 2 | B | Write the person tracker for pose JSONs (match players across frames by hip-centre distance). | Each of the four players gets a stable track over a 1,000-frame sample, checked by eye |
| 3 | A | Infer the hitter per window: the track whose dominant-side wrist speed peaks highest, or for left/right unknown, whichever wrist is faster. | 40 windows hand-checked against video; accuracy recorded |
| 3 | A | Estimate each pro's handedness from the hitter's fast wrist across their forehands. | A table of 8 players with a handedness and the evidence |
| 3 | C | Implement the features (design.md section 4) on synthetic keypoints with known angles. | Unit tests pass: a straight arm gives 180°, a right angle gives 90° |
| 4 | A | Estimate the contact frame per window from the wrist-speed peak. Compare with video on 30 windows. | Median contact error in frames recorded; goes into `datasets.md` |
| 4 | A | Decide the camera-side filter: keep only players on the near side of the court (seen from behind), see design.md section 5. | Decision recorded with a sample image per side |
| 4 | B | Run pose on three 10-second phone clips from the team (any court). Measure keypoint availability. | Report: % frames where dominant arm keypoints have confidence at least 0.5 |
| 5 | C | Compute features for every near-side stroke window; plot per-feature curves for 20 forehands over each other. | Plot looks like a coherent family of curves, not noise |
| 5 | E | Draft the evaluation protocol (section 8) with numeric starting targets. | Protocol merged and agreed by all five |
| 5 | all | M1 review: each owner explains their code to their reviewer. | Five sign-offs |

### Week 2: reference, DTW scoring and first validation

Milestone M2 (end of week 2): `score_stroke` runs and reports similarity, confidence, and explicit
quality failures on held-out professional and synthetic test strokes.

| Day | Owner | Task | Done when |
|---|---|---|---|
| 6 | C | Implement DTW (cost matrix, accumulation, backtracking) in plain numpy. | Output matches a hand-worked 5×5 example in a unit test |
| 6 | C | Anchor strokes at contact and resample each to 60 steps. | A stroke plotted before and after resampling overlays correctly |
| 7 | C | Build the reference: align all strokes of a type to a medoid with DTW; store mean and standard deviation per step per feature. | `reference/forehand.npz` and `backhand.npz` with shapes in the design |
| 7 | A | Hold out players: split the 8 pros into build set and test set for leave-players-out evaluation. | `splits.json` committed; no player in both sides |
| 8 | C | Implement scoring (design.md section 6) and the feedback sentences. | Given a pro stroke, returns a score and three ranked deviations |
| 8 | E | Create flawed test strokes by editing real ones: add a fixed offset to one feature (for example reduce the elbow angle by 20°). | Script produces five fault types per held-out stroke |
| 9 | E | Run the first evaluation: held-out pro strokes vs flawed strokes. Report separation. | Table and one plot in `docs/results.md`; if separation is weak, record why |
| 9 | B | Run the pose model on about 50 pro stroke windows from the video, compute features, compare with the dataset-pose features. | Per-feature mean difference reported. This decides whether the reference uses dataset poses or re-run poses |
| 10 | C | Act on that comparison. If the gap is large, re-run poses on all near-side stroke windows and rebuild the reference. | Reference rebuilt or the decision not to is recorded with the numbers |
| 10 | D | Build the minimal server: upload endpoint that returns a hard-coded result. Open it from a phone. | A phone on the same Wi-Fi gets the JSON |
| 10 | all | M2 review. | Five sign-offs |

### Week 3: end-to-end with real video

Milestone M3 (end of week 3): record on a phone, get a real score and overlay.

| Day | Owner | Task | Done when |
|---|---|---|---|
| 11 | B | Pipeline for a player clip: decode video, run pose, pick the single player, smooth keypoints, fill short gaps. | One command turns an `.mp4` into a keypoint file |
| 11 | C | Swing detection on player clips: wrist-speed peaks, windowing around contact. | On a test clip with 5 known swings, finds 5 |
| 12 | C | Wire detection, features, reference and scoring into `analyze_clip(video, hand, stroke)`. | Returns the result structure from design.md section 3 |
| 12 | D | Overlay renderer: skeleton, stroke label, score, drawn on every frame; encode H.264 so phones can play it. | Overlay plays on a phone browser |
| 13 | D | Wire the server to `analyze_clip`; run it as a background job with a progress status. | Phone upload returns a job id; polling shows progress; final result shows overlay and text |
| 13 | E | Film the first real sessions with consent: the protocol clips (section 9). | At least 20 clips per stroke from at least 2 players, logged in `clips.csv` |
| 14 | E | Run the app on the real clips; log every failure with a screenshot. | `docs/failures.md` with at least 15 entries categorised |
| 14 | B | Fix the top two pose failures (for example the wrong person selected, dropped frames). | Re-run shows the failures fixed or documented as not fixable |
| 15 | C | Tune the scoring weights using only the build set, not the test set. | Weights fixed and committed before the final evaluation |
| 15 | all | M3 review and a full demo rehearsal on a phone. | Demo completes without restarting the server |

### Week 4: evaluation, hardening and delivery

Milestone M4 (end of week 4): final evaluation, report, demo.

| Day | Owner | Task | Done when |
|---|---|---|---|
| 16 | E | Final evaluation on the held-out pros and on the team's own clips, with the fixed weights. | `docs/results.md` complete with all metrics from section 8 |
| 16 | D | Error handling: no person found, several people, clip too short, wrong orientation, upload failure. | Each case shows a readable message, not a crash |
| 17 | B | Speed pass: measure per-stage time; remove the worst bottleneck. | Latency table before and after in `docs/measurements.md` |
| 17 | C | Explainability pass: make sure every feedback sentence names a feature, a direction and a size. | Ten sample feedback outputs read by the whole team |
| 18 | all | Freeze features. Only bug fixes after this point. | Tag `v1.0` |
| 18 | E | Write the report sections from results and the literature review. | First full draft |
| 19 | all | Dry-run the oral exam: each member explains and live-modifies a module they did not write. | Each member passes; gaps get another day |
| 19 | D | Demo script and a recorded backup demo. | Backup video exists |
| 20 | all | Final report and presentation; clean the repo; confirm nothing large or private is committed. | Submission package complete |

### Stretch tier (only if a milestone finishes early)

These are not part of the MVP and must not delay the required forehand/backhand pipeline. In priority
order:

1. Smash as a third stroke.
2. A swing-type classifier so the user does not choose the stroke.
3. Live mode: stream frames and score each stroke as it finishes.
4. A coach rating session on 30 clips to calibrate the pass/fail threshold.
5. Ball tracking and shot speed.

## 6. Long-term product roadmap

The four-week project is a validated technical foundation, not the complete product. The long-term
ambition is to develop a professional performance-analysis platform that could be evaluated for use by
high-performance organizations such as Aspire Academy in Qatar. The prototype must earn that ambition
through evidence; it should not imply that the MVP is ready for professional decision-making.

### Phase 1: Validate the foundation

- Test the user-selected forehand/backhand workflow with players and coaches.
- Measure usefulness, repeatability, confidence calibration, and failure rates.
- Confirm that feedback is understandable and does not turn similarity into a “correct/incorrect”
  judgement.
- Resolve professional-footage licensing, consent, retention, and access controls.

### Phase 2: Build a stronger reference

- Collect or license more professional footage with clear provenance.
- Include multiple players, handedness, body types, styles, camera positions, and match contexts.
- Add coach-labelled examples to calibrate feedback and validate whether deviations matter in practice.
- Maintain player-level and session-level evaluation splits to prevent data leakage.

### Phase 3: Expand analysis

- Add smash, serve, bandeja, víbora, dropshot, and other relevant padel strokes.
- Automatically detect stroke type and improve hitter and contact-frame estimation.
- Support longer sessions, multiple players, court-aware tracking, and robust camera calibration.
- Add ball and racket tracking, shot timing, placement, and speed where accuracy is sufficient.

### Phase 4: Professional academy platform

- Provide coach dashboards, athlete profiles, longitudinal progress, session comparison, and drill
  recommendations.
- Support secure team deployment, role-based access, consent and deletion workflows, and audit logs.
- Integrate coach review so automated feedback supports—not replaces—professional expertise.
- Run prospective validation with academy players before using the system in training decisions.

The roadmap is intentionally gated: each phase depends on validation and data quality from the previous
phase. A professional deployment would require additional technical, ethical, legal, and operational
review beyond this course project.

## 7. Dependencies and critical path

```
decision on pose model ──► B day 1 ──► B day 4 ──► B day 11 ──► C day 12 ──► D day 13 ──► demo
A data (days 1–4) ──► C features (day 5) ──► reference (day 7) ──► scoring (day 8) ──► C day 12
E filming (day 13) ──► evaluation (day 16)
```

The critical path runs through data preparation and the pose model. If either slips by two days, use
the cut list in section 11 immediately rather than hoping to catch up.

## 8. Evaluation protocol

These are **starting targets**, to be revised after week 1 data shows what is realistic. Agree them before
running experiments, and do not change them after seeing test results.

| What | Method | Starting target |
|---|---|---|
| Swing detection | Peaks within a dataset-labelled window (plus or minus 3 frames) count as hits, otherwise false alarms; measure on pros | Recall at least 0.8, precision at least 0.7 |
| Contact estimate | Compare with hand-marked contact on 30 pro windows | Median error at most 3 frames |
| Pose availability | Share of frames with confident dominant-arm keypoints on our clips | At least 90% |
| Score separation | Leave-players-out: held-out pro strokes vs their flawed copies; report the area under the ROC curve | At least 0.8 |
| Fault sensitivity | Per fault type (elbow, rotation, knee, wrist height, timing): smallest offset that changes the score by a set amount | Report, no target |
| Real-player sanity | Teammates' deliberate good form vs deliberate faults on camera | Faults score lower on average, report effect size |
| Latency | Seconds from upload finish to result for a 10-second clip | At most 60 s (laptop) |
| Failure rate | Share of clips ending in an error or unusable result | Report |

Notes:
- Flawed copies are synthetic edits of real strokes, so passing this test shows the score reacts to
  changes in the features it uses. It does not show the score matches a coach's judgement.
- Only a coach rating can support a claim that a stroke is "right" or "wrong". Without it, results are
  phrased as "similarity to professional technique".

## 9. Filming protocol

- **Camera:** phone on a tripod, landscape, 1080p, 30 fps (60 fps if the phone supports it and storage
  allows), exposure and focus locked.
- **Position:** behind the baseline on the centre line, as high as the tripod or a fence allows. The player
  hits from the near half of the court, seen from behind, like the near-side players in the pro footage.
- **Framing:** the whole body, racket and about 1 m of margin stay in view for the whole swing.
- **Record per clip:** player ID, handedness, stroke, camera height and distance, date, and whether the
  swing was good form or a deliberate fault (and which).
- **Per session:** 10 forehands and 10 backhands in good form; 5 of each with a named fault (elbow too low,
  no shoulder turn, contact too late, no knee bend, arm too straight).
- **Consent:** each filmed person signs before filming. Raw video is stored privately, not in git. Only
  keypoints and skeleton overlays leave the team.
- **Do not** change camera position within a session.

## 10. Conditional MATLAB track

Needed only if the instructor requires MATLAB work. Staff it from week 1 with owners C and E.

| Task | Week | Output |
|---|---|---|
| DTW from first principles (cost matrix, accumulation, backtracking), checked against the Python result on the same inputs | 2 | `dtw_scratch.m` and a comparison table |
| Synthetic stroke simulator: parametrised swing, noise, virtual camera projection | 2–3 | Sensitivity plots (offset vs score, noise vs score, camera angle vs score) |
| Evaluation plots from the Python keypoint and score files | 4 | Figures for the report |

The Python and MATLAB sides meet through files only (`.npz` converted to `.mat` or `.csv`).

## 11. Risk register and cut list

| Risk | Warning sign | Response |
|---|---|---|
| Pro reference is a poor fit for our camera view | Feature curves from our clips lie far outside the pro spread on good swings | Lower the number of features used; widen ranges; keep the claim to "relative to pros" |
| Dataset poses differ from our pose model | Day 9 comparison shows large feature gaps | Re-run our model on pro windows (about 600 windows, short) |
| Hitter or contact inference is wrong too often | Day 3–4 hand checks below 80% | Restrict to windows where one near-side player clearly dominates the motion |
| Pose fails on fast swings or motion blur | Many dropped wrist keypoints | Raise fps to 60; interpolate gaps up to 3 frames; document the limit |
| Processing is too slow | Over 60 s for 10 s of video | Process only the swing window, lower resolution, skip frames outside it |
| Phone cannot reach the server | Upload fails on the venue Wi-Fi | Fall back to a laptop hotspot; keep a recorded demo |
| Filming slips | No clips by day 13 | Film on any available court or indoors; weaken the camera-match claim |
| A member is unavailable | Owner absent for two days | Reviewer takes over; this is why reviewers exist |
| Left/right swap mishandled | Handedness wrong for many pros | Day 2 check; do not proceed to features until settled |

**Cut order if behind** (drop from the top first):

1. Stretch tier.
2. Overlay video (return the score and text only).
3. Per-feature feedback (return the score and confidence only).
4. Backhand (ship forehand only, while preserving the user-selected stroke contract).

The minimum viable result: user-selected forehand similarity against the filtered professional reference,
with confidence and failure reporting, validated on held-out professionals and synthetic perturbations.
Backhand is the second required stroke and should be included unless the documented cut list is invoked.

## 12. Report outline (maps to the rubric)

| Section | Rubric weight | Source |
|---|---|---|
| Problem, background, ethics, success criteria | 10% | [literature-review.md](literature-review.md), sections 1, 4 and 5 |
| Method and implementation | 20% | [design.md](design.md), plus the from-scratch DTW explained |
| Results, analysis, performance, failure cases | 30% | `docs/results.md`, `docs/failures.md`, `docs/measurements.md` |
| Report quality and presentation | 20% | Figures from the evaluation, clear structure |
| Teamwork and milestones | 10% | Git history, five sign-offs per milestone, ownership table |
| Creativity and oral defence | 10% | Phone app demo, live modification drill (day 19) |

Results carry the most weight, so the evaluation protocol is fixed in week 1 and run once, at the end.

## 13. Open decisions

- [ ] Written answers from the instructor (section 1). Python and AI use approved verbally on 2026-11-08; the phone-app and demo-timing questions are still open.
- [x] Which pose model: YOLO11-pose (COCO-17 output, finds the near-side players on broadcast footage; MediaPipe found almost none). Re-test MediaPipe on phone clips.
- [x] Near-side rule: replaced the hip y >= 600 px placeholder with the v2 court homography (feet at contact, court y > 10 m); 355 windows instead of 231 (`docs/design.md` section 5).
- [x] Left/right pose swaps: fixed in code before features (`docs/design.md` section 6.2).
- [x] Score separation: scorer made continuous, tuned on the build set only, hold-out re-run (`docs/results.md` 6b). The AUC target of 0.8 looks out of reach on this reference.
- [ ] Whether `wrist_speed` enters the score (currently left out, circular).
- [ ] Whether to keep the shoulder angle as a feature, since the literature shows it is the least reliable joint
  from 2D.
- [ ] Footage rights for the pro match video before any of it appears in the demo or report.
- [ ] Confirm the draft ownership split in section 4, and how a group of 4 is handled.
