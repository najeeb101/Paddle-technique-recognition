# Paddle Technique Recognition

Analyse a padel player's stroke technique from video and compare it against professional players.

Given a short recording from a single fixed camera, the system:

1. estimates the player's body pose in every frame,
2. analyzes one user-selected forehand or backhand,
3. measures pose-derived features of the stroke (joint angles, rotation, contact height, timing),
4. compares those features with a carefully filtered professional reference, and
5. reports an explainable similarity score, confidence indicators, and specific deviations.

> **Status:** pro-data pipeline, features, reference and scoring are built and tested; the phone app is built but not yet tried on a real clip, and the score separates flawed strokes only weakly. This is a CMAI 480 (Computer Vision) group project
> on a 4-week schedule. The plan is Python-first with a phone web app. The instructor approved Python and AI assistance (reported verbally, 2026-11-08). See [docs/project-plan.md](docs/project-plan.md) for course
> requirements and timeline, [docs/design.md](docs/design.md) for the planned pipeline, and
> [docs/datasets.md](docs/datasets.md) for the data sources.

## Scope (MVP)

| Decision | Choice | Why |
|---|---|---|
| Camera | Single fixed camera | Keeps pose geometry consistent; no multi-view calibration |
| Strokes | User-selected forehand or backhand | Avoids an unvalidated automatic classifier |
| Reference | Filtered near-side professional strokes from PadelTracker100 | Reduces viewpoint and style mismatch |
| Output | Similarity score, confidence, failure reason, and feature feedback | Explainable; works without "wrong technique" labels |

Out of scope: smash, serve, dropshot, bandeja/víbora, automatic stroke classification, ball tracking,
shot speed, multi-player analysis, live scoring, and medical or “correct/incorrect” judgements.

## Long-term vision

The MVP is deliberately narrow; it is not the final product vision. The longer-term goal is a
professional performance-analysis platform that could support player development at organizations such
as Aspire Academy in Qatar. A production system would combine reliable camera-based technique analysis
with coach oversight, athlete progress tracking, and a much broader reference library.

The next stages are:

1. **Validate the MVP:** test forehand and backhand analysis with coaches and players, quantify
   confidence and failure rates, and establish whether the feedback is useful in practice.
2. **Broaden the reference:** collect licensed, diverse professional footage across players, handedness,
   camera positions, playing styles, and stroke situations.
3. **Expand stroke coverage:** add smash, serve, bandeja, víbora, dropshot, and other strokes only after
   each can be evaluated with suitable data.
4. **Improve automatic analysis:** detect the stroke type, identify the hitter, estimate contact more
   accurately, and support multiple players and longer training sessions.
5. **Add performance context:** combine pose with racket, ball, court-position, timing, and (where
   appropriate) sensor data to study technique and tactical outcomes.
6. **Build the academy product:** coach dashboards, athlete profiles, longitudinal progress, session
   comparison, drill recommendations, privacy controls, and secure deployment for professional programs.

These are future directions, not claims about the current prototype. Any deployment for professional
players would require additional validation, data licensing, privacy review, and coach involvement.

## Key idea: why a reference model, not a classifier

No public padel dataset labels technique as good or bad. Professional footage is labelled by
*stroke type* only. The MVP therefore reports **similarity to a filtered professional reference**,
feature by feature; it does not claim to identify objectively correct or incorrect technique.

## Planned layout

```
docs/
  project-plan.md   course requirements, MVP scope, timeline, success criteria, open decisions
  literature-review.md   problem definition, related work, gap, ethics
  design.md         pipeline, features, scoring, assumptions
  datasets.md       data sources, licences, caveats
  measurements.md   benchmark and pipeline numbers
  pipeline.md       step-by-step guide to the code, files and methods
  results.md        every finding so far, failures, and what is left
  label-suspects.md forehand labels that look wrong
  model-choice.md   why YOLO11-pose and COCO-17, alternatives, evidence
padel/              pipeline package; scripts/ and tests/ alongside; reference/ holds the built references
```

The MATLAB track is conditional and not started.

## Requirements (planned)

- Python 3.10+ (numpy, FastAPI, OpenCV, ffmpeg)
- An instructor-permitted COCO-17 pose model (to confirm)
- A phone on a tripod as the fixed camera, and a laptop as the server
- MATLAB only if the instructor requires it (see the conditional track in the plan)

## Current status (2026-11-20)

The specification and plan are in `docs/`; `docs/project-plan.md` has the full done and left lists. Implemented (`padel/`, `scripts/`, `tests/`, 62 tests):

- PadelTracker100 loaders and stroke-window grouping (575 windows: 293 forehand, 282 backhand)
- A position-based four-player tracker; per-window hitter and contact frame
- Court homography near-side filter (355 usable windows: 194 forehand, 161 backhand)
- Handedness labelled by eye (311 right, 31 left, 13 unclear); 13 suspect forehand labels recorded in `docs/label-suspects.md`
- Pose model chosen (YOLO11-pose, `docs/measurements.md`); phone clip pose extraction with cleaning, gap filling and smoothing
- Ten per-frame features, with a fix for the pose model's mid-stroke left/right swaps
- Hand-written DTW, medoid reference (`reference/forehand.npz`, `reference/backhand.npz`) and scoring with feedback
- Phone app: swing detection on clips, `analyze_clip` (result JSON with failure codes), H.264 skeleton overlay, FastAPI server and phone page
- Per-stage latency and YOLO11 size benchmark (`docs/measurements.md`); model choice explained in `docs/model-choice.md`

Scoring (2026-11-21): held-out strokes score higher against their own type in 86% (forehand) and 65% (backhand) of cases. Flawed-stroke AUC is 0.53 to 0.64; the professionals' own spread caps it near 0.5 to 0.78 at these fault sizes (`docs/results.md` 6b).

Not done: trying the phone app on a real padel clip, own-clip evaluation, failure log, report.
Still open: rights to use the professional match video, written confirmation of the instructor's approval, the phone-app vs "camera sensor connected to MATLAB" question, and a fifth member.

Run: `py -m pip install numpy opencv-python pytest scipy ultralytics mediapipe matplotlib fastapi uvicorn python-multipart`, then `py -m pytest -q`.

Phone app: put `yolo11m-pose.pt` in a `models/` folder next to the repo (or set `PADEL_WEIGHTS`), have ffmpeg on the PATH,
run `py -m padel.server`, and open the printed address on a phone on the same Wi-Fi.
Pipeline order (each script documents its arguments): `build_strokes_index.py`, `infer_hitters.py`, `finalize_index.py`,
`build_stroke_features.py`, `build_reference.py`; `eval_holdout.py` runs the hold-out check.
