# Paddle Technique Recognition

Analyse a padel player's stroke technique from video and compare it against professional players.

Given a short recording from a single fixed camera, the system:

1. estimates the player's body pose in every frame,
2. analyzes one user-selected forehand or backhand,
3. measures pose-derived features of the stroke (joint angles, rotation, contact height, timing),
4. compares those features with a carefully filtered professional reference, and
5. reports an explainable similarity score, confidence indicators, and specific deviations.

> **Status:** pro-data pipeline, features, reference and scoring are built and tested; the score does not discriminate well enough yet and the phone clip analysis and web app are not built. This is a CMAI 480 (Computer Vision) group project
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
  label-suspects.md forehand labels that look wrong
padel/              pipeline package; scripts/ and tests/ alongside; reference/ holds the built references
```

The MATLAB track is conditional and not started.

## Requirements (planned)

- Python 3.10+ (numpy, FastAPI, OpenCV, ffmpeg)
- An instructor-permitted COCO-17 pose model (to confirm)
- A phone on a tripod as the fixed camera, and a laptop as the server
- MATLAB only if the instructor requires it (see the conditional track in the plan)

## Current status (2026-11-18)

The specification and plan are in `docs/`; `docs/project-plan.md` has the full done and left lists. Implemented (`padel/`, `scripts/`, `tests/`, 37 tests):

- PadelTracker100 loaders and stroke-window grouping (575 windows: 293 forehand, 282 backhand)
- A position-based four-player tracker; per-window hitter and contact frame
- Court homography near-side filter (355 usable windows: 194 forehand, 161 backhand)
- Handedness labelled by eye (311 right, 31 left, 13 unclear); 13 suspect forehand labels recorded in `docs/label-suspects.md`
- Pose model chosen (YOLO11-pose, `docs/measurements.md`); phone clip pose extraction with cleaning, gap filling and smoothing
- Ten per-frame features, with a fix for the pose model's mid-stroke left/right swaps
- Hand-written DTW, medoid reference (`reference/forehand.npz`, `reference/backhand.npz`) and scoring with feedback

First hold-out check: held-out strokes score about 94 against the right reference and 89-93 against the wrong one, so the score barely discriminates (61-80% correct ordering). Tuning it is next.

Not built: swing detection on phone clips, `analyze_clip`, overlay video, FastAPI server and phone page, flawed-stroke and own-clip evaluation, failure log, report.
Still open: rights to use the professional match video, written confirmation of the instructor's approval, the phone-app vs "camera sensor connected to MATLAB" question, and a fifth member.

Run: `py -m pip install numpy opencv-python pytest scipy ultralytics mediapipe matplotlib`, then `py -m pytest -q`.
Pipeline order (each script documents its arguments): `build_strokes_index.py`, `infer_hitters.py`, `finalize_index.py`,
`build_stroke_features.py`, `build_reference.py`; `eval_holdout.py` runs the hold-out check.
