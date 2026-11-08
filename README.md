# Paddle Technique Recognition

Analyse a padel player's stroke technique from video and compare it against professional players.

Given a short recording from a single fixed camera, the system:

1. estimates the player's body pose in every frame,
2. analyzes one user-selected forehand or backhand,
3. measures pose-derived features of the stroke (joint angles, rotation, contact height, timing),
4. compares those features with a carefully filtered professional reference, and
5. reports an explainable similarity score, confidence indicators, and specific deviations.

> **Status:** Week 1 data pipeline built; features, reference, scoring and the web app are not started. This is a CMAI 480 (Computer Vision) group project
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
```

MATLAB source folders are listed in [docs/design.md](docs/design.md) and are not created yet.

## Requirements (planned)

- Python 3.10+ (numpy, FastAPI, OpenCV, ffmpeg)
- An instructor-permitted COCO-17 pose model (to confirm)
- A phone on a tripod as the fixed camera, and a laptop as the server
- MATLAB only if the instructor requires it (see the conditional track in the plan)

## Current status

The specification and plan are in `docs/`. Implemented so far (`padel/`, `scripts/`, `tests/`):

- PadelTracker100 loaders and stroke-window grouping (575 windows: 293 forehand, 282 backhand)
- A position-based four-player tracker
- Per-window hitter, contact frame, near-side flag and handedness (231 usable near-side windows)
- Pose model chosen: YOLO11-pose (see `docs/measurements.md`)

Not started: features, reference and DTW scoring, phone clip pose extraction, web app, evaluation.
Still open: rights to use the professional match video, and written confirmation of the instructor's approval.

Run: `py -m pip install numpy opencv-python pytest scipy ultralytics mediapipe`, then `py -m pytest -q`;
the pipeline scripts take the dataset labels folder as their first argument.
