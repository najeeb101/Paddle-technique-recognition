# Paddle Technique Recognition

Analyse a padel player's stroke technique from video and compare it against professional players.

Given a fixed-camera recording of a player, the system:

1. estimates the player's body pose in every frame,
2. finds each stroke and classifies its type,
3. measures biomechanical features of the stroke (joint angles, rotation, contact height, timing),
4. compares those features with a reference built from professional footage, and
5. reports a score plus the specific deviations ("elbow too low at contact").

> **Status:** planning. Nothing is implemented yet. This is a CMAI 480 (Computer Vision) group project
> on a 4-week schedule. The plan is Python-first with a phone web app; the course asks for MATLAB, so
> the allowance and any MATLAB work are open items in [docs/project-plan.md](docs/project-plan.md). See [docs/project-plan.md](docs/project-plan.md) for course
> requirements and timeline, [docs/design.md](docs/design.md) for the planned pipeline, and
> [docs/datasets.md](docs/datasets.md) for the data sources.

## Scope (first version)

| Decision | Choice | Why |
|---|---|---|
| Camera | Single fixed camera | Keeps pose geometry consistent; no multi-view calibration |
| Strokes | Forehand, backhand, smash | Present as labelled classes in the public pro dataset |
| Reference | Professional matches (PadelTracker100) | Only open padel dataset with stroke labels and pose |
| Output | Per-stroke score + per-feature feedback | Explainable; works without "wrong technique" labels |

Out of scope for now: serve, dropshot, bandeja/víbora, ball tracking and shot speed, multi-player
analysis, live (real-time) use.

## Key idea: why a reference model, not a classifier

No public padel dataset labels technique as good or bad. Professional footage is labelled by
*stroke type* only. So "right vs wrong" is defined as **distance from the professional range** for
that stroke type, feature by feature. If a coach later rates clips of real players, those ratings can
be used to calibrate the score thresholds, but the system does not depend on them.

## Planned layout

```
docs/
  project-plan.md   course requirements, timeline, success criteria, open decisions
  literature-review.md   problem definition, related work, gap, ethics
  design.md         pipeline, features, scoring, assumptions
  datasets.md       data sources, licences, caveats
```

MATLAB source folders are listed in [docs/design.md](docs/design.md) and are not created yet.

## Requirements (planned)

- Python 3.10+ (numpy, FastAPI, OpenCV, ffmpeg)
- A COCO-17 pose model the instructor permits (to confirm)
- A phone on a tripod as the fixed camera, and a laptop as the server
- MATLAB only if the instructor requires it (see the conditional track in the plan)
