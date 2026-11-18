# Design

Technical specification for the 4-week plan in [project-plan.md](project-plan.md). Data facts are in
[datasets.md](datasets.md). Numeric choices below are proposals to be confirmed by the measurements named
in the plan; they are not yet tested.

## 1. System overview

```
phone browser ──upload──► server (Python) ──► pose ──► track & smooth ──► swing detection
      ▲                                                                        │
      │                                                                        ▼
      └──── result JSON + overlay video ◄── scoring ◄── reference ◄── features ┘
                                              ▲
                  PadelTracker100 ──► stroke database ──► reference build (offline)
```

The MVP supports two user-selected stroke types: forehand and backhand. It does not attempt automatic
stroke classification.

Two separate flows share the same feature and scoring code:

- **Offline build (once):** turn PadelTracker100 into a stroke database, then references for the supported
  forehand and backhand types.
- **Online analysis (per clip):** turn a phone video into keypoints, find the stroke, score it.

Sharing one implementation of the features is important: if the reference and the player are computed by
different code, any mismatch looks like a technique difference.

## 2. Technology choices

| Part | Proposal | Why | Alternative |
|---|---|---|---|
| Language | Python 3.10+ | Pose models and web servers are easiest here | MATLAB for some parts if the course requires it |
| Numerics | numpy only for features, DTW, scoring | Easy to test and to explain; DTW written by hand | scipy for smoothing filters |
| Pose model | YOLO11-pose (Ultralytics, COCO-17, multi-person) | Chosen 2026-11-08 from a benchmark on the sample clip: it found both near-side players, MediaPipe almost none; dataset poses are COCO-17 too. AGPL-3.0 licence | MediaPipe for close-up phone clips (untested); smaller YOLO11 sizes if too slow on CPU (4.4 fps for the medium model) |
| Server | FastAPI | Simple file upload and background jobs | Flask |
| Phone UI | A web page opened in the phone browser | No native app, no store, works on iOS and Android | Native app (out of scope) |
| Overlay | OpenCV drawing, encoded as H.264 with ffmpeg | Phones play H.264 reliably | |
| Storage | Files on disk (`.npz`, `.csv`, `.json`) | Nothing to install | A database |

Phone upload uses an `<input type="file" accept="video/*" capture>` control. It can offer camera capture
or gallery selection depending on the browser. The MVP analyzes an uploaded short clip; live camera
streaming is explicitly out of scope.

## 3. Data contracts

Plain files so each owner can work without the others' code.

### 3.1 Keypoints: `(T, 17, 3)` float32 array

`x, y` in pixels of the source video, `conf` in [0, 1]. COCO-17 order (0-based):

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | nose | 9 | left wrist |
| 1 | left eye | 10 | right wrist |
| 2 | right eye | 11 | left hip |
| 3 | left ear | 12 | right hip |
| 4 | right ear | 13 | left knee |
| 5 | left shoulder | 14 | right knee |
| 6 | right shoulder | 15 | left ankle |
| 7 | left elbow | 16 | right ankle |
| 8 | right elbow | | |

The men's pose file in PadelTracker100 names the keypoints with left and right swapped relative to this
order. The loader must read each file's own `categories` list and convert to the table above after the
day-2 check. Dataset visibility flags (0, 1, 2) are converted to a confidence of 0, 0.5 and 1.

### 3.2 Stroke database (output of area A)

`data/processed/strokes_index.csv`, one row per stroke window:

| Column | Meaning |
|---|---|
| `stroke_id` | Unique ID, e.g. `FinalF_000123` |
| `match` | `FinalF` or `FinalM` |
| `player_track` | Track ID from our tracker |
| `player_id` | One of eight pros |
| `side` | `near` or `far` (see section 5) |
| `type` | `forehand` or `backhand` |
| `handedness` | `right` or `left`, estimated |
| `start_frame`, `contact_frame`, `end_frame` | Frame numbers in the match video |
| `hitter_confidence` | Score from the hitter-inference step |

`data/processed/keypoints_{match}.npz`: for each `stroke_id`, a `(T, 17, 3)` array covering the window plus
a margin of 10 frames each side (so wrist speed at the edges is valid). Large files stay out of git.

### 3.3 Reference (output of area C)

`reference/{type}.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `mean` | `(N, F)` | Per-step, per-feature mean, with `N = 60` |
| `std` | `(N, F)` | Per-step standard deviation, floored (section 7) |
| `feature_names` | `(F,)` | Order of columns |
| `n_strokes` | scalar | Strokes used |
| `players` | list | Player IDs used |

Strokes are time-normalised so that index `contact_idx` (proposed: step 40 of 60) is ball contact.

### 3.4 Analysis result (output of `analyze_clip`)

```json
{
  "clip": "clip_0007.mp4",
  "hand": "right",
  "stroke_type": "forehand",
  "n_swings_found": 1,
  "strokes": [
    {
      "start_frame": 120, "contact_frame": 133, "end_frame": 150,
      "score": 72.4,
      "confidence": 0.86,
      "feature_scores": {"elbow_angle": 0.81, "...": 0.0},
      "feedback": [
        {"feature": "elbow_angle", "phase": "contact", "direction": "lower", "size_deg": 18,
         "text": "Elbow is about 18 degrees more bent than the professionals at contact."}
      ],
      "pose_quality": {"frames": 31, "confident_fraction": 0.94}
    }
  ],
  "warnings": [],
  "failure": null
}
```

`confidence` summarizes pose quality, stroke-isolation quality, and how far the stroke is from the
reliable professional reference. If a required quality check fails, `failure` contains a stable error
code and human-readable message and `strokes` is empty; the system does not return a misleading score.

### 3.5 Server API

| Method and path | Purpose |
|---|---|
| `POST /analyze` | Multipart upload: `video`, `hand`, `stroke_type`. Returns `{ "job_id": ... }` |
| `GET /status/{job_id}` | `queued`, `running` with a stage name, `done`, or `error` with a message |
| `GET /result/{job_id}` | The JSON in 3.4 |
| `GET /video/{job_id}` | The H.264 overlay video |

Jobs run one at a time in a background worker, so the pose model is not loaded twice.

## 4. Features

All features are computed per frame, so a stroke becomes an `(T, F)` matrix. "Dominant" means the arm the
player hits with. Distances are divided by **torso length** = distance from mid-shoulder to mid-hip,
measured per frame, so features do not depend on how far the player is from the camera.

Left-handed players are mirrored horizontally (x → image width − x) so one reference serves both
handednesses. Mirroring has to be checked on video before it is trusted, because it also flips the apparent
direction of body rotation.

| Feature | Definition | Unit |
|---|---|---|
| `elbow_angle` | Angle at the dominant elbow between the shoulder and wrist | degrees |
| `shoulder_angle` | Angle at the dominant shoulder between the hip and elbow | degrees |
| `knee_angle_dom`, `knee_angle_off` | Angle at each knee between hip and ankle | degrees |
| `shoulder_rotation` | Angle of the shoulder-to-shoulder vector in the image plane | degrees |
| `hip_rotation` | Angle of the hip-to-hip vector in the image plane | degrees |
| `separation` | `shoulder_rotation − hip_rotation` | degrees |
| `torso_lean` | Angle of the mid-hip to mid-shoulder vector from vertical | degrees |
| `wrist_height` | (shoulder y − wrist y) of the dominant arm divided by torso length | torso lengths |
| `wrist_speed` | Frame-to-frame dominant wrist displacement divided by torso length, times fps | torso lengths per second |

Notes for the team:

- A 2D angle is a projection onto the image. The elbow angle is stable if the camera sees the arm side-on
  or from behind, and gets less reliable when the arm points at the camera. That is why the camera
  position is fixed and recorded.
- The literature suggests the shoulder is the least reliable joint from 2D pose, so `shoulder_angle` is
  the first feature to drop if evaluation shows it adds noise.
- `wrist_speed` is used for finding swings and contact. Whether it also enters the score is a decision for
  week 2.

## 5. Pro data selection

The broadcast camera sits behind one baseline, so **near-side players are seen from behind and far-side
players from the front**. They look very different in 2D, and left and right appear mirrored between the
two sides. Our own camera also sees the player from behind. So:

1. Classify each stroke's hitter as near or far from their court position (homography, below).
2. Build the reference from near-side strokes only.
3. Report how many strokes this removes. If too few remain for a stroke type, revisit; do not silently mix
   the sides.

Implemented with the PadelTracker100 v2 court homography (`padel/court.py`): one constant pixel-to-metre matrix per match maps the hitter's feet (mean of the ankles, else the hips) at the contact frame onto the court; near side means court y > 10 m (the net). 355 of 575 windows pass (194 forehand, 161 backhand). The earlier hip y >= 600 px rule gave 231, a strict subset. Players change ends and cross the court, so the side is decided per stroke window, not per tracked player; track IDs are not player identities.

Handedness is also decided per window: seen from behind, the racket-wrist side at contact versus the stroke type gives right or left (about 70% right in the near-side windows). Left-handed strokes are mirrored. This rule is unvalidated against known handedness.

## 6. Pipeline stages

### 6.1 Swing detection (for player clips)

1. Compute `wrist_speed` for the dominant wrist from smoothed keypoints.
2. Smooth it (moving average, about 5 frames).
3. Find local maxima above a threshold set as a multiple of the clip's median speed. Enforce a minimum gap
   (proposal: 0.8 s) between swings.
4. Contact is the peak frame. The stroke window runs from the last frame before the peak where speed was
   below a low threshold to the first frame after the peak where it falls below it, bounded to
   about 0.7 s each side.

For pro data the same peak finder runs inside each dataset-labelled window, which gives the contact
estimate. The dataset does not label contact, so this is an estimate checked against video (day 4).

### 6.2 Smoothing and gap filling

- Mark keypoints with confidence below 0.3 as missing.
- Linearly interpolate gaps of up to 3 frames, otherwise leave the frame invalid.
- Smooth with a short filter (proposal: Savitzky–Golay, window 7, order 2). Do not smooth so heavily that
  the fast part of the swing is flattened; check on a real clip.

### 6.3 Player selection on user clips

- One person in view: use them.
- More than one: pick the person with the largest box, and show a warning. The user is told to film one
  player at a time.

### 6.4 Tracking pro players (offline)

The dataset has four annotations per frame and no IDs. Match each frame's four people to the previous
frame's by smallest distance between hip centres (Hungarian assignment). Check on a 1,000-frame sample by
eye; fix identity swaps near the net where players cross.

## 7. Reference building and scoring

### 7.1 Reference

For each stroke type:

1. Compute the `(T_i, F)` feature matrix for every near-side stroke `i` of the build players.
2. Resample each to `N = 60` steps with contact at a fixed step (proposal: 40). Time before and after
   contact are resampled separately so contact stays aligned.
3. Pick a medoid: the stroke with the smallest total DTW distance to all others.
4. DTW-align every stroke to the medoid and average along the alignment path. Store the mean and the
   standard deviation per step per feature.
5. Floor the standard deviation (proposal: a small fraction of the feature's overall spread) so a feature
   that happens to be very consistent does not make every small difference look huge.

### 7.2 DTW (from first principles)

For two sequences `a` (length `n`) and `b` (length `m`) of feature vectors:

- Local cost `d(i, j)` = Euclidean distance between feature vectors (after per-feature standardisation).
- Accumulated cost `D(i, j) = d(i, j) + min(D(i−1, j), D(i, j−1), D(i−1, j−1))`, with `D(0, 0) = d(0, 0)`.
- The warping path is found by backtracking from `D(n−1, m−1)` to `D(0, 0)` along the smallest neighbour.
- A band constraint (proposal: 20% of length) keeps the path from warping unrealistically.

It is implemented by hand so each member can explain it; tests compare it with a small worked example.

### 7.3 Scoring a stroke

1. Compute the stroke's features, resample to 60 steps with contact aligned.
2. DTW-align it to the reference mean.
3. For each feature `f` and aligned step `t`, `z(f, t) = (value − mean) / std`.
4. Per-feature score: the share of steps with `|z| ≤ 2`. Steps near contact (proposal: steps 30 to 50 of 60)
   count double.
5. Overall score: weighted mean of the feature scores, times 100. Weights start equal and are tuned on the
   build set only (plan day 15), never on the test set.
6. Feedback: the three features with the lowest scores, each with the phase where the deviation is largest
   (before, at or after contact), the direction (higher or lower than the reference mean) and the size in
   the feature's own unit.

The score means similarity to professional technique. It is not a verified measure of correctness.

## 8. Phone web app

### 8.1 Screens

1. **Setup:** choose hand and either forehand or backhand. A diagram shows where to place the camera.
2. **Record or upload:** opens the phone camera or gallery.
3. **Processing:** progress bar with stage names from `/status`.
4. **Result:** overlay video, similarity score, confidence indicators, three feedback lines, and a
   "record again" button.
5. **Error:** a plain message and what to do (see below).

### 8.2 Error messages the app must show

| Condition | Message idea |
|---|---|
| No person found | "No player detected. Step back so your whole body is visible." |
| Several people | "More than one person detected. Using the largest; film one player." |
| No swing detected | "No stroke found. Make sure the swing is inside the clip." |
| Clip too long | "Clip is over 20 seconds. Trim to one or two strokes." |
| Low pose quality | "Pose tracking was unreliable on this clip (X% of frames). Result may be wrong." |
| Low reference confidence | "This stroke is outside the reliable professional reference range. Treat the score cautiously." |
| Ambiguous stroke | "The selected stroke could not be isolated reliably. Record one clear forehand or backhand." |
| Server unreachable | "Cannot reach the server. Check that you are on the same Wi-Fi." |

### 8.3 Privacy

Overlays can be saved with the video shown or with a skeleton on a blank background. Default to the
skeleton-only video on shared devices. Uploaded raw video is deleted after processing unless a flag
keeps it for evaluation, and only when the filmed person consented.

## 9. Repository layout (proposed)

```
src/padel/
  data/        load PadelTracker100, tracking, hitter and contact inference, stroke database
  pose/        pose model wrapper, smoothing, player selection
  features.py  per-frame features
  dtw.py       DTW from first principles
  reference.py reference build and load
  scoring.py   score and feedback
  analyze.py   analyze_clip
  server/      FastAPI app and the web page
tests/         unit tests on synthetic keypoints
reference/     small reference files (committed)
data/          large data (not committed)
docs/          this documentation
```

## 10. Testing

- **Features:** synthetic skeletons with known angles (a straight arm is 180°, a right-angle elbow is 90°).
- **DTW:** a hand-worked 5×5 example; two identical sequences give distance 0; a stretched copy of a
  sequence aligns back to it.
- **Swing detection:** a synthetic wrist trajectory with three known peaks.
- **Scoring:** the reference mean scores 100; adding a constant offset to one feature lowers that
  feature's score and leaves the others near unchanged.
- **End to end:** the 30 MB sample clip produces a result without error.

## 11. Known limits

- Eight professionals from one tournament give narrow style coverage, and their handedness is unknown
  until it is estimated from the data (plan day 3).
- 2D angles shift with camera position, so scores from a very different camera are not comparable.
- A score is similarity to professionals, not a verdict, until a coach rates clips.
- Padel glass walls and shadows may confuse the pose model; failures are logged in `docs/failures.md`.
- Latency is not yet measured.
- The user selects forehand or backhand; automatic stroke-classification accuracy is not an MVP metric.
- The score is a similarity measure with confidence and failure checks, not a correctness judgement.

## 12. Future product direction

The current architecture is intended to make the MVP testable while leaving room for a larger
professional system. Future versions could add:

- a licensed multi-player professional reference library with metadata for player, handedness, camera
  view, stroke context, and coach labels;
- automatic stroke classification and better hitter/contact detection;
- camera calibration, court geometry, ball and racket tracking, and multi-camera or 3D analysis;
- session-level aggregation, athlete progress trends, coach annotations, and drill recommendations;
- secure academy deployment with authentication, role-based access, consent, retention, and audit logs.

These capabilities must be introduced only after the MVP's similarity score and confidence reporting are
validated. The system should remain a decision-support tool for coaches, not an unsupervised replacement
for professional judgement.
