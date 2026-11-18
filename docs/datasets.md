# Datasets

## Primary: PadelTracker100

- Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12926558/
- Licence: CC BY 4.0 (credit the authors)
- Contact listed in the paper: PadelTracker100Dataset@gmail.com

Use in this project: build the professional reference model and train the stroke-type classifier.

### Two versions on Zenodo

Both belong to one concept record (10.5281/zenodo.14653705). Checked through the Zenodo API.

| | v1: [14653706](https://zenodo.org/records/14653706) (2025-01-15) | v2: [17020011](https://zenodo.org/records/17020011) (2025-02-06) |
|---|---|---|
| Main file | `padel-data-labels.zip`, 7.1 GB | `labels.zip`, 70 MB |
| Match videos | **Yes**: two full `.mp4` (2.9 GB women's final, 4.3 GB men's final) plus a 30 MB sample clip | No |
| Labels | pose, ball, shots | pose, ball, shots, plus **court homography** (150–180 MB JSON per match) |
| Extras | notebook | notebook, `scripts.zip` (frame extraction, YOLO training configs) |

For technique work we need the video, so v1 is required. v2's labels are a superset but it ships no
footage. The two have the same pose and shot files in the sizes we compared, but we did not diff them.

### Verified by inspecting the files

| Item | Finding |
|---|---|
| Matches | 2022 WPT Master Final, Barcelona: `FinalF` (women) and `FinalM` (men). Four players each, so **eight players in total**. |
| Video | 1920×1080, 30 fps, single main camera 7.6 m high and 15.5 m behind the glass |
| Pose | COCO-format JSON, one per match. 45,934 frames (F) and 53,888 frames (M), 4 people per frame. Keypoints are `[x, y, visibility]` in pixels. |
| Pose has no track IDs | Each annotation has `bbox`, `keypoints`, `attributes` (`occluded`, `keyframe`) only. Following one player across frames needs our own tracking. |
| Shots CSV | **Semicolon-delimited**, columns `file_name;has_shot;category`. The paper describes the columns as Frame/Shot/Type; the files differ. |
| Shot coverage | Only the first ~20,100 frames per match (20,100 F rows, 20,035 M rows; 40,135 total). The pose files cover more frames than the shot labels do. |
| Classes | Forehand, Backhand, Smash, Serve, Other, Dropshot (capitalised), `0` for no shot |
| Shot windows | Contiguous labelled frames: median **10–17 frames** (about 0.3–0.6 s) for forehand, backhand and smash; serves 23–28. Runs range from 1 to 48 frames. |

Stroke counts (labelled windows, F + M): forehand 293, backhand 282, smash 216, serve 73, other 68,
dropshot 5. Dropshot is too rare to use.

### Problems found

1. **Left/right keypoint names differ between the two match files.** The women's file lists
   `left_eye, right_eye, ..., left_shoulder, right_shoulder` (standard COCO). The men's file lists
   `right_eye, left_eye, ..., right_shoulder, left_shoulder`, i.e. swapped. It is not yet known
   whether the men's coordinates are really swapped or only the names. This decides which arm is
   the dominant arm, so read the keypoint order from each file's `categories` and confirm on a
   frame by eye before computing any feature.
2. **The hitter is not labelled.** `has_shot` says a shot happens in that frame, not which of the
   four players hit it. We have to infer the hitter, for example as the player whose dominant wrist
   moves fastest in the window.
3. **Contact frame is not labelled.** The window is the hitter's racket movement; whether it starts at
   swing onset or ball contact is not stated. Contact must be estimated, for example as the wrist-speed
   peak inside the window.
4. **Handedness is not labelled.** Left- and right-handed players are not distinguished, and
   forehand/backhand depend on it. Estimate it per player from which wrist moves in the labelled
   windows.
5. **Narrow style coverage:** eight players from one tournament. State this in any result.

### Still unverified

- [ ] **Source-footage rights.** The authors redistribute the match video under CC BY 4.0, but the
  broadcast footage may belong to someone else (World Padel Tour / the broadcaster). We have not
  established that using or sharing it is allowed. The annotations are clearly CC BY 4.0; the video's
  status is unclear. Ask the authors, and avoid redistributing the video until it is resolved.
- [ ] Whether v1 and v2 pose and shot files are byte-identical (v2 is only known to be
  checksum-different as a zip).
- [ ] Where contact falls inside the shot window (needs viewing the video).
- [x] Which of the men's keypoint orderings is correct: **resolved 2026-11-08.** Drawing keypoints on near-side players (`scripts/check_lr.py`) shows the men's file has mislabelled names but standard COCO column order, so its names are ignored (`NAMES_TRUSTED` in `padel/data.py`). The women's file is correct as labelled.

## Other sources considered

| Source | Type | Use here |
|---|---|---|
| [padex](https://pypi.org/project/padex/) | Python tool, not a dataset | Possible reference; bandeja/víbora classification is only on its roadmap |
| [PadelVic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151013/) related work | Amateur match, multi-view | Not professional; background on position-estimation accuracy |
| [Comillas IMU dataset](https://repositorio.comillas.edu/jspui/handle/11531/66873) | Wearable sensors, 2,328 strokes, 12 players, 13 shot types | Not video; possible later for sensor-based validation |

The pose-estimation comparison study (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151013/) found
top-down methods most accurate for single-angle padel video, which informs the choice of pose model.

## Data we must collect ourselves

No public padel dataset labels technique quality, so these do not exist and have to be recorded:

1. **Player clips** from the target fixed camera setup, covering forehand and backhand.
2. **Coach ratings** (for example correct / needs work, plus the fault) on a subset of those clips.
   These calibrate the score thresholds and are the only way to measure whether the system's verdicts
   are right.

Keep raw video and downloaded datasets out of git; track only code, configs and small reference
files.

## Findings from the first pipeline pass (2026-11-08)

Scripts: `scripts/infer_hitters.py`, `handedness.py`, `finalize_index.py`, `check_hitters.py`. Output: `data/strokes_index_full.csv` (git-ignored; rebuild with the scripts).

- **Hitter** = the tracked player whose faster wrist has the highest torso-normalised peak speed in the window; **contact** = frame of that wrist's speed peak. On a 24-window contact sheet (men's match) the inferred player was visibly mid-swing in nearly every tile; this is an eyeball check, not the planned 40-window accuracy count, and contact error in frames is not measured.
- **Track IDs are not player identities.** Players cross the court and the tracker is position-based, so one ID spans near and far side over 20k frames (median hip y 305-684 px). Per-player handedness from track IDs therefore gave only 54-84% agreement and is not used.
- **Handedness is decided per window instead**: seen from behind, a forehand wrist is on the racket side, a backhand wrist on the other. Near-side windows split about 70% R / 30% L, which fits a mix of right- and left-handers. Left-handed strokes must be mirrored before scoring. The rule is unvalidated against known handedness.
- **Near-side filter**: hitter hip y at contact >= 600 px (placeholder threshold, no sample images per side yet). Usable near-side windows: 136 forehand, 95 backhand (231 of 575). 5 windows had no hitter found.
- The wrist-speed hitter rule can pick a wrong player or the non-racket wrist; this is the main known weakness.
- **Handedness (2026-11-18):** labelled by eye from the racket on contact sheets of all 355 near-side windows. One left-hander (women's final, dark kit, dark hair); everyone else right-handed. 342 labelled, 13 unclear. Several windows labelled Forehand show a right-hander with the racket on the backhand side, so some shot labels or contact frames are off; not yet counted.
