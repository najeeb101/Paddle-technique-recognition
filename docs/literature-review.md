# Literature review and problem definition

Supports the "Literature Review & Problem Definition" criterion (10%): background review, ethical
assessment, measurable success criteria and a clear scope.

> **Reading status.** The summaries below come from search-result abstracts and snippets, not from reading
> the full papers. Before any of this goes into the report, each team member should read the sources they
> cite, because every member can be questioned on them at the oral exam. The reference list marks how far
> each source was checked.

## 1. Problem definition

Padel players improve by getting feedback on technique, but that normally needs a coach present. We
propose a camera-based system that estimates a player's body pose during user-selected forehand and
backhand strokes, measures how the stroke differs from professional players' strokes, and reports which
body movements differ.

**Core question:** can 2D pose from a single fixed camera support a meaningful, explainable comparison
between a player's stroke and professional strokes?

**What the system outputs:** per-stroke similarity to professional technique plus the specific features
that deviate. It does not claim to prove a stroke is objectively wrong, because no public data labels
technique quality (see [datasets.md](datasets.md)).

**Scope**

| In scope | Out of scope |
|---|---|
| User-selected forehand and backhand | Smash, serve, dropshot, bandeja, víbora |
| Single fixed camera, one player at a time | Multi-camera 3D, multi-player analysis |
| Similarity to professional range, per-feature feedback | Injury risk, medical advice, match tactics |
| Recorded video uploaded from a phone | Live scoring, ball speed and placement |

## 2. Related work

### 2.1 Pose estimation in sport

- **Padel position and pose from a single camera.** A comparison of vision methods on padel high-angle
  video found that top-down pose estimation gave the best results, with a detection rate of 99.8% and
  court-space errors below 5 cm horizontally and 12 cm vertically ([Sensors 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151013/)).
  This supports using a top-down pose model. It measured player *position*, not joint angles, so it does
  not show that limb angles are accurate.
- **Validity of 2D pose for joint angles is mixed.** A comparison of OpenPose against marker-based motion
  capture across four camera viewpoints found reasonable correlations for knee, hip and elbow but not for
  the shoulder, and significant bias between viewpoints, with back-facing cameras performing best
  ([PMC11819822](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11819822/)). In real athletics competition
  footage, OpenPose was judged unsuitable for long-jump analysis because of poor agreement and unrealistic
  values ([Leeds Beckett](https://eprints.leedsbeckett.ac.uk/id/eprint/10344/8/FeasibilityOfOpenposeMarkerlessMotionAnalysisInARealAthleticsCompetitionPV-WALKER.pdf)).
  This matters for us: shoulder angle is one of our features, and padel strokes are fast.

### 2.2 Stroke recognition from pose in racket sports

- **Table tennis:** a temporal CNN on 2D pose classified 11 stroke types, reporting 99.37% validation
  accuracy ([CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Kulkarni_Table_Tennis_Stroke_Recognition_Using_Two-Dimensional_Human_Pose_Estimation_CVPRW_2021_paper.html)).
  This is on a controlled setup, so it is an upper bound for what to expect, not a prediction.
- **Badminton:** skeleton-based models (pose from RTMPose or MMPose, then a graph or transformer
  network) classify stroke types, and court-line filtering is used to pick out the two players
  ([BST, arXiv 2502.21085](https://arxiv.org/pdf/2502.21085)).

The same pipeline shape (pose, then a temporal model) transfers to padel. In the searches we ran, no
padel stroke-classification paper appeared.

### 2.3 Comparing a performance with an expert

- **Tennis serve pro vs amateur.** A study ran pose estimation on serve videos and compared professional and
  amateur players using Euclidean distances between joints, finding a significant difference in consistency,
  notably at the shoulders ([Baily et al., 2020](https://www.scitepress.org/Papers/2020/101458)). It compares
  joint positions directly, without time alignment.
- **Spatial and temporal error separated by DTW.** A two-level dynamic time warping method scores a novice
  against an expert database without sport-specific knowledge, tested on tennis serves and karate strikes
  and compared with coach assessments ([Rennes, HAL-01586401](https://hal-univ-rennes1.archives-ouvertes.fr/hal-01586401)).
  This is the closest published match to our scoring idea.
- **Other DTW assessments.** DTW has been used to score fitness movements against expert recordings using
  manually chosen joint angles, to compare weightlifting against a "golden standard" lift, and to align
  baseball pitching motions ([survey of skeleton-based assessment, arXiv 2007.10737](https://arxiv.org/pdf/2007.10737)
  gives the wider picture, including how scoring is separated from feedback).

### 2.4 Techniques we rely on

| Technique | Role | Where it comes from |
|---|---|---|
| Top-down pose estimation | Keypoints per frame | Section 2.1 |
| Joint angles normalised by body size | Features comparable across players and distances | Section 2.3 (manually chosen joint angles in DTW scoring) |
| Dynamic time warping | Align swings of different speed before comparing | Section 2.3 |
| Wrist-speed peak detection | Find swings and estimate contact | Our design; no source checked |

## 3. Gap and contribution

- Padel has position and pose work and now a professional dataset ([PadelTracker100](https://zenodo.org/records/14653706)),
  but in our searches no published system scores a padel player's technique against professionals.
- Tennis and other sports show pose-plus-DTW comparison works in principle, but it has not been tried on
  padel footage, whose camera angle and glass walls differ.
-   Our contribution is a padel-specific, explainable pipeline with a from-scratch DTW, an evaluation of
  similarity and confidence on user-selected forehand and backhand clips, and an explicit analysis of
  failure cases caused by pose quality, camera position, and limited professional-reference coverage.

We have not done an exhaustive search. A claim of novelty in the report needs a proper search first
(for example Google Scholar, IEEE Xplore and arXiv for "padel" combined with pose, technique or DTW).

## 4. Ethical assessment

| Issue | Risk | Mitigation |
|---|---|---|
| Filming people | Players filmed without informed consent | Written consent from anyone we record; explain what is stored and for how long |
| Identifiability | Faces visible in saved video | Show and save skeleton overlays; avoid storing raw video where not needed; delete raw clips after keypoints are extracted when consent allows |
| Third-party footage | Professional match video may belong to the broadcaster | Use the dataset's labels as published; confirm footage rights before showing or redistributing video (open item in [datasets.md](datasets.md)) |
| Misleading feedback | Score read as medical or objective truth; a pose error produces wrong advice | State clearly that the score measures similarity to eight professionals; show confidence; do not give injury advice |
| Bias | Reference covers eight pros from one tournament, so styles and body types are narrow | Report this limit; avoid labelling different-but-valid styles as "wrong" |
| Data provenance | Rules require public data or consent | PadelTracker100 is CC BY 4.0 for the annotations; cite it as required |

Non-biometric alternative (course rule): the system uses body keypoints only and never identifies a person.

## 5. Measurable success criteria

Proposed targets are listed in [project-plan.md](project-plan.md). The team should agree numbers before
experiments start, so results are not fitted afterwards. Suggested metrics:

- stroke detection precision and recall against the dataset's shot labels, per stroke type
- keypoint availability on our own camera
- similarity separation between held-out professional strokes and synthetic perturbations or deliberately
  flawed teammate strokes
- processing time per frame
- confidence and failure rates

## 6. References

"Checked" says how much we looked at. Nothing here was read in full.

| Source | Checked |
|---|---|
| [Padel player positions from high-angle video (Sensors 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8151013/) | Abstract and search summary |
| [PadelTracker100 dataset and paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12926558/) | Paper sections on annotations, plus the dataset files themselves |
| [Camera viewing angle and OpenPose validity](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11819822/) | Search summary |
| [OpenPose in a real athletics competition](https://eprints.leedsbeckett.ac.uk/id/eprint/10344/8/FeasibilityOfOpenposeMarkerlessMotionAnalysisInARealAthleticsCompetitionPV-WALKER.pdf) | Search summary |
| [Table tennis stroke recognition from 2D pose (CVPRW 2021)](https://openaccess.thecvf.com/content/CVPR2021W/CVSports/html/Kulkarni_Table_Tennis_Stroke_Recognition_Using_Two-Dimensional_Human_Pose_Estimation_CVPRW_2021_paper.html) | Search summary |
| [BST: badminton stroke-type transformer](https://arxiv.org/pdf/2502.21085) | Search summary |
| [Stroke comparison, professional vs amateur tennis (Baily et al. 2020)](https://www.scitepress.org/Papers/2020/101458) | Search summary |
| [Automatic evaluation of sports motion, spatial and temporal errors](https://hal-univ-rennes1.archives-ouvertes.fr/hal-01586401) | Search summary |
| [Review of skeleton-based movement assessment](https://arxiv.org/pdf/2007.10737) | Search summary |

Dropped: a table-tennis DTW study and a 2026 tennis biomechanics preprint came up in searches, but we
could not identify their sources reliably, so they are not cited.
