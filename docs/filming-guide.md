# Filming guide (on-court checklist)

One page to take to the court. The full protocol, with the reasons, is in [project-plan.md](project-plan.md) section 9. The points marked *new* come from the first test of the app on real padel video (2026-11-21, [results.md](results.md) section 6c).

## Before you start

- [ ] Every filmed person has signed the consent form.
- [ ] Phone on a tripod, landscape, 1080p, 30 fps (60 if possible). Lock focus and exposure.
- [ ] Tripod behind the baseline on the centre line, as high as it goes. Do not move it during the session.
- [ ] The player stands in the near half, seen from behind. Check the whole body, racket and about 1 m of margin stay in frame through the swing.
- [ ] The player fills a good part of the frame (*new*: on TV footage players are about 140 px tall and the skeleton is noisy; closer is better, as long as the whole swing stays in frame).

## Each clip

- [ ] One stroke per clip, 3 to 6 seconds: start about 1 s before the swing, stop about 1 s after.
- [ ] Only the filmed player near the camera (*new*: the app follows the first player it sees, but a second person close by can still confuse it).
- [ ] Write a row in `clips.csv` straight away (template below).

## Per player

- 10 forehands and 10 backhands in good form.
- 5 of each with one named fault, said out loud before the swing: elbow too low, no shoulder turn, contact too late, no knee bend, arm too straight.

## clips.csv

Keep it next to the videos (not in git: raw video and this log stay private).

```
clip,date,player,hand,stroke,form,fault,camera_height_m,camera_distance_m,notes
p1_fh_001.mp4,2026-11-24,P1,right,forehand,good,,1.6,6,
p1_fh_011.mp4,2026-11-24,P1,right,forehand,fault,no_shoulder_turn,1.6,6,said fault before swing
```

- `hand`: right or left. `stroke`: forehand or backhand. `form`: good or fault.
- `fault`: elbow_low, no_shoulder_turn, late_contact, no_knee_bend or straight_arm (empty when good).

## After the session

- Copy the videos to the team's private drive; do not commit them.
- Run each clip through the app (`py -m padel.server`, or `analyze_clip`) and note any "no swing found" results: the swing detector is still too strict for fast swings, and these clips are what we tune it on.
