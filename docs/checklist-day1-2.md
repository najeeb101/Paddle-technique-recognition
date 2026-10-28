# Checklist: days 1 and 2

Companion to [project-plan.md](project-plan.md), week 1. Tick items as they are done and write the
result next to each one. Anything that surprises you goes in the "Findings" box at the bottom, not in
someone's head.

Rule for both days: if a number does not match the expected value, **stop and find out why** before
building on it.

## Before anyone starts (team lead, 30 minutes)

- [ ] Everyone has the repo cloned and Python 3.10 or newer working.
- [ ] Dataset holder: Najeeb. The zip is 7.1 GB, so Najeeb extracts it and shares only what others need
  (see A1). Current copy: `C:\Users\ACER\datasets\padeltracker100\padel-data-labels.zip`.
- [ ] Owners are named in the table in `project-plan.md` section 4.
- [ ] Instructor message sent (Python, AI use, pose model, camera requirement). Reply not needed to start
  day 1, but chase it before the end of week 1.
- [ ] Create three scratch files for results: `docs/measurements.md`, `docs/failures.md`,
  `docs/results.md` (a title line is enough for now).

## Day 1

### A. Data preparation: load and verify the dataset

- [ ] **A1. Extract the zip into its own empty folder** outside the repo (for example
  `C:\Users\<you>\datasets\padeltracker100\extracted`). Keep the zip. Do not extract inside the repo or
  next to your code.
  - Expected contents: `2022_BCN_FinalF_1.mp4`, `2022_BCN_FinalM_1.mp4`, `2022_BCN_FinalF_1_sample.mp4`,
    and a `labels/` folder with `*_pose.json`, `*_shots.csv`, `*_ball.json`.
  - Disk needed: about 7.2 GB.
  - If you re-download, check the MD5 equals `8711f5d584f602fba1dcb745c2ba70a6`.
- [ ] **A2. Load the shot CSVs.** They are semicolon-delimited with columns `file_name;has_shot;category`.
  Expected: 20,100 rows (FinalF) and 20,035 rows (FinalM); frame numbers 0 to 20,099.
- [ ] **A3. Load the pose JSONs** one at a time. They are 110 to 130 MB each, so check free memory first
  and close other programs. Expected: 45,934 images and 183,744 annotations (FinalF); 53,888 images and
  215,173 annotations (FinalM). Each image should have four people.
- [ ] **A4. Count labelled strokes.** Group consecutive frames with the same category into one window.
  Expected windows per category, women's plus men's: Forehand 293, Backhand 282, Smash 216, Serve 73,
  Other 68, Dropshot 5.
- [ ] **A5. Print the window lengths** for the supported categories. Expected medians (frames),
  FinalF then FinalM: forehand 14 and 10, backhand 13 and 10. Record smash and serve only as dataset
  facts; they are outside the MVP.
- [ ] **A6. Record any mismatch** in the Findings box.

Done when: all counts match, or every difference is explained.

### B. Pose and tracking: get a pose model running

- [ ] **B1. Pick a candidate pose model** from what the instructor permits (or, until the answer arrives,
  one candidate to test: a COCO-17 top-down model). Note why.
- [ ] **B2. Install it** in a clean virtual environment. Record the version and install time.
- [ ] **B3. Run it on `2022_BCN_FinalF_1_sample.mp4`** (30 MB). Save the keypoints as a `(T, 17, 3)` array.
- [ ] **B4. Time it.** Write in `docs/measurements.md`: frames processed, seconds per frame, hardware used
  (CPU or GPU, model name), and whether it ran on all four players or one.
- [ ] **B5. Draw the skeleton on 10 frames** and look at them. Note any frame where an arm or wrist is
  clearly wrong.

Done when: a skeleton overlay exists for the sample clip and the speed is written down.

### C. Features and scoring: agree the contracts

- [ ] **C1. Read `design.md` section 3** (file formats) and section 4 (features) with D.
- [ ] **C2. Fix any disagreement now**: shape of the keypoint array, joint order, units, how missing keypoints
  are marked. Edit the document, do not agree verbally.
- [ ] **C3. Create the repo skeleton** from `design.md` section 9: empty modules with docstrings, a `tests/`
  folder, and a `requirements.txt`.
- [ ] **C4. Make sure the test runner works** (one trivial passing test).

Done when: `pytest` runs from the repo root.

### D. Web app: agree the contracts and open the door

- [ ] **D1. Read `design.md` sections 3.4, 3.5 and 8** with C.
- [ ] **D2. Set up the server skeleton** with the four endpoints from the API table, returning fixed example
  data.
- [ ] **D3. Find out how a phone reaches the laptop**: check both are on the same Wi-Fi, find the laptop's
  address, and open the server's page from the phone browser. Note anything that blocked you (firewall,
  guest network).
- [ ] **D4. Try the file input** `accept="video/*" capture`: open it on the phone, record two seconds,
  upload it. Check the file arrives and how large it is.

Done when: the phone shows the page and an uploaded clip arrives on the laptop.

### E. Evaluation, recording and report: protocol and consent

- [ ] **E1. Read the filming protocol** (`project-plan.md` section 9) and change anything that is unrealistic
  for your court and equipment.
- [ ] **E2. Find a court and a time** for filming in week 2 or early week 3. Confirm tripod height options.
- [ ] **E3. Write the consent form**: who is filmed, what is stored (raw video, keypoints, overlays), how long,
  who sees it, how to withdraw. One page.
- [ ] **E4. Make the clip log**: a `clips.csv` with columns player, handedness, stroke, fault type (or none),
  camera height, camera distance, date, file name.
- [ ] **E5. Start a reading list** for the report from `literature-review.md`, assigning each source to one
  member who reads it in full.

Done when: filming is scheduled and the consent form exists.

### End of day 1 (all, 15 minutes)

- [ ] Each owner states what they finished and what blocked them.
- [ ] Any blocker lasting more than a day is raised with the instructor or reassigned.

## Day 2

### A. Data preparation: resolve the keypoint order, group strokes

- [ ] **A7. Settle the left/right question for the men's file.** In FinalM's `categories`, left and right
  names are swapped compared with FinalF. For each match take 10 video frames at known indices:
  - Draw each person's keypoints with labels using the file's own names.
  - Look at the frame: is the keypoint labelled "right wrist" on the player's right hand?
  - Note that players on the far side face the camera, so their left is on the image's right.
  - Write the verdict in `datasets.md` with the frames you checked, and make the loader convert to the
    standard order in `design.md` section 3.1.
- [ ] **A8. Write the stroke-window grouping** and save `strokes_index.csv` with the columns in
  `design.md` section 3.2 that can be filled now (`stroke_id`, `match`, `type`, `start_frame`,
  `end_frame`). The others come on day 3.
- [ ] **A9. Check the file**: the row counts per type equal the counts from A4.
- [ ] **A10. Check the edge cases**: windows of length 1, windows of length near 48, and two adjacent windows
  of the same type that may have been merged. Decide the rule and write it down.

Done when: the left/right verdict is written and `strokes_index.csv` matches expectations.

### B. Pose and tracking: pro-player tracker

- [ ] **B6. Write the tracker** for the dataset pose files: for each frame, match its four people to the
  previous frame's by smallest distance between hip centres.
- [ ] **B7. Test on 1,000 consecutive frames** from each match: draw track IDs on 20 frames, spread out, and
  check the identity stays on the same person, especially when players cross near the net.
- [ ] **B8. Record how often identities swap** in the sample and any pattern (for example near the net).
- [ ] **B9. Save the tracked keypoints** for the labelled frames only (the first about 20,100 of each match),
  not for the whole match, to keep memory and file size small.

Done when: IDs are stable on the checked sample and a swap rate is written down.

### C. Features: test them on made-up skeletons

- [ ] **C5. Write the features** from `design.md` section 4 (elbow angle, shoulder angle, knee angles,
  shoulder and hip rotation, separation, torso lean, wrist height, wrist speed).
- [ ] **C6. Write unit tests with known answers:**
  - a straight arm gives an elbow angle of 180 degrees
  - a right-angle elbow gives 90 degrees
  - a horizontal shoulder line gives a rotation of 0 degrees
  - scaling a skeleton by 2 leaves every feature except wrist speed unchanged (wrist speed is divided by
    torso length, so check it too)
  - a missing keypoint produces a missing feature, not an error
- [ ] **C7. Check the sign conventions** with D and E: what direction is a positive rotation, and does
  mirroring a left-hander flip them correctly? Write the convention in `design.md`.

Done when: all feature tests pass and sign conventions are written down.

### D. Web app: placeholder to real status

- [ ] **D5. Make the job flow real** with fake work: upload returns a job ID, `/status` goes through stage
  names over a few seconds, `/result` returns the example JSON.
- [ ] **D6. Make the page show the stages** and the result on a phone.
- [ ] **D7. Try the error paths** with a wrong file type and an empty upload; the page should show a readable
  message.
- [ ] **D8. Note the upload size and time** for a 10-second phone clip on your Wi-Fi in `docs/measurements.md`.

Done when: a phone can upload a clip, watch progress and see the placeholder result.

### E. Evaluation and report: prepare the checks

- [ ] **E6. Write the evaluation protocol draft** (`project-plan.md` section 8): confirm each metric is
  measurable with the data you will actually have.
- [ ] **E7. Write the hand-check sheets** for days 3 and 4: a table with columns `stroke_id`, frame, what to
  look for, verdict. Fill the stroke IDs from `strokes_index.csv` once A8 is done: 40 for the hitter check
  and 30 for the contact check.
- [ ] **E8. Test the consent and clip log** by filling them in for one practice recording of a teammate.
- [ ] **E9. Start the report outline** from `project-plan.md` section 12 as a document, with each heading
  assigned to a member.

Done when: the sheets exist and the outline has owners.

### End of day 2 (all, 30 minutes)

- [ ] Each owner demonstrates their result to their reviewer, who must be able to say what it does and
  why without reading the code line by line. This is the same skill as the oral exam.
- [ ] Update the Findings box and the open decisions in `project-plan.md` section 13.
- [ ] Read the day 3 and 4 tasks and check nothing there is blocked by something unfinished.

## Findings (fill in as you go)

| Item | What we found | Who | Date |
|---|---|---|---|
| Count mismatches (A6) | | | |
| Men's keypoint order verdict (A7) | | | |
| Pose model speed (B4) | | | |
| Phone to laptop connection issues (D3) | | | |
| Tracker identity swaps (B8) | | | |
| Window edge-case rule (A10) | | | |
| Anything unexpected | | | |

## If you are stuck

| Problem | First thing to try |
|---|---|
| Out of memory loading a pose JSON | Close other programs; load one file at a time; drop fields you do not need as soon as they are read |
| Pose model too slow | Run on every second frame for the check; try a smaller model; note the speed anyway |
| Phone cannot reach the laptop | Same network? Firewall allowing the port? Try the laptop's mobile hotspot |
| Counts do not match | Re-check the delimiter and header handling before suspecting the data |
| Identity swaps everywhere | Check you are matching against the previous frame's tracks, not recomputing from scratch |
