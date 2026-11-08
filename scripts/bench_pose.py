"""Compare YOLO11-pose and MediaPipe Pose Landmarker on a clip: speed, people found, keypoint confidence."""
import sys, time
from pathlib import Path
import cv2, numpy as np

clip, models, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60
outdir.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(clip))
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
print("frames", len(frames))

# --- YOLO11-pose (COCO-17, multi-person)
from ultralytics import YOLO
yolo = YOLO(str(models / "yolo11m-pose.pt"))
yolo(frames[0], verbose=False)  # warmup
t = time.time(); ypeople = []; yconf = []
for i, f in enumerate(frames):
    r = yolo(f, verbose=False)[0]
    k = r.keypoints
    n = 0 if k is None or k.conf is None else len(k.conf)
    ypeople.append(n)
    if n: yconf.append(float(k.conf.mean()))
    if i == 20:
        cv2.imwrite(str(outdir / "yolo_frame20.jpg"), r.plot())
yt = (time.time() - t) / len(frames)

# --- MediaPipe Pose Landmarker (33 landmarks, num_poses=4)
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
opts = vision.PoseLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=str(models / "pose_landmarker_heavy.task")),
    running_mode=vision.RunningMode.VIDEO, num_poses=4)
mpl = vision.PoseLandmarker.create_from_options(opts)
t = time.time(); mpeople = []; mvis = []
for i, f in enumerate(frames):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    res = mpl.detect_for_video(img, int(i * 1000 / 30))
    mpeople.append(len(res.pose_landmarks))
    for p in res.pose_landmarks: mvis.append(np.mean([l.visibility for l in p]))
    if i == 20:
        g = f.copy(); h, w = g.shape[:2]
        for p in res.pose_landmarks:
            for l in p: cv2.circle(g, (int(l.x * w), int(l.y * h)), 3, (0, 255, 0), -1)
        cv2.imwrite(str(outdir / "mediapipe_frame20.jpg"), g)
mt = (time.time() - t) / len(frames)

print(f"YOLO11m-pose : {yt:.3f} s/frame ({1/yt:.1f} fps) people/frame mean {np.mean(ypeople):.2f} min {min(ypeople)} max {max(ypeople)} mean kp conf {np.mean(yconf):.2f}")
print(f"MediaPipe hvy: {mt:.3f} s/frame ({1/mt:.1f} fps) people/frame mean {np.mean(mpeople):.2f} min {min(mpeople)} max {max(mpeople)} mean visibility {np.mean(mvis) if mvis else 0:.2f}")
