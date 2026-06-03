"""
Athletic Intelligence — Self-contained Kaggle Kernel.

Reads: /kaggle/input/{dataset-slug}/current_job.json  (job config)
       /kaggle/input/{dataset-slug}/{job_id}.mp4       (video)
Writes: /kaggle/working/{job_id}/verdict.json
        /kaggle/working/{job_id}/freeze.png
        /kaggle/working/{job_id}/clip.mp4
        /kaggle/working/{job_id}/diagram.png

All logic is self-contained — no custom imports from dataset needed.
Uses: ultralytics, opencv-python, numpy, scikit-learn, matplotlib (all pre-installed on Kaggle).
"""

import os
import sys
import subprocess
import json

# Install ultralytics (not pre-installed in all Kaggle environments)
try:
    import ultralytics
except ImportError:
    print('[SETUP] Installing ultralytics...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics'])
    print('[SETUP] ultralytics installed')

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# 1. Locate dataset
# ---------------------------------------------------------------------------
DATASET_PATHS = [
    '/kaggle/input/athletic-intelligence-dataset',
    '/kaggle/input/athletic-intelligence-soccer-var-dataset',
    '/kaggle/input/datasets/shakil19/athletic-intelligence-dataset',
]

dataset_root = None
for p in DATASET_PATHS:
    if os.path.isdir(p):
        dataset_root = Path(p)
        break

if dataset_root is None:
    # Fallback: search all input dirs
    for p in Path('/kaggle/input').iterdir():
        if p.is_dir() and (p / 'current_job.json').exists():
            dataset_root = p
            break

if dataset_root is None:
    raise RuntimeError(f'Dataset not found. Searched: {DATASET_PATHS}')

print(f'[INFO] Dataset: {dataset_root}')

# ---------------------------------------------------------------------------
# 2. Read job config
# ---------------------------------------------------------------------------
config_path = dataset_root / 'current_job.json'
if not config_path.exists():
    raise RuntimeError(f'current_job.json not found in {dataset_root}')

with open(config_path) as f:
    job = json.load(f)

JOB_ID = job['job_id']
VIDEO_FILENAME = job['video_filename']
INCIDENT_TYPE = job.get('incident_type', 'offside')
FRAME_NUMBER = job.get('frame_number')

print(f'[INFO] Job: {JOB_ID[:12]}')
print(f'[INFO] Incident: {INCIDENT_TYPE}')
print(f'[INFO] Video: {VIDEO_FILENAME}')
print(f'[INFO] Frame: {FRAME_NUMBER or "auto"}')

# ---------------------------------------------------------------------------
# 3. Locate video
# ---------------------------------------------------------------------------
video_path = dataset_root / VIDEO_FILENAME
if not video_path.exists():
    # Search by job_id prefix
    matches = [p for p in dataset_root.iterdir()
               if p.is_file() and p.stem.startswith(JOB_ID[:8])
               and p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')]
    if not matches:
        raise RuntimeError(f'Video not found: {video_path}')
    video_path = matches[0]

print(f'[INFO] Video: {video_path}')

# ---------------------------------------------------------------------------
# 4. Output directory in /kaggle/working (the ONLY persisted path)
# ---------------------------------------------------------------------------
output_dir = Path('/kaggle/working') / JOB_ID
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 5. Determine incident frame
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(str(video_path))
FPS = int(cap.get(cv2.CAP_PROP_FPS)) or 30
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

incident_frame = FRAME_NUMBER if FRAME_NUMBER is not None else max(0, TOTAL_FRAMES // 2)
print(f'[INFO] Video: {W}x{H} @ {FPS}fps  {TOTAL_FRAMES} frames')
print(f'[INFO] Incident frame: {incident_frame}')

# ---------------------------------------------------------------------------
# 6. Load YOLO model (yolo11x-pose, auto-downloads if not cached)
# ---------------------------------------------------------------------------
from ultralytics import YOLO

MODEL_NAME = 'yolo11x-pose.pt'
# Check for pre-cached model in dataset
model_candidates = [
    dataset_root / 'models' / MODEL_NAME,
    Path('/kaggle/working') / MODEL_NAME,
    Path(MODEL_NAME),
]
model_path = MODEL_NAME  # default: auto-download
for candidate in model_candidates:
    if candidate.exists():
        model_path = str(candidate)
        print(f'[INFO] Using cached model: {candidate}')
        break

model = YOLO(model_path)
print('[INFO] Model loaded')

# ---------------------------------------------------------------------------
# 7. Extract frozen frame at incident
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, incident_frame)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    raise RuntimeError(f'Cannot read frame {incident_frame} from {video_path}')

print(f'[INFO] Frozen frame extracted: {frame.shape}')

# ---------------------------------------------------------------------------
# 8. Detect players (YOLOv11x-pose, persons only, imgsz=1280)
# ---------------------------------------------------------------------------
results = model.predict(
    frame,
    conf=0.25,            # lower threshold for portrait/distant camera footage
    classes=[0],          # person only
    imgsz=1280,
    agnostic_nms=True,
    verbose=True,         # show detection count in logs
)
result = results[0]

detections = []
if result.boxes is not None:
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    kps = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        detections.append({
            'bbox': tuple(box.tolist()),
            'confidence': float(conf),
            'keypoints': kps[i].tolist() if kps is not None else None,
        })

print(f'[INFO] Detected {len(detections)} players')

# ---------------------------------------------------------------------------
# 9. Team classification via K-means on jersey torso color
# ---------------------------------------------------------------------------
def get_jersey_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = y1 + int(h * 0.30), y1 + int(h * 0.70)
    tx1, tx2 = x1 + int(w * 0.20), x2 - int(w * 0.20)
    if tx2 <= tx1 or ty2 <= ty1:
        return np.array([128.0, 128.0, 128.0])
    roi = frame[ty1:ty2, tx1:tx2]
    return np.mean(roi, axis=(0, 1)) if roi.size > 0 else np.array([128.0, 128.0, 128.0])

def get_foot_x(det):
    kps = det.get('keypoints')
    if kps and len(kps) > 16:
        lx, rx = kps[15][0], kps[16][0]
        if lx > 0 and rx > 0:
            return max(lx, rx)
    x1, _, x2, _ = det['bbox']
    return (x1 + x2) / 2.0

teams = {}  # player_index → 0 (attacking) or 1 (defending)
if len(detections) >= 2:
    colors = np.array([get_jersey_color(frame, d['bbox']) for d in detections])
    labels = KMeans(n_clusters=min(2, len(detections)), random_state=0, n_init=10).fit_predict(colors)
    centers_x = [(d['bbox'][0] + d['bbox'][2]) / 2 for d in detections]
    avg_x = [np.mean([centers_x[i] for i in range(len(detections)) if labels[i] == c]) for c in [0, 1]]
    attacking_label = 0 if avg_x[0] < avg_x[1] else 1
    teams = {i: (0 if labels[i] == attacking_label else 1) for i in range(len(detections))}

attackers = [detections[i] for i, t in teams.items() if t == 0]
defenders = [detections[i] for i, t in teams.items() if t == 1]
print(f'[INFO] Attackers: {len(attackers)}  Defenders: {len(defenders)}')

# ---------------------------------------------------------------------------
# 10. Offside / Goal analysis
# ---------------------------------------------------------------------------
verdict = 'UNCERTAIN'
confidence = 0.0
analysis = {}

if INCIDENT_TYPE.lower() == 'offside':
    if len(attackers) >= 1 and len(defenders) >= 2:
        defenders_sorted = sorted(defenders, key=get_foot_x, reverse=True)
        second_last = defenders_sorted[1]
        defender_x = get_foot_x(second_last)
        attackers_sorted = sorted(attackers, key=get_foot_x, reverse=True)
        most_advanced = attackers_sorted[0]
        attacker_x = get_foot_x(most_advanced)

        is_offside = attacker_x > defender_x
        verdict = 'OFFSIDE' if is_offside else 'ONSIDE'
        distance = abs(attacker_x - defender_x)
        confidence = min(1.0, 0.7 + (distance / 100.0) * 0.3)
        analysis = {
            'attacker_x': attacker_x, 'defender_x': defender_x,
            'distance': distance, 'is_offside': is_offside,
        }
        print(f'[INFO] Offside: attacker_x={attacker_x:.1f} defender_x={defender_x:.1f}')
    else:
        verdict = 'UNCERTAIN'
        confidence = 0.0
        print(f'[WARN] Not enough players for offside call')

elif INCIDENT_TYPE.lower() == 'goal':
    # Simple ball-in-goal heuristic (ball detection via HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255])),
        cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255])),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_x = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 500:
            (bx, _), br = cv2.minEnclosingCircle(cnt)
            p = cv2.arcLength(cnt, True)
            if p > 0 and 4 * np.pi * area / (p * p) > 0.7 and 3 < br < 30:
                ball_x = bx
                break

    goal_line_x = W * 0.95
    if ball_x is not None:
        verdict = 'GOAL' if ball_x >= goal_line_x else 'NO-GOAL'
        confidence = 0.80
        analysis = {'ball_x': float(ball_x), 'goal_line_x': float(goal_line_x)}
    else:
        verdict = 'UNCERTAIN'
        confidence = 0.0

print(f'[RESULT] {verdict} ({confidence:.0%} confidence)')

# ---------------------------------------------------------------------------
# 11. Annotate freeze frame
# ---------------------------------------------------------------------------
annotated = frame.copy()
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # green=attacker, red=defender

for i, det in enumerate(detections):
    x1, y1, x2, y2 = map(int, det['bbox'])
    team_id = teams.get(i, 0)
    color = COLORS[team_id]
    label = f"{'ATK' if team_id == 0 else 'DEF'}"
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    kps = det.get('keypoints')
    if kps:
        for kid in [15, 16]:
            if kid < len(kps) and kps[kid][0] > 0:
                kx, ky = int(kps[kid][0]), int(kps[kid][1])
                cv2.circle(annotated, (kx, ky), 5, color, -1)

if INCIDENT_TYPE.lower() == 'offside' and 'defender_x' in analysis:
    ox = int(analysis['defender_x'])
    for y in range(0, H, 30):
        cv2.line(annotated, (ox, y), (ox, min(y + 20, H)), (0, 255, 255), 3)
    cv2.putText(annotated, 'OFFSIDE LINE', (ox + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Verdict bar
bar_y = H - 70
bar_color = (0, 0, 200) if verdict in ('OFFSIDE', 'NO-GOAL') else (0, 160, 0) if verdict in ('ONSIDE', 'GOAL') else (0, 140, 200)
overlay = annotated.copy()
cv2.rectangle(overlay, (0, bar_y), (W, H), bar_color, -1)
cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
cv2.putText(annotated, verdict, (20, bar_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
cv2.putText(annotated, f'{confidence*100:.0f}% confidence', (W - 250, bar_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

freeze_path = output_dir / 'freeze.png'
cv2.imwrite(str(freeze_path), annotated)
print(f'[INFO] Freeze frame: {freeze_path}')

# ---------------------------------------------------------------------------
# 12. Extract clip (5-15s centered on incident frame)
# ---------------------------------------------------------------------------
clip_path = output_dir / 'clip.mp4'
clip_dur = 10
clip_frames = clip_dur * FPS
start_f = max(0, incident_frame - clip_frames // 2)
end_f = min(TOTAL_FRAMES - 1, start_f + clip_frames - 1)

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
out = None
for codec in ['avc1', 'H264', 'mp4v']:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(clip_path), fourcc, FPS, (W, H))
    if out.isOpened():
        break

for fi in range(start_f, end_f + 1):
    ret, fr = cap.read()
    if not ret:
        break
    if fi == incident_frame:
        fr = annotated  # use annotated frame at the incident moment
    out.write(fr)

cap.release()
out.release()
print(f'[INFO] Clip: {clip_path}')

# ---------------------------------------------------------------------------
# 13. Generate top-down diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_facecolor('#4a7c3f')
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_title(f'Atlético Intelligence — {INCIDENT_TYPE.upper()} Analysis\nVerdict: {verdict} ({confidence:.0%})', fontsize=14, fontweight='bold')
ax.set_xlabel('X position (pixels)')
ax.set_ylabel('Y position (pixels)')

for i, det in enumerate(detections):
    x1, y1, x2, y2 = det['bbox']
    cx, cy = (x1 + x2) / 2, y2
    team_id = teams.get(i, 0)
    color_str = '#00ff00' if team_id == 0 else '#ff4444'
    label = 'ATK' if team_id == 0 else 'DEF'
    ax.scatter(cx, cy, c=color_str, s=200, zorder=5, edgecolors='white', linewidths=1.5)
    ax.annotate(label, (cx, cy), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8, color='white', fontweight='bold')

if INCIDENT_TYPE.lower() == 'offside' and 'defender_x' in analysis:
    ax.axvline(x=analysis['defender_x'], color='yellow', linewidth=2.5, linestyle='--', label=f'Offside line x={analysis["defender_x"]:.0f}')
    ax.legend(loc='upper right', facecolor='#2a4a2a', labelcolor='white', fontsize=9)

ax.grid(True, alpha=0.2, color='white')
fig.patch.set_facecolor('#1a2e1a')
diagram_path = output_dir / 'diagram.png'
plt.savefig(str(diagram_path), dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'[INFO] Diagram: {diagram_path}')

# ---------------------------------------------------------------------------
# 14. Write verdict JSON
# ---------------------------------------------------------------------------
verdict_data = {
    'job_id': JOB_ID,
    'incident_type': INCIDENT_TYPE,
    'frame_number': incident_frame,
    'video_filename': VIDEO_FILENAME,
    'verdict': verdict,
    'confidence': confidence,
    'analysis': analysis,
    'players_detected': len(detections),
    'attackers': len(attackers),
    'defenders': len(defenders),
    'processed_at': datetime.utcnow().isoformat(),
    'output_files': {
        'freeze': str(freeze_path),
        'clip': str(clip_path),
        'diagram': str(diagram_path),
    },
}
verdict_path = output_dir / 'verdict.json'
with open(verdict_path, 'w') as f:
    json.dump(verdict_data, f, indent=2)

print(f'[INFO] Verdict: {verdict_path}')
print()
print('=' * 50)
print(f'  {verdict}  ({confidence:.0%} confidence)')
print(f'  Players: {len(detections)} total ({len(attackers)} ATK / {len(defenders)} DEF)')
print('=' * 50)
