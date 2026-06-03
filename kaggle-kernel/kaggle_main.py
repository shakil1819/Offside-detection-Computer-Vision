"""
Kaggle Kernel entry point for Athletic Intelligence offside/goal detection.

Execution environment:
  - Platform: Kaggle T4 GPU kernel
  - Input:    /kaggle/input/athletic-intelligence-dataset/
  - Output:   /kaggle/working/  (ONLY this dir is retrievable)
  - Source:   src/kernel/ shipped inside the dataset

Triggered by: backend calling kaggle.api.kernels_push()
Config:       input/current_job.json contains job_id, frame, incident_type
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Locate the dataset mount (path varies between Kaggle sessions)
# ---------------------------------------------------------------------------
DATASET_SEARCH_PATHS = [
    '/kaggle/input/athletic-intelligence-dataset',
    '/kaggle/input/datasets/shakil19/athletic-intelligence-dataset',
]

dataset_root = None
for candidate in DATASET_SEARCH_PATHS:
    if os.path.isdir(candidate):
        dataset_root = Path(candidate)
        break

if dataset_root is None:
    raise RuntimeError(
        "Dataset 'athletic-intelligence-dataset' not mounted. "
        f"Searched: {DATASET_SEARCH_PATHS}"
    )

print(f"[INFO] Dataset mounted at: {dataset_root}")

# ---------------------------------------------------------------------------
# 2. Add src/kernel to Python path so we can import our modules
# ---------------------------------------------------------------------------
src_path = dataset_root  # src/ is at dataset root
sys.path.insert(0, str(src_path))

# ---------------------------------------------------------------------------
# 3. Install ultralytics if not already present (Kaggle has most deps)
# ---------------------------------------------------------------------------
try:
    import ultralytics
    print(f"[INFO] ultralytics {ultralytics.__version__} already installed")
except ImportError:
    print("[INFO] Installing ultralytics...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

# ---------------------------------------------------------------------------
# 4. Read job config from dataset input folder
# ---------------------------------------------------------------------------
config_path = dataset_root / 'input' / 'current_job.json'
if not config_path.exists():
    # Fallback: find any *_config.json and use the most recent
    config_files = sorted(dataset_root.glob('input/*_config.json'))
    if not config_files:
        raise RuntimeError(f"No job config found in {dataset_root / 'input'}")
    config_path = config_files[-1]

with open(config_path) as f:
    job_config = json.load(f)

job_id = job_config['job_id']
incident_type = job_config.get('incident_type', 'offside')
frame_number = job_config.get('frame_number', None)
video_filename = job_config.get('video_filename')

print(f"[INFO] Job: {job_id}")
print(f"[INFO] Incident: {incident_type}")
print(f"[INFO] Video: {video_filename}")
print(f"[INFO] Frame: {frame_number}")

# ---------------------------------------------------------------------------
# 5. Locate the video file
# ---------------------------------------------------------------------------
video_path = dataset_root / 'input' / video_filename
if not video_path.exists():
    # Try searching by job_id prefix
    matches = list((dataset_root / 'input').glob(f'{job_id}.*'))
    video_matches = [m for m in matches if m.suffix in ('.mp4', '.avi', '.mov', '.mkv')]
    if not video_matches:
        raise RuntimeError(f"Video file not found: {video_path}")
    video_path = video_matches[0]

print(f"[INFO] Video path: {video_path}")

# ---------------------------------------------------------------------------
# 6. Set up output directory inside /kaggle/working (only this is persisted)
# ---------------------------------------------------------------------------
working_dir = Path('/kaggle/working')
output_dir = working_dir / job_id
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 7. Determine incident frame (use middle frame as fallback)
# ---------------------------------------------------------------------------
if frame_number is None:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_number = max(0, total_frames // 2)
    print(f"[INFO] No frame specified, using middle frame: {frame_number}")

# ---------------------------------------------------------------------------
# 8. Check for pre-cached YOLO model in dataset
# ---------------------------------------------------------------------------
models_dir = dataset_root / 'models'
yolo_model_path = models_dir / 'yolo11x-pose.pt'
if yolo_model_path.exists():
    # Copy to working dir so YOLO can find it by name
    import shutil
    local_model = working_dir / 'yolo11x-pose.pt'
    if not local_model.exists():
        shutil.copy2(yolo_model_path, local_model)
    os.chdir(str(working_dir))
    print(f"[INFO] Using pre-cached model: {yolo_model_path}")
else:
    # Will auto-download from Ultralytics (requires enable_internet: true)
    print("[INFO] Model not cached, will download from Ultralytics")

# ---------------------------------------------------------------------------
# 9. Run the processing pipeline
# ---------------------------------------------------------------------------
from src.kernel.main import process_single_incident
from src.kernel.config import get_config
import logging

kernel_config = get_config()
kernel_config.ENABLE_VISUALIZATION = True

logger = logging.getLogger('kaggle-kernel')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

print(f"\n[INFO] Starting processing pipeline...")
result = process_single_incident(
    video_path=str(video_path),
    frame_number=frame_number,
    incident_type=incident_type,
    output_dir=output_dir,
    config=kernel_config,
    logger=logger,
)

# ---------------------------------------------------------------------------
# 10. Write verdict JSON to /kaggle/working/{job_id}/verdict.json
# ---------------------------------------------------------------------------
verdict_data = {
    'job_id': job_id,
    'incident_type': incident_type,
    'frame_number': frame_number,
    'video_filename': video_filename,
    'verdict': result.get('verdict'),
    'confidence': result.get('confidence'),
    'status': result.get('status'),
    'error': result.get('error'),
    'output_files': {
        'clip': result.get('clip_path'),
        'diagram': result.get('diagram_path'),
        'freeze': result.get('freeze_path'),
    },
}

verdict_path = output_dir / 'verdict.json'
with open(verdict_path, 'w') as f:
    json.dump(verdict_data, f, indent=2)

# ---------------------------------------------------------------------------
# 11. Print summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"[RESULT] Job: {job_id}")
print(f"[RESULT] Verdict: {result.get('verdict', 'UNKNOWN')}")
print(f"[RESULT] Confidence: {result.get('confidence', 0.0):.2%}")
print(f"[RESULT] Status: {result.get('status', 'unknown')}")
if result.get('error'):
    print(f"[ERROR] {result['error']}")
print(f"[RESULT] Output: {output_dir}")
print(f"{'='*50}\n")
