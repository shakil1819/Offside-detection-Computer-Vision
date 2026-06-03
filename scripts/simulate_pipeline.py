"""
End-to-end pipeline simulation.

Simulates the full automated flow:
  Frontend upload → Backend → Kaggle dataset push → Kernel push →
  Status polling → Download results → Verdict

Usage:
    python scripts/simulate_pipeline.py                  # full Kaggle flow
    python scripts/simulate_pipeline.py --local          # local mode (no Kaggle)
    python scripts/simulate_pipeline.py --dry-run        # mock all Kaggle calls
    python scripts/simulate_pipeline.py --video path.mp4 # custom video
"""

import sys
import time
import argparse
import requests
import tempfile
from pathlib import Path

# Project root
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

API_BASE = 'http://localhost:8000'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step(n: int, label: str) -> None:
    print(f'\n{"-"*55}')
    print(f'  Step {n}: {label}')
    print(f'{"-"*55}')


def ok(msg: str) -> None:
    print(f'  [OK]  {msg}')


def info(msg: str) -> None:
    print(f'  [..]  {msg}')


def fail(msg: str) -> None:
    print(f'  [!!]  {msg}')
    sys.exit(1)


def make_test_video(path: Path) -> None:
    """Create a 3-second synthetic soccer clip."""
    import cv2
    import numpy as np
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(90):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)                                    # green pitch
        cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 2)  # center line
        cv2.rectangle(frame, (190, 180), (220, 280), (0, 255, 0), -1)   # attacker
        cv2.rectangle(frame, (350, 180), (380, 280), (0, 0, 255), -1)   # defender
        cv2.circle(frame, (310 + i // 3, 240), 10, (255, 255, 255), -1) # ball
        out.write(frame)
    out.release()


def wait_for_server(timeout: int = 15) -> bool:
    for _ in range(timeout):
        try:
            r = requests.get(f'{API_BASE}/health', timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def poll_status(job_id: str, interval: float = 2.0, timeout: int = 600) -> dict:
    """Poll /status/{job_id} until completed or failed."""
    start = time.time()
    last_progress = -1

    while time.time() - start < timeout:
        r = requests.get(f'{API_BASE}/status/{job_id}', timeout=10)
        data = r.json()
        status = data.get('status')
        progress = data.get('progress', 0)

        if progress != last_progress:
            bar = '#' * (progress // 5) + '.' * (20 - progress // 5)
            print(f'\r  [{bar}] {progress:3d}%  {status:<12}', end='', flush=True)
            last_progress = progress

        if status in ('completed', 'failed'):
            print()  # newline after progress bar
            return data

        time.sleep(interval)

    return {'status': 'timeout', 'error': f'Timed out after {timeout}s'}


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Simulate full pipeline')
    parser.add_argument('--local', action='store_true', help='Force LOCAL_PROCESSING=true')
    parser.add_argument('--dry-run', action='store_true', help='Mock all Kaggle API calls')
    parser.add_argument('--video', default=None, help='Path to custom test video')
    parser.add_argument('--incident', default='offside', choices=['offside', 'goal'])
    parser.add_argument('--frame', type=int, default=None, help='Frame number (default: middle)')
    args = parser.parse_args()

    print('\n' + '=' * 55)
    print('  Athletic Intelligence -- Pipeline Simulation')
    print('=' * 55)
    mode = 'LOCAL' if args.local else ('DRY-RUN (mocked Kaggle)' if args.dry_run else 'KAGGLE')
    print(f'  Mode: {mode}')
    print(f'  Incident: {args.incident}')
    print(f'  Frame: {args.frame or "auto (middle)"}')

    # -- Step 1: Health check ------------------------------------------
    step(1, 'Backend health check')
    if not wait_for_server(timeout=5):
        fail('Backend not running. Start with: python run.py')
    r = requests.get(f'{API_BASE}/health').json()
    ok(f'Backend online  kaggle_configured={r.get("kaggle_configured")}')

    # -- Step 2: Prepare test video ------------------------------------
    step(2, 'Preparing test video')
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            fail(f'Video not found: {video_path}')
        ok(f'Using: {video_path} ({video_path.stat().st_size/1024:.1f} KB)')
    else:
        tmp = Path(tempfile.mktemp(suffix='.mp4'))
        make_test_video(tmp)
        video_path = tmp
        ok(f'Created synthetic 3s soccer clip ({video_path.stat().st_size/1024:.1f} KB)')

    # -- Step 3: Upload via /upload endpoint ---------------------------
    step(3, 'Uploading video (simulates frontend button)')
    with open(video_path, 'rb') as fh:
        files = {'file': (video_path.name, fh, 'video/mp4')}
        data = {'incident_type': args.incident}
        if args.frame is not None:
            data['frame_number'] = str(args.frame)
        # simulate=true → server runs kernel locally regardless of LOCAL_PROCESSING config
        # simulate=false → server follows LOCAL_PROCESSING config (real Kaggle if false)
        if args.dry_run:
            data['simulate'] = 'true'
            info('simulate=true sent — server will use local pipeline (no Kaggle calls)')
        r = requests.post(f'{API_BASE}/upload', files=files, data=data, timeout=60)

    if r.status_code not in (200, 202):
        fail(f'Upload failed: {r.status_code} -- {r.text}')

    upload_data = r.json()
    job_id = upload_data['job_id']
    ok(f'Job created: {job_id}')
    ok(f'Status: {upload_data["status"]}')
    info(upload_data.get('message', ''))

    # -- Step 4: Automated pipeline runs (no intervention) -------------
    step(4, 'Automated pipeline (no intervention required)')
    print()
    info('Polling /status every 2s ...')
    print()
    timeout = 60 if args.dry_run else 600
    result = poll_status(job_id, interval=2.0, timeout=timeout)

    # -- Step 5: Results -----------------------------------------------
    step(5, 'Results')
    status = result.get('status')
    if status == 'completed':
        ok(f'Verdict:    {result.get("verdict")}')
        ok(f'Confidence: {result.get("confidence", 0)*100:.0f}%')
        ok(f'Completed:  {result.get("completed_at", "—")}')

        # Download video if available
        dl = requests.get(f'{API_BASE}/download/{job_id}?format=json', timeout=10)
        if dl.status_code == 200:
            ok('Verdict JSON downloadable ✓')
        else:
            info(f'Download: {dl.status_code} (output files may need Kaggle for real run)')
    elif status == 'timeout':
        fail('Pipeline timed out (check Kaggle kernel status manually)')
    else:
        fail(f'Pipeline failed: {result.get("error", "unknown error")}')

    # -- Step 6: Job history -------------------------------------------
    step(6, 'Job history persisted')
    jobs = requests.get(f'{API_BASE}/jobs').json()
    ok(f'Total jobs in history: {jobs["count"]}')
    for j in jobs['jobs'][:3]:
        status_icon = '[OK]' if j['status'] == 'completed' else '[!!]' if j['status'] == 'failed' else '[..]'
        print(f'  {status_icon}  {j["job_id"][:12]}  {j["status"]:<12}  {j["verdict"] or "-"}')

    # -- Cleanup -------------------------------------------------------
    if not args.video and video_path.exists():
        video_path.unlink()

    print()
    print('=' * 55)
    print('  Simulation complete')
    print('=' * 55)
    print()


if __name__ == '__main__':
    main()
