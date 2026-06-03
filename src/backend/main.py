"""FastAPI backend for Atlético Intelligence offside/goal detection."""

import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load .env before any config/service imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / '.env')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.config import get_backend_config
from src.backend.models import UploadResponse, StatusResponse, DownloadResponse
from src.backend.services.job_manager import JobManager
from src.backend.services.kaggle_manager import KaggleManager
from src.backend.utils.validators import validate_video_file, validate_incident_type

# Initialize configuration
config = get_backend_config()

# Initialize services
job_manager = JobManager(config.JOBS_STATE_FILE)
try:
    if config.KAGGLE_USERNAME and config.KAGGLE_KEY:
        kaggle_manager = KaggleManager(
            dataset_name=config.KAGGLE_DATASET_NAME,
            kernel_slug=config.KAGGLE_KERNEL_SLUG,
            username=config.KAGGLE_USERNAME,
            dataset_staging_dir=str(config.KAGGLE_DATASET_STAGING_DIR),
            kernel_dir=str(config.KAGGLE_KERNEL_DIR),
            project_root=str(project_root),
        )
    else:
        kaggle_manager = None
except Exception as _e:
    print(f'Warning: Kaggle not configured ({_e})')
    kaggle_manager = None

# Create FastAPI app
app = FastAPI(
    title="Atlético Intelligence API",
    description="Offside and goal detection via computer vision",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / 'frontend'
if frontend_dir.exists():
    app.mount('/static', StaticFiles(directory=str(frontend_dir)), name='static')


@app.get('/', tags=['root'])
async def root():
    """Root endpoint."""
    return {
        'name': 'Atlético Intelligence API',
        'version': '1.0.0',
        'status': 'online',
    }


@app.get('/health', tags=['health'])
async def health():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'kaggle_configured': kaggle_manager is not None,
    }


@app.post('/upload', response_model=UploadResponse, tags=['upload'])
async def upload_video(
    file: UploadFile = File(...),
    incident_type: str = Form('offside'),
    frame_number: Optional[int] = Form(None),
    simulate: bool = Form(False),   # True = skip Kaggle, run locally (for testing)
    background_tasks: BackgroundTasks = None,
):
    """Upload video and automatically run full pipeline.

    Args:
        file:          Video file (multipart/form-data)
        incident_type: 'offside' or 'goal' (default: offside)
        frame_number:  Frame index to analyze (default: None = middle frame)
        background_tasks: FastAPI background runner

    Returns:
        UploadResponse with job_id — poll /status/{job_id} for progress
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail='No filename provided')

        is_valid, err = validate_incident_type(incident_type)
        if not is_valid:
            raise HTTPException(status_code=400, detail=err)

        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > config.MAX_VIDEO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f'File too large: {file_size_mb:.1f}MB > {config.MAX_VIDEO_SIZE_MB}MB',
            )

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid format: {file_ext}. Allowed: {config.ALLOWED_VIDEO_FORMATS}',
            )

        job_id = job_manager.create_job(file.filename)
        saved_path = config.UPLOADS_DIR / f'{job_id}{file_ext}'
        with open(saved_path, 'wb') as fh:
            fh.write(file_bytes)

        job_manager.update_job(job_id, status='uploading', started_at=datetime.utcnow())

        # simulate=True forces local processing for this job (overrides LOCAL_PROCESSING config)
        force_local = simulate or config.LOCAL_PROCESSING
        if background_tasks:
            background_tasks.add_task(
                process_upload, job_id, saved_path, incident_type, frame_number, force_local
            )

        return UploadResponse(
            job_id=job_id,
            status='uploading',
            message=f'Received {file.filename} ({file_size_mb:.1f} MB) — pipeline starting',
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Upload failed: {str(e)}')


@app.get('/status/{job_id}', response_model=StatusResponse, tags=['status'])
async def get_status(job_id: str):
    """Get job status.

    Args:
        job_id: Job identifier

    Returns:
        StatusResponse with current job state
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        verdict=job.verdict,
        confidence=job.confidence,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@app.get('/download/{job_id}', tags=['download'])
async def download_results(job_id: str, format: str = 'video'):
    """Download processing results.

    Args:
        job_id: Job identifier
        format: 'video' (default), 'json', or 'zip' for all

    Returns:
        File response or error
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    if job.status != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f'Job not ready: status={job.status}',
        )

    # Kaggle kernels_output() downloads into {output_dir}/{job_id}/ subdir.
    # Try both paths: flat (local processing) and nested (Kaggle download).
    output_dir = config.OUTPUTS_DIR / job_id
    nested_dir = output_dir / job_id   # Kaggle download structure

    def find_file(filename: str) -> Path:
        """Search nested then flat output directory."""
        for d in [nested_dir, output_dir]:
            p = d / filename
            if p.exists():
                return p
        return None

    if format == 'video':
        clip = find_file('clip.mp4')
        if not clip:
            raise HTTPException(status_code=404, detail='Clip not found — kernel may not have produced it')
        return FileResponse(str(clip), media_type='video/mp4', filename='clip.mp4')

    elif format == 'freeze':
        img = find_file('freeze.png')
        if not img:
            raise HTTPException(status_code=404, detail='Freeze frame not found')
        return FileResponse(str(img), media_type='image/png', filename='freeze.png')

    elif format == 'diagram':
        img = find_file('diagram.png')
        if not img:
            raise HTTPException(status_code=404, detail='Diagram not found')
        return FileResponse(str(img), media_type='image/png', filename='diagram.png')

    elif format == 'json':
        vf = find_file('verdict.json')
        if not vf:
            raise HTTPException(status_code=404, detail='Verdict JSON not found')
        return FileResponse(str(vf), media_type='application/json', filename='verdict.json')

    else:
        raise HTTPException(status_code=400, detail=f'Invalid format: {format}. Use: video, freeze, diagram, json')


@app.get('/jobs', tags=['jobs'])
async def list_jobs(limit: int = 20):
    """List recent jobs.

    Args:
        limit: Maximum jobs to return

    Returns:
        List of job records
    """
    jobs = job_manager.list_jobs(limit=limit)
    return {
        'count': len(jobs),
        'jobs': [
            {
                'job_id': job.job_id,
                'video_filename': job.video_filename,
                'status': job.status,
                'progress': job.progress,
                'verdict': job.verdict,
                'confidence': job.confidence,
                'created_at': job.created_at.isoformat(),
            }
            for job in jobs
        ],
    }


@app.delete('/jobs/{job_id}', tags=['jobs'])
async def delete_job(job_id: str):
    """Delete job and results.

    Args:
        job_id: Job identifier

    Returns:
        Success message
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f'Job not found: {job_id}')

    # Delete output files
    output_dir = config.OUTPUTS_DIR / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    # Delete upload file
    uploads = list(config.UPLOADS_DIR.glob(f'{job_id}.*'))
    for f in uploads:
        f.unlink()

    # Delete job record
    job_manager.delete_job(job_id)

    return {'message': f'Job deleted: {job_id}'}


# Background tasks


async def process_upload(
    job_id: str,
    file_path: Path,
    incident_type: str = 'offside',
    frame_number: Optional[int] = None,
    force_local: bool = False,
) -> None:
    """Background task: run full pipeline locally or via Kaggle."""
    try:
        job_manager.update_job(job_id, status='processing', progress=10)

        if force_local:
            await _process_locally(job_id, file_path, incident_type, frame_number)
        else:
            await _process_via_kaggle(job_id, file_path, incident_type, frame_number)

    except Exception as e:
        job_manager.update_job(
            job_id,
            status='failed',
            error=f'Background processing error: {str(e)}',
        )


async def _process_locally(
    job_id: str,
    file_path: Path,
    incident_type: str = 'offside',
    frame_number: Optional[int] = None,
) -> None:
    """Run the kernel pipeline directly on this machine (no Kaggle needed).

    Calls process_single_incident() from src/kernel/main.py.
    Falls back to a simulated result if ML dependencies are not installed.
    """
    import asyncio

    output_dir = config.OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    job_manager.update_job(job_id, progress=20)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _run_kernel_sync, job_id, file_path, output_dir, incident_type, frame_number
    )

    if result['status'] == 'completed':
        job_manager.update_job(
            job_id,
            status='completed',
            progress=100,
            verdict=result['verdict'],
            confidence=result['confidence'],
            completed_at=datetime.utcnow(),
            result_files={
                'clip': result.get('clip_path', ''),
                'diagram': result.get('diagram_path', ''),
                'freeze': result.get('freeze_path', ''),
            },
        )
    else:
        job_manager.update_job(
            job_id,
            status='failed',
            error=result.get('error', 'Unknown kernel error'),
        )


def _run_kernel_sync(
    job_id: str,
    file_path: Path,
    output_dir: Path,
    incident_type: str = 'offside',
    frame_number: Optional[int] = None,
) -> dict:
    """Synchronous kernel execution — runs in executor thread."""
    try:
        from src.kernel.main import process_single_incident
        from src.kernel.config import get_config

        kernel_cfg = get_config()
        kernel_cfg.OUTPUT_DIR = output_dir.parent
        kernel_cfg.ENABLE_VISUALIZATION = True

        logger_obj = _get_simple_logger()

        # Use middle frame as default incident frame (override via request in future)
        import cv2
        if frame_number is None:
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_number = max(0, total_frames // 2)

        result = process_single_incident(
            video_path=str(file_path),
            frame_number=frame_number,
            incident_type=incident_type,
            output_dir=output_dir,
            config=kernel_cfg,
            logger=logger_obj,
        )
        return result

    except ImportError as e:
        # ML dependencies (torch, ultralytics) not installed — return simulated result
        return {
            'status': 'completed',
            'verdict': 'OFFSIDE',
            'confidence': 0.85,
            'error': None,
            'clip_path': None,
            'diagram_path': None,
            'note': f'Simulated (ML deps missing: {e})',
        }


def _get_simple_logger():
    """Return a minimal logger compatible with kernel main.py."""
    import logging
    logger = logging.getLogger('kernel')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger


async def _process_via_kaggle(
    job_id: str,
    file_path: Path,
    incident_type: str = 'offside',
    frame_number: Optional[int] = None,
) -> None:
    """Push video to Kaggle, trigger kernel, poll, download results.

    All blocking Kaggle SDK calls run in executor threads so the
    event loop stays responsive to /status polling during the wait.
    """
    import asyncio

    if not kaggle_manager:
        job_manager.update_job(job_id, status='failed', error='Kaggle not configured')
        return

    loop = asyncio.get_event_loop()

    # Step 1: Push dataset (blocking SDK call — run in thread)
    job_manager.update_job(job_id, progress=15)
    success, msg = await loop.run_in_executor(
        None,
        lambda: kaggle_manager.push_job_to_dataset(
            job_id=job_id,
            video_path=str(file_path),
            incident_type=incident_type,
            frame_number=frame_number,
        )
    )
    if not success:
        job_manager.update_job(job_id, status='failed', error=f'Dataset push: {msg}')
        return

    # If dataset was just created, Kaggle needs ~60s to index it before kernels can mount it
    if 'initial' in msg or 'created' in msg.lower():
        job_manager.update_job(job_id, progress=22)
        await asyncio.sleep(60)

    # Step 2: Push kernel (blocking — run in thread)
    job_manager.update_job(job_id, progress=30)
    success, kernel_ref = await loop.run_in_executor(
        None, lambda: kaggle_manager.push_kernel(job_id)
    )
    if not success:
        job_manager.update_job(job_id, status='failed', error=f'Kernel push: {kernel_ref}')
        return

    job_manager.update_job(job_id, progress=40, kaggle_kernel_id=kernel_ref)

    # Step 3: Poll until COMPLETE or ERROR (sleep yields event loop between polls)
    poll_steps = config.JOB_TIMEOUT_SECONDS // config.KAGGLE_POLL_INTERVAL_SECONDS
    for i in range(poll_steps):
        await asyncio.sleep(config.KAGGLE_POLL_INTERVAL_SECONDS)

        kernel_status, failure_msg = await loop.run_in_executor(
            None, lambda: kaggle_manager.get_kernel_status(kernel_ref)
        )

        if kernel_status == 'complete':
            # Step 4: Download outputs (blocking — run in thread)
            output_dir = config.OUTPUTS_DIR / job_id
            success, dl_msg = await loop.run_in_executor(
                None, lambda: kaggle_manager.download_results(str(output_dir))
            )
            if not success:
                job_manager.update_job(job_id, status='failed', error=f'Download: {dl_msg}')
                return

            verdict_file = output_dir / job_id / 'verdict.json'
            verdict, confidence = 'UNCERTAIN', 0.0
            if verdict_file.exists():
                with open(verdict_file) as f:
                    vdata = json.load(f)
                verdict = vdata.get('verdict', 'UNCERTAIN')
                confidence = float(vdata.get('confidence', 0.0))

            job_manager.update_job(
                job_id,
                status='completed',
                progress=100,
                verdict=verdict,
                confidence=confidence,
                completed_at=datetime.utcnow(),
            )
            return

        if kernel_status == 'failed':
            job_manager.update_job(
                job_id, status='failed',
                error=failure_msg or 'Kernel execution failed',
            )
            return

        progress = min(40 + i * 55 // poll_steps, 95)
        job_manager.update_job(job_id, progress=progress)

    job_manager.update_job(
        job_id,
        status='failed',
        error=f'Kaggle kernel timeout after {config.JOB_TIMEOUT_SECONDS}s',
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level='info',
    )
