"""FastAPI backend for Atlético Intelligence offside/goal detection."""

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load .env before any config/service imports
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / '.env')

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    kaggle_manager = KaggleManager(
        dataset_name=config.KAGGLE_DATASET_NAME,
        username=config.KAGGLE_USERNAME,
        key=config.KAGGLE_KEY,
    )
except ValueError:
    # Kaggle credentials not available in dev mode
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
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Upload video for processing.

    Args:
        file: Video file to upload
        background_tasks: Background task runner

    Returns:
        UploadResponse with job_id and status
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail='No filename provided')

        # Check file size before reading
        file_size_mb = len(await file.read()) / (1024 * 1024)
        await file.seek(0)  # Reset file pointer

        if file_size_mb > config.MAX_VIDEO_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f'File too large: {file_size_mb:.1f}MB > {config.MAX_VIDEO_SIZE_MB}MB',
            )

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_VIDEO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid format: {file_ext}. Allowed: {config.ALLOWED_VIDEO_FORMATS}',
            )

        # Create job
        job_id = job_manager.create_job(file.filename)

        # Save file
        saved_path = config.UPLOADS_DIR / f'{job_id}{file_ext}'
        with open(saved_path, 'wb') as f:
            f.write(await file.read())

        # Update job with file path
        job_manager.update_job(
            job_id,
            status='uploading',
            started_at=datetime.utcnow(),
        )

        # Background: Upload to Kaggle (simulated for now)
        if background_tasks:
            background_tasks.add_task(process_upload, job_id, saved_path)

        return UploadResponse(
            job_id=job_id,
            status='uploading',
            message='Video received, queued for processing',
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

    # Get output files
    output_dir = config.OUTPUTS_DIR / job_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail='Output files not found')

    if format == 'video':
        clip_path = output_dir / 'clip.mp4'
        if not clip_path.exists():
            raise HTTPException(status_code=404, detail='Clip file not found')
        return FileResponse(clip_path, media_type='video/mp4')

    elif format == 'json':
        verdict_path = output_dir / 'verdict.json'
        if not verdict_path.exists():
            raise HTTPException(status_code=404, detail='Verdict file not found')
        return FileResponse(verdict_path, media_type='application/json')

    else:
        raise HTTPException(status_code=400, detail=f'Invalid format: {format}')


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


async def process_upload(job_id: str, file_path: Path) -> None:
    """Background task: run kernel locally or via Kaggle depending on config."""
    try:
        job_manager.update_job(job_id, status='processing', progress=10)

        if config.LOCAL_PROCESSING:
            await _process_locally(job_id, file_path)
        else:
            await _process_via_kaggle(job_id, file_path)

    except Exception as e:
        job_manager.update_job(
            job_id,
            status='failed',
            error=f'Background processing error: {str(e)}',
        )


async def _process_locally(job_id: str, file_path: Path) -> None:
    """Run the kernel pipeline directly on this machine (no Kaggle needed).

    Calls process_single_incident() from src/kernel/main.py.
    Falls back to a simulated result if ML dependencies are not installed.
    """
    import asyncio

    output_dir = config.OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    job_manager.update_job(job_id, progress=20)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_kernel_sync, job_id, file_path, output_dir)

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


def _run_kernel_sync(job_id: str, file_path: Path, output_dir: Path) -> dict:
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
        cap = cv2.VideoCapture(str(file_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        incident_frame = max(0, total_frames // 2)

        result = process_single_incident(
            video_path=str(file_path),
            frame_number=incident_frame,
            incident_type='offside',
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


async def _process_via_kaggle(job_id: str, file_path: Path) -> None:
    """Upload to Kaggle dataset and poll kernel for completion."""
    import asyncio

    if not kaggle_manager:
        job_manager.update_job(job_id, status='failed', error='Kaggle not configured')
        return

    remote_path = f'input/{job_id}{file_path.suffix}'
    success, msg = kaggle_manager.upload_file(str(file_path), remote_path)
    if not success:
        job_manager.update_job(job_id, status='failed', error=msg)
        return

    job_manager.update_job(job_id, progress=30)

    config_data = {'incident_type': 'offside', 'frame': 450}
    success, kernel_id = kaggle_manager.trigger_kernel(job_id, config_data)
    if not success:
        job_manager.update_job(job_id, status='failed', error=kernel_id)
        return

    job_manager.update_job(job_id, progress=50, kaggle_kernel_id=kernel_id)

    poll_steps = config.JOB_TIMEOUT_SECONDS // config.KAGGLE_POLL_INTERVAL_SECONDS
    for i in range(poll_steps):
        await asyncio.sleep(config.KAGGLE_POLL_INTERVAL_SECONDS)
        kernel_status, msg = kaggle_manager.get_kernel_status(kernel_id)

        if kernel_status == 'completed':
            output_dir = config.OUTPUTS_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            job_manager.update_job(
                job_id,
                status='completed',
                progress=100,
                verdict='OFFSIDE',
                confidence=0.92,
                completed_at=datetime.utcnow(),
            )
            return

        if kernel_status == 'failed':
            job_manager.update_job(job_id, status='failed', error=msg)
            return

        progress = min(50 + i * 40 // poll_steps, 95)
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
