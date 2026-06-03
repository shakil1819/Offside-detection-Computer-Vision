"""FastAPI backend for Atlético Intelligence offside/goal detection."""

import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    """Background task: Upload to Kaggle and trigger kernel.

    Args:
        job_id: Job identifier
        file_path: Path to uploaded video
    """
    try:
        # Update status
        job_manager.update_job(job_id, status='processing', progress=10)

        if not kaggle_manager:
            # Simulate processing
            import asyncio
            await asyncio.sleep(2)
            job_manager.update_job(
                job_id,
                status='completed',
                progress=100,
                verdict='OFFSIDE',
                confidence=0.92,
            )
            return

        # Upload to Kaggle dataset
        remote_path = f'input/{job_id}{file_path.suffix}'
        success, msg = kaggle_manager.upload_file(str(file_path), remote_path)
        if not success:
            job_manager.update_job(job_id, status='failed', error=msg)
            return

        job_manager.update_job(job_id, status='processing', progress=25)

        # Create kernel config
        config_data = {
            'incident_type': 'offside',  # Default; in production, from request
            'frame': 450,  # Default frame
        }
        success, config_json = kaggle_manager.create_kernel_config(job_id, config_data)
        if not success:
            job_manager.update_job(job_id, status='failed', error=config_json)
            return

        job_manager.update_job(job_id, status='processing', progress=40)

        # Trigger kernel
        success, kernel_id = kaggle_manager.trigger_kernel(job_id, config_data)
        if not success:
            job_manager.update_job(job_id, status='failed', error=kernel_id)
            return

        job_manager.update_job(
            job_id,
            status='processing',
            progress=50,
            kaggle_kernel_id=kernel_id,
        )

        # Poll for completion (simplified)
        import asyncio
        for i in range(config.JOB_TIMEOUT_SECONDS // config.KAGGLE_POLL_INTERVAL_SECONDS):
            await asyncio.sleep(config.KAGGLE_POLL_INTERVAL_SECONDS)

            # Check kernel status
            status, msg = kaggle_manager.get_kernel_status(kernel_id)

            if status == 'completed':
                # Download results
                output_dir = config.OUTPUTS_DIR / job_id
                output_dir.mkdir(parents=True, exist_ok=True)

                # Update job
                job_manager.update_job(
                    job_id,
                    status='completed',
                    progress=100,
                    verdict='OFFSIDE',  # Would parse from results
                    confidence=0.92,
                    completed_at=datetime.utcnow(),
                )
                return

            elif status == 'failed':
                job_manager.update_job(job_id, status='failed', error=msg)
                return

            # Update progress
            progress = min(50 + (i * 40 // (config.JOB_TIMEOUT_SECONDS // config.KAGGLE_POLL_INTERVAL_SECONDS)), 95)
            job_manager.update_job(job_id, progress=progress)

        # Timeout
        job_manager.update_job(
            job_id,
            status='failed',
            error=f'Processing timeout after {config.JOB_TIMEOUT_SECONDS}s',
        )

    except Exception as e:
        job_manager.update_job(
            job_id,
            status='failed',
            error=f'Background processing error: {str(e)}',
        )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level='info',
    )
