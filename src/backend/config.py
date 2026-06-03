"""Backend configuration."""

import os
from pathlib import Path
from typing import Optional


class BackendConfig:
    """Backend configuration."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    BACKEND_DIR = PROJECT_ROOT / 'src' / 'backend'
    STORAGE_DIR = BACKEND_DIR / 'storage'
    DATA_DIR = BACKEND_DIR / 'data'
    UPLOADS_DIR = DATA_DIR / 'uploads'
    OUTPUTS_DIR = DATA_DIR / 'outputs'

    # Create directories
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Kaggle Configuration
    KAGGLE_USERNAME: Optional[str] = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY: Optional[str] = os.getenv('KAGGLE_KEY')
    KAGGLE_DATASET_NAME: str = os.getenv('KAGGLE_DATASET_NAME', 'athletic-intelligence-dataset')
    KAGGLE_KERNEL_SLUG: str = os.getenv('KAGGLE_KERNEL_SLUG', 'athletic-intelligence-kernel')

    # Local staging dirs for Kaggle (committed to repo, pushed to Kaggle)
    KAGGLE_DATASET_STAGING_DIR: Path = PROJECT_ROOT / 'kaggle-dataset'
    KAGGLE_KERNEL_DIR: Path = PROJECT_ROOT / 'kaggle-kernel'

    # Server Configuration
    HOST: str = os.getenv('BACKEND_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('BACKEND_PORT', '8000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'

    # Job Configuration
    JOB_TIMEOUT_SECONDS: int = 1800   # 30 min — Kaggle kernels queue + run can take 15 min
    KAGGLE_POLL_INTERVAL_SECONDS: int = 30  # Poll every 30s (kernel state changes slowly)
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: int = 2

    # Local processing mode — bypass Kaggle, run kernel directly on this machine
    # Set LOCAL_PROCESSING=true in .env for local dev/demo
    LOCAL_PROCESSING: bool = os.getenv('LOCAL_PROCESSING', 'true').lower() == 'true'

    # Video Configuration
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_VIDEO_FORMATS: tuple = ('.mp4', '.avi', '.mov', '.mkv')

    # Job State File
    JOBS_STATE_FILE: Path = STORAGE_DIR / 'jobs.json'

    # Job State
    JOB_STATUSES = {
        'PENDING': 'pending',
        'UPLOADING': 'uploading',
        'PROCESSING': 'processing',
        'COMPLETED': 'completed',
        'FAILED': 'failed',
    }


def get_backend_config() -> BackendConfig:
    """Get backend configuration instance."""
    return BackendConfig()
