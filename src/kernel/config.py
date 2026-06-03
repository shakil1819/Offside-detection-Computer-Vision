"""
Configuration for Atlético Intelligence Kaggle Kernel.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for the system."""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    KAGGLE_INPUT_DIR = os.getenv('KAGGLE_INPUT_DIR', '/kaggle/input/athletic-intelligence-dataset')
    KAGGLE_WORKING_DIR = os.getenv('KAGGLE_WORKING_DIR', '/kaggle/working')

    # Local development paths
    LOCAL_INPUT_DIR = PROJECT_ROOT / 'input'
    LOCAL_OUTPUT_DIR = PROJECT_ROOT / 'output'

    # Determine environment
    IS_KAGGLE = os.path.exists(KAGGLE_INPUT_DIR) and os.path.isdir(KAGGLE_INPUT_DIR)

    # Set actual input/output dirs based on environment
    if IS_KAGGLE:
        INPUT_DIR = Path(KAGGLE_INPUT_DIR) / 'input'
        OUTPUT_DIR = Path(KAGGLE_WORKING_DIR) / 'output'
        MODELS_DIR = Path(KAGGLE_INPUT_DIR) / 'models'
    else:
        INPUT_DIR = LOCAL_INPUT_DIR
        OUTPUT_DIR = LOCAL_OUTPUT_DIR
        MODELS_DIR = PROJECT_ROOT / 'models'

    # Create directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Video processing
    TARGET_FPS = 30
    TARGET_RESOLUTION = (640, 480)
    MAX_VIDEO_SIZE_MB = 500
    CLIP_DURATION_SEC = None  # full video — no clipping

    # Detection
    DETECTION_MODEL = 'yolo11x-pose'
    DETECTION_CONFIDENCE = 0.5
    TRACKER_CONFIG = 'bytetrack.yaml'

    # Processing
    BATCH_PROCESSING = True
    ASYNC_PROCESSING = False  # Kaggle doesn't support async well
    REQUEST_TIMEOUT_SECONDS = 300

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_JSON_FORMAT = True
    LOG_FILE = OUTPUT_DIR / 'processing.log'

    # Feature flags
    ENABLE_VISUALIZATION = True
    ENABLE_BALL_DETECTION = True
    SAVE_INTERMEDIATE_FRAMES = False


def get_config() -> Config:
    """Get configuration instance."""
    return Config()
