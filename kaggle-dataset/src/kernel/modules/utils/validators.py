"""
Input validation utilities.
"""

import os
import cv2
from pathlib import Path
from typing import Tuple


def validate_video_file(video_path: str, max_size_mb: int = 500) -> Tuple[bool, str]:
    """
    Validate video file format, existence, and size.

    Args:
        video_path: Path to video file
        max_size_mb: Maximum allowed file size in MB

    Returns:
        (is_valid, error_message)
    """
    # Check existence
    if not os.path.exists(video_path):
        return False, f"Video file not found: {video_path}"

    # Check extension
    valid_extensions = {'.mp4', '.avi', '.webm', '.mov', '.mkv'}
    ext = Path(video_path).suffix.lower()
    if ext not in valid_extensions:
        return False, f"Invalid format: {ext}. Supported: {valid_extensions}"

    # Check file size
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB > {max_size_mb}MB limit"

    # Check readability
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file (corrupted or unsupported codec)"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frame_count == 0 or fps == 0:
        return False, "Invalid video: no frames or FPS"

    return True, ""


def validate_frame_number(frame_num: int, total_frames: int) -> Tuple[bool, str]:
    """
    Validate frame number within video range.

    Args:
        frame_num: Frame index to validate
        total_frames: Total number of frames in video

    Returns:
        (is_valid, error_message)
    """
    if frame_num < 0:
        return False, f"Frame number cannot be negative: {frame_num}"

    if frame_num >= total_frames:
        return False, f"Frame {frame_num} exceeds video length {total_frames}"

    return True, ""


def validate_incident_type(incident_type: str) -> Tuple[bool, str]:
    """
    Validate incident type.

    Args:
        incident_type: Type of incident ('offside' or 'goal')

    Returns:
        (is_valid, error_message)
    """
    valid_types = {'offside', 'goal'}
    incident_type_lower = incident_type.lower()

    if incident_type_lower not in valid_types:
        return False, f"Invalid incident type: {incident_type}. Valid: {valid_types}"

    return True, ""


def validate_output_directory(output_dir: str) -> Tuple[bool, str]:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Output directory path

    Returns:
        (is_valid, error_message)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permission
        test_file = os.path.join(output_dir, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True, ""
    except Exception as e:
        return False, f"Cannot write to output directory: {e}"
