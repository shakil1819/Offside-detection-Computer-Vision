"""Video validation utilities."""

from pathlib import Path
from typing import Tuple


def validate_video_file(file_path: str, max_size_mb: int = 500, allowed_formats: tuple = ('.mp4', '.avi', '.mov', '.mkv')) -> Tuple[bool, str]:
    """Validate video file.

    Args:
        file_path: Path to video file
        max_size_mb: Maximum file size in MB
        allowed_formats: Allowed file extensions

    Returns:
        (is_valid: bool, error_message: str)
    """
    try:
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Check file extension
        if file_path.suffix.lower() not in allowed_formats:
            return False, f"Invalid format: {file_path.suffix}. Allowed: {allowed_formats}"

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB"

        # Check file is readable
        if not file_path.is_file():
            return False, f"Not a regular file: {file_path}"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_incident_type(incident_type: str) -> Tuple[bool, str]:
    """Validate incident type.

    Args:
        incident_type: Type of incident (offside, goal)

    Returns:
        (is_valid: bool, error_message: str)
    """
    valid_types = ('offside', 'goal')
    incident_type_lower = incident_type.lower()

    if incident_type_lower not in valid_types:
        return False, f"Invalid incident type: {incident_type}. Allowed: {valid_types}"

    return True, ""


def validate_frame_number(frame_number: int, total_frames: int) -> Tuple[bool, str]:
    """Validate frame number.

    Args:
        frame_number: Frame to analyze
        total_frames: Total frames in video

    Returns:
        (is_valid: bool, error_message: str)
    """
    if frame_number < 0:
        return False, "Frame number must be non-negative"

    if frame_number >= total_frames:
        return False, f"Frame {frame_number} out of range (0-{total_frames - 1})"

    return True, ""
