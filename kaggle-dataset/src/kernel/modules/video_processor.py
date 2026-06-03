"""
Video processing utilities: loading, frame extraction, and clip generation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Iterator
from pathlib import Path


class VideoProcessor:
    """Handle video loading, frame extraction, and clip generation."""

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by index.

        Args:
            frame_index: Frame index (0-based)

        Returns:
            Frame as BGR numpy array, or None if invalid
        """
        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frames_range(
        self,
        start_frame: int,
        end_frame: int,
    ) -> Iterator[np.ndarray]:
        """
        Yield frames in a range.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)

        Yields:
            Frames as BGR numpy arrays
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, min(end_frame + 1, self.total_frames)):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def extract_clip(
        self,
        output_path: str,
        center_frame: int,
        duration_sec: int = 10,
    ) -> str:
        """
        Extract a clip (5-15 seconds) centered on a specific frame.

        Tries H.264 (avc1) first for browser-compatible MP4.
        Falls back to mp4v if H.264 encoder is not available.

        Args:
            output_path: Path to save output clip
            center_frame: Center frame index
            duration_sec: Duration of clip in seconds

        Returns:
            Path to output clip
        """
        # Calculate frame range
        clip_frames = int(duration_sec * self.fps)
        start_frame = max(0, center_frame - clip_frames // 2)
        end_frame = min(self.total_frames - 1, start_frame + clip_frames - 1)

        # Adjust start if we hit the video end
        if end_frame - start_frame < clip_frames:
            start_frame = max(0, end_frame - clip_frames + 1)

        # Try H.264 first (browser-compatible), fall back to mp4v
        codecs = ["avc1", "H264", "mp4v"]
        out = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )
            if out.isOpened():
                break
            out = None

        if out is None:
            raise ValueError(f"Cannot create output video (no supported codec): {output_path}")

        for frame in self.get_frames_range(start_frame, end_frame):
            out.write(frame)

        out.release()
        return output_path

    def get_video_info(self) -> dict:
        """
        Get video metadata.

        Returns:
            Dictionary with video properties
        """
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "width": self.frame_width,
            "height": self.frame_height,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
        }

    def __del__(self):
        """Cleanup: release video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()


def load_video(video_path: str) -> VideoProcessor:
    """
    Load a video file.

    Args:
        video_path: Path to video

    Returns:
        VideoProcessor instance
    """
    return VideoProcessor(video_path)
