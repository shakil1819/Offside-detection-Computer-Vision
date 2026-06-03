"""
Ball detection using HSV color segmentation and circle detection.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from .constants import BALL_MIN_AREA, BALL_MAX_AREA, BALL_MIN_RADIUS, BALL_MAX_RADIUS, BALL_CIRCULARITY_THRESHOLD


class BallDetector:
    """Detect soccer ball in video frames using color and shape."""

    def __init__(self):
        """Initialize ball detector with color ranges for white/orange balls."""
        # HSV ranges for white balls
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])

        # HSV ranges for orange/yellow balls
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([15, 255, 255])

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect ball in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            (x, y) center coordinates or None if not found
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for white and orange balls
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        mask_orange = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        mask = cv2.bitwise_or(mask_white, mask_orange)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_circle = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area
            if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
                continue

            # Fit circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)

            # Filter by radius
            if radius < BALL_MIN_RADIUS or radius > BALL_MAX_RADIUS:
                continue

            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < BALL_CIRCULARITY_THRESHOLD:
                continue

            # Score: combination of area and circularity
            score = area * circularity
            if score > best_score:
                best_score = score
                best_circle = (int(x), int(y))

        return best_circle

    def detect_trajectory(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Tuple[list, list]:
        """Detect ball in all frames of a video.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None=all)

        Returns:
            (ball_positions, frame_indices) where positions are (x,y) or None
        """
        cap = cv2.VideoCapture(video_path)
        ball_positions = []
        frame_indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            ball_pos = self.detect(frame)
            ball_positions.append(ball_pos)
            frame_indices.append(frame_idx)
            frame_idx += 1

        cap.release()
        return ball_positions, frame_indices

    @staticmethod
    def detect_kick_event(
        ball_positions: list,
        frame_indices: list,
        velocity_sigma: float = 1.5,
    ) -> Optional[int]:
        """Detect the frame where the ball was kicked (velocity spike).

        Computes frame-to-frame ball velocity and returns the first frame
        where velocity exceeds mean + (velocity_sigma * std). This approximates
        the moment of pass/shot for offside timing.

        Args:
            ball_positions: List of (x, y) or None per frame
            frame_indices: Corresponding frame index list
            velocity_sigma: Threshold multiplier above mean velocity

        Returns:
            Frame index immediately after the kick, or None if not detected
        """
        if len(ball_positions) < 4:
            return None

        velocities = []
        for i in range(1, len(ball_positions)):
            prev = ball_positions[i - 1]
            curr = ball_positions[i]
            if prev is None or curr is None:
                velocities.append(0.0)
            else:
                vel = float(np.linalg.norm(np.array(curr) - np.array(prev)))
                velocities.append(vel)

        if not velocities:
            return None

        mean_vel = np.mean(velocities)
        std_vel = np.std(velocities)
        threshold = mean_vel + velocity_sigma * std_vel

        for i, vel in enumerate(velocities):
            if vel > threshold and i + 1 < len(frame_indices):
                return frame_indices[i + 1]

        return None
