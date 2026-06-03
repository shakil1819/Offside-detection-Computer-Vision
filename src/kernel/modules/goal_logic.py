"""
Goal-line detection logic.
"""

from typing import Tuple, Dict, Optional
from .object_detector import Detection


class GoalAnalyzer:
    """Analyze goal-line crossings."""

    @staticmethod
    def analyze_goal(
        ball: Optional[Detection],
        frame_height: int,
        goal_line_x: Optional[float] = None,
    ) -> Tuple[str, float, Dict]:
        """
        Determine if ball crossed the goal line.

        Args:
            ball: Ball detection (or None if not detected)
            frame_height: Frame height in pixels
            goal_line_x: X-coordinate of goal line (None for auto-detect)

        Returns:
            (verdict, confidence, analysis_data) where:
            - verdict: "GOAL", "NO_GOAL", or "UNCERTAIN"
            - confidence: Confidence score (0-1)
            - analysis_data: Dict with details for visualization
        """
        if ball is None:
            # Cannot prove goal — benefit of the doubt: NO-GOAL
            return "NO-GOAL", 0.5, {"reason": "ball_not_detected"}

        ball_x = ball.center[0]

        # Default: goal line at edges (x=0 or x=frame_width)
        if goal_line_x is None:
            # Assume goal line is at the frame edge (right edge)
            # This is a simplification; real implementation would use field calibration
            goal_line_x = frame_height * 0.95  # Slightly before edge

        # Simple logic: if ball center is beyond goal line, it's a goal
        # In reality, would need more sophisticated ball tracking and 3D analysis
        is_crossing = ball_x > goal_line_x

        verdict = "GOAL" if is_crossing else "NO_GOAL"

        # Confidence based on distance from goal line
        distance_to_line = abs(ball_x - goal_line_x)
        confidence = min(1.0, 0.5 + (distance_to_line / 100.0) * 0.5)

        analysis_data = {
            "ball_x": float(ball_x),
            "goal_line_x": float(goal_line_x),
            "distance_to_line": float(distance_to_line),
            "confidence": float(confidence),
            "ball_bbox": ball.bbox,
        }

        return verdict, confidence, analysis_data
