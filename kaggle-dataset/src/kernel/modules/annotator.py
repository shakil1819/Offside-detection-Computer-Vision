"""
Frame annotation: draw bounding boxes, keypoints, offside lines, and verdict overlay.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from .object_detector import Detection
from .constants import TEAM_COLORS, FOOT_KEYPOINTS


class Annotator:
    """Draw detection overlays on frames for annotated output video."""

    # Font and size constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LABEL = 0.55
    FONT_SCALE_VERDICT = 1.8
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2
    LINE_THICKNESS = 3
    VERDICT_BAR_HEIGHT = 80

    @staticmethod
    def draw_player_bbox(
        frame: np.ndarray,
        detection: Detection,
        team_id: int,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw bounding box and label for one player.

        Args:
            frame: BGR frame
            detection: Player detection
            team_id: 0=attacking, 1=defending
            label: Optional custom label (e.g. track ID)

        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = map(int, detection.bbox)
        color = TEAM_COLORS['attacking'] if team_id == 0 else TEAM_COLORS['defending']

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Annotator.BOX_THICKNESS)

        # Label background
        text = label or f"{'ATK' if team_id == 0 else 'DEF'}"
        if detection.track_id is not None:
            text = f"{text}#{detection.track_id}"
        (tw, th), _ = cv2.getTextSize(text, Annotator.FONT, Annotator.FONT_SCALE_LABEL, Annotator.FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 2, y1 - 4),
            Annotator.FONT, Annotator.FONT_SCALE_LABEL, (255, 255, 255), Annotator.FONT_THICKNESS,
        )

        return frame

    @staticmethod
    def draw_foot_keypoints(
        frame: np.ndarray,
        detection: Detection,
        team_id: int,
    ) -> np.ndarray:
        """Draw ankle keypoints (the reference points for offside).

        Args:
            frame: BGR frame
            detection: Player detection with keypoints
            team_id: 0=attacking, 1=defending

        Returns:
            Annotated frame
        """
        if detection.keypoints is None or len(detection.keypoints) < 17:
            return frame

        color = TEAM_COLORS['attacking'] if team_id == 0 else TEAM_COLORS['defending']

        for kp_idx in FOOT_KEYPOINTS:
            kp = detection.keypoints[kp_idx]
            kx, ky = int(kp[0]), int(kp[1])
            if kx > 0 and ky > 0:  # Valid keypoint
                cv2.circle(frame, (kx, ky), 5, color, -1)
                cv2.circle(frame, (kx, ky), 7, (255, 255, 255), 1)  # White outline

        return frame

    @staticmethod
    def draw_offside_line(
        frame: np.ndarray,
        line_x: float,
        label: str = "OFFSIDE LINE",
    ) -> np.ndarray:
        """Draw vertical offside line at the second-last defender's position.

        Args:
            frame: BGR frame
            line_x: X coordinate for the offside line
            label: Text label for the line

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        x = int(line_x)

        # Draw dashed vertical line
        dash_len = 20
        gap_len = 10
        color = TEAM_COLORS['offside_line']

        y = 0
        while y < h:
            y_end = min(y + dash_len, h)
            cv2.line(frame, (x, y), (x, y_end), color, Annotator.LINE_THICKNESS)
            y += dash_len + gap_len

        # Label
        (tw, th), _ = cv2.getTextSize(label, Annotator.FONT, 0.6, 2)
        cv2.rectangle(frame, (x + 4, 10), (x + tw + 10, th + 18), color, -1)
        cv2.putText(frame, label, (x + 7, th + 14), Annotator.FONT, 0.6, (0, 0, 0), 2)

        return frame

    @staticmethod
    def draw_verdict_bar(
        frame: np.ndarray,
        verdict: str,
        confidence: float,
        incident_type: str = "offside",
    ) -> np.ndarray:
        """Draw verdict overlay bar at the bottom of the frame.

        Args:
            frame: BGR frame
            verdict: e.g. "OFFSIDE", "ONSIDE", "GOAL", "NO-GOAL"
            confidence: 0.0-1.0 confidence score
            incident_type: "offside" or "goal"

        Returns:
            Annotated frame with verdict bar
        """
        h, w = frame.shape[:2]
        bar_y = h - Annotator.VERDICT_BAR_HEIGHT

        # Background color by verdict
        offside_verdicts = {"OFFSIDE", "NO_GOAL", "NO-GOAL"}
        bar_color = (0, 0, 200) if verdict in offside_verdicts else (0, 160, 0)
        if "UNCERTAIN" in verdict:
            bar_color = (0, 140, 200)

        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), bar_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Verdict text
        verdict_text = verdict.replace("_", " ")
        cv2.putText(
            frame, verdict_text,
            (20, bar_y + 52),
            Annotator.FONT, Annotator.FONT_SCALE_VERDICT, (255, 255, 255),
            Annotator.FONT_THICKNESS + 1,
        )

        # Confidence text
        conf_text = f"{confidence * 100:.0f}% confidence"
        cv2.putText(
            frame, conf_text,
            (w - 280, bar_y + 52),
            Annotator.FONT, 0.9, (255, 255, 255), 2,
        )

        return frame

    @classmethod
    def annotate_offside_frame(
        cls,
        frame: np.ndarray,
        players: List[Detection],
        teams: Dict[int, int],
        offside_line_x: Optional[float],
        verdict: str,
        confidence: float,
    ) -> np.ndarray:
        """Full annotation for an offside freeze frame.

        Draws: player bboxes + team colors, foot keypoints,
        offside line, verdict bar.

        Args:
            frame: BGR frame
            players: All detected players
            teams: {player_index: team_id} mapping
            offside_line_x: X coordinate of second-last defender foot
            verdict: "OFFSIDE" or "ONSIDE"
            confidence: Confidence score

        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()

        for i, player in enumerate(players):
            team_id = teams.get(i, 0)
            cls.draw_player_bbox(annotated, player, team_id)
            cls.draw_foot_keypoints(annotated, player, team_id)

        if offside_line_x is not None:
            cls.draw_offside_line(annotated, offside_line_x)

        cls.draw_verdict_bar(annotated, verdict, confidence, "offside")

        return annotated

    @classmethod
    def annotate_goal_frame(
        cls,
        frame: np.ndarray,
        ball_pos: Optional[Tuple[int, int]],
        goal_line_x: float,
        verdict: str,
        confidence: float,
    ) -> np.ndarray:
        """Full annotation for a goal-line freeze frame.

        Args:
            frame: BGR frame
            ball_pos: (x, y) center of detected ball, or None
            goal_line_x: X coordinate of the goal line
            verdict: "GOAL" or "NO-GOAL"
            confidence: Confidence score

        Returns:
            Fully annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Goal line (solid white)
        gx = int(goal_line_x)
        cv2.line(annotated, (gx, 0), (gx, h), (255, 255, 255), 3)
        cv2.putText(annotated, "GOAL LINE", (gx + 5, 30), cls.FONT, 0.7, (255, 255, 255), 2)

        # Ball marker
        if ball_pos is not None:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 14, TEAM_COLORS['ball'], -1)
            cv2.circle(annotated, (bx, by), 16, (0, 0, 0), 2)
            cv2.putText(annotated, "BALL", (bx + 18, by), cls.FONT, 0.6, TEAM_COLORS['ball'], 2)

        cls.draw_verdict_bar(annotated, verdict, confidence, "goal")

        return annotated
