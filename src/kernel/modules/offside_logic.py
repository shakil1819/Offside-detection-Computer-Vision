"""
Offside detection logic using player positions and keypoints.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from .object_detector import Detection
from .constants import (
    FOOT_KEYPOINTS,
    KMEANS_N_CLUSTERS,
    KMEANS_RANDOM_STATE,
    KMEANS_N_INIT,
    TORSO_START_PERCENT,
    TORSO_END_PERCENT,
    TORSO_X_MARGIN_PERCENT,
)


class OffsideAnalyzer:
    """Analyze offside positions based on player detections."""

    @staticmethod
    def get_jersey_color(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extract average jersey color from torso region of player.

        Args:
            frame: Input frame (BGR)
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Average color as BGR array
        """
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1

        # Define torso region (30%-70% of height, with margins on sides)
        torso_y1 = y1 + int(height * TORSO_START_PERCENT)
        torso_y2 = y1 + int(height * TORSO_END_PERCENT)
        torso_x1 = x1 + int(width * TORSO_X_MARGIN_PERCENT)
        torso_x2 = x2 - int(width * TORSO_X_MARGIN_PERCENT)

        # Validate region
        if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
            return np.array([128, 128, 128])

        # Extract ROI
        roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if roi.size == 0:
            return np.array([128, 128, 128])

        # Return average color
        return np.mean(roi, axis=(0, 1))

    @staticmethod
    def classify_teams(
        players: List[Detection],
        frame: np.ndarray,
    ) -> Dict[int, int]:
        """
        Classify players as attacking (0) or defending (1) using K-means on jersey colors.

        Args:
            players: List of player detections
            frame: Input frame

        Returns:
            Dictionary mapping player index to team ID (0=attacking, 1=defending)
        """
        if len(players) < 2:
            return {i: 0 for i in range(len(players))}

        # Extract jersey colors
        colors = np.array([OffsideAnalyzer.get_jersey_color(frame, p.bbox) for p in players])

        # Cluster into 2 teams
        kmeans = KMeans(
            n_clusters=KMEANS_N_CLUSTERS,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        )
        labels = kmeans.fit_predict(colors)

        # Determine which cluster is attacking (lower avg x-coordinate)
        centers_x = [p.center[0] for p in players]
        avg_x0 = np.mean([centers_x[i] for i in range(len(players)) if labels[i] == 0])
        avg_x1 = np.mean([centers_x[i] for i in range(len(players)) if labels[i] == 1])
        attacking_label = 0 if avg_x0 < avg_x1 else 1

        # Return team assignments (0=attacking, 1=defending)
        return {i: (0 if labels[i] == attacking_label else 1) for i in range(len(players))}

    @staticmethod
    def get_foot_x(player: Detection) -> float:
        """
        Get x-coordinate of player's foot (closer to goal).

        Tries to use ankle keypoints, falls back to bbox center.

        Args:
            player: Player detection

        Returns:
            X-coordinate of foot position
        """
        if player.keypoints is not None and len(player.keypoints) > 16:
            left_foot_x = player.keypoints[15][0]  # left_ankle
            right_foot_x = player.keypoints[16][0]  # right_ankle

            if left_foot_x > 0 and right_foot_x > 0:
                # Return foot farther from goal (higher x-coordinate)
                return max(left_foot_x, right_foot_x)

        # Fallback to bbox center
        return player.center[0]

    @staticmethod
    def analyze_offside(
        attackers: List[Detection],
        defenders: List[Detection],
        teams: Dict[int, int],  # Player index to team ID
    ) -> Tuple[str, float, Dict]:
        """
        Determine if attacking player is in offside position.

        Offside rule: Attacker is offside if farther from goal (higher x) than
        second-last defender and ball.

        Args:
            attackers: List of attacking players
            defenders: List of defending players
            teams: Mapping of player detection index to team ID

        Returns:
            (verdict, confidence, analysis_data) where:
            - verdict: "OFFSIDE" or "ONSIDE"
            - confidence: Confidence score (0-1)
            - analysis_data: Dict with positions for visualization
        """
        # FIFA rule: if offside cannot be proven, the call is ONSIDE.
        # Benefit of the doubt always goes to the attacker.
        if not attackers:
            return "ONSIDE", 0.5, {"reason": "no_attackers_detected"}

        if not defenders:
            return "ONSIDE", 0.5, {"reason": "no_defenders_detected"}

        # Sort defenders by x-coordinate (goal proximity, highest x = closest to goal)
        defenders_sorted = sorted(defenders, key=lambda p: OffsideAnalyzer.get_foot_x(p), reverse=True)

        # Use second-last defender when available; fall back to last defender
        # (covers the case where only the goalkeeper is detected)
        second_last_defender = defenders_sorted[1] if len(defenders) >= 2 else defenders_sorted[0]
        defender_x = OffsideAnalyzer.get_foot_x(second_last_defender)

        # Find most advanced attacker
        attackers_sorted = sorted(attackers, key=lambda p: OffsideAnalyzer.get_foot_x(p), reverse=True)
        most_advanced_attacker = attackers_sorted[0]
        attacker_x = OffsideAnalyzer.get_foot_x(most_advanced_attacker)

        # Determine offside
        is_offside = attacker_x > defender_x
        verdict = "OFFSIDE" if is_offside else "ONSIDE"

        # Confidence based on distance
        distance = abs(attacker_x - defender_x)
        confidence = min(1.0, 0.7 + (distance / 100.0) * 0.3)

        analysis_data = {
            "defender_x": float(defender_x),
            "attacker_x": float(attacker_x),
            "distance": float(distance),
            "confidence": float(confidence),
            "defender_bbox": second_last_defender.bbox,
            "attacker_bbox": most_advanced_attacker.bbox,
        }

        return verdict, confidence, analysis_data
