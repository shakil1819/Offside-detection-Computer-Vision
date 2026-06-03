"""
Generate visualization diagrams (3D player positions, goal-line crossings).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, Dict, List
from .constants import (
    DIAGRAM_WIDTH,
    DIAGRAM_HEIGHT,
    DIAGRAM_DPI,
    DIAGRAM_BGCOLOR,
    DIAGRAM_FIELD_COLOR,
    DIAGRAM_LINE_COLOR,
    DIAGRAM_PLAYER_SIZE,
    DIAGRAM_BALL_SIZE,
    DIAGRAM_DEFENDER_COLOR,
    DIAGRAM_ATTACKER_COLOR,
    DIAGRAM_OFFSIDE_LINE_COLOR,
)


class VisualGenerator:
    """Generate visualization diagrams."""

    @staticmethod
    def offside_3d_diagram(
        defender_bbox: Tuple[float, float, float, float],
        attacker_bbox: Tuple[float, float, float, float],
        ball_x: float,
        verdict: str,
        analysis_data: Dict,
        output_path: str,
    ) -> str:
        """
        Generate a top-down view of offside situation.

        Args:
            defender_bbox: Second-last defender bounding box
            attacker_bbox: Most advanced attacker bounding box
            ball_x: Ball x-coordinate
            verdict: "OFFSIDE" or "ONSIDE"
            analysis_data: Additional analysis data
            output_path: Path to save diagram

        Returns:
            Path to output diagram
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=DIAGRAM_DPI)

        # Draw field
        field_patch = patches.Rectangle(
            (0, 0),
            DIAGRAM_WIDTH,
            DIAGRAM_HEIGHT,
            linewidth=2,
            edgecolor=DIAGRAM_LINE_COLOR,
            facecolor=DIAGRAM_FIELD_COLOR,
            alpha=0.3,
        )
        ax.add_patch(field_patch)

        # Goal lines
        ax.axvline(x=0, color=DIAGRAM_LINE_COLOR, linestyle='--', linewidth=2)
        ax.axvline(x=DIAGRAM_WIDTH, color=DIAGRAM_LINE_COLOR, linestyle='--', linewidth=2)

        # Center line
        ax.axvline(x=DIAGRAM_WIDTH / 2, color=DIAGRAM_LINE_COLOR, linestyle='--', linewidth=1, alpha=0.5)

        # Extract positions from analysis data
        defender_x = analysis_data.get('defender_x', 300)
        attacker_x = analysis_data.get('attacker_x', 400)

        # Draw offside line (second-last defender position)
        ax.axvline(x=defender_x, color=DIAGRAM_OFFSIDE_LINE_COLOR, linestyle='-', linewidth=3, label='Offside Line')

        # Draw ball position
        ax.scatter(ball_x, DIAGRAM_HEIGHT / 2, s=DIAGRAM_BALL_SIZE, c='white', marker='o', edgecolors='black', linewidth=2, label='Ball', zorder=10)

        # Draw defender
        ax.scatter(
            defender_x,
            DIAGRAM_HEIGHT * 0.3,
            s=DIAGRAM_PLAYER_SIZE,
            c=DIAGRAM_DEFENDER_COLOR,
            marker='s',
            edgecolors='black',
            linewidth=2,
            label='2nd-Last Defender',
            zorder=5,
        )

        # Draw attacker
        ax.scatter(
            attacker_x,
            DIAGRAM_HEIGHT * 0.7,
            s=DIAGRAM_PLAYER_SIZE,
            c=DIAGRAM_ATTACKER_COLOR,
            marker='^',
            edgecolors='black',
            linewidth=2,
            label='Most Advanced Attacker',
            zorder=5,
        )

        # Title with verdict
        color = 'red' if verdict == 'OFFSIDE' else 'green'
        ax.set_title(f'OFFSIDE ANALYSIS: {verdict}', fontsize=16, fontweight='bold', color=color)

        # Labels and formatting
        ax.set_xlabel('Distance to Goal (pixels)', fontsize=12)
        ax.set_ylabel('Field Position', fontsize=12)
        ax.set_xlim(-50, DIAGRAM_WIDTH + 50)
        ax.set_ylim(-50, DIAGRAM_HEIGHT + 50)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=DIAGRAM_DPI, bbox_inches='tight')
        plt.close()

        return output_path

    @staticmethod
    def goal_crossing_diagram(
        ball_x: float,
        goal_line_x: float,
        verdict: str,
        analysis_data: Dict,
        output_path: str,
    ) -> str:
        """
        Generate a visualization of goal-line crossing.

        Args:
            ball_x: Ball x-coordinate
            goal_line_x: Goal line x-coordinate
            verdict: "GOAL" or "NO_GOAL"
            analysis_data: Additional analysis data
            output_path: Path to save diagram

        Returns:
            Path to output diagram
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=DIAGRAM_DPI)

        # Draw field
        field_patch = patches.Rectangle(
            (0, 0),
            DIAGRAM_WIDTH,
            DIAGRAM_HEIGHT,
            linewidth=2,
            edgecolor=DIAGRAM_LINE_COLOR,
            facecolor=DIAGRAM_FIELD_COLOR,
            alpha=0.3,
        )
        ax.add_patch(field_patch)

        # Goal line
        ax.axvline(x=goal_line_x, color='red', linestyle='-', linewidth=4, label='Goal Line')

        # Ball position
        ax.scatter(ball_x, DIAGRAM_HEIGHT / 2, s=DIAGRAM_BALL_SIZE * 2, c='white', marker='o', edgecolors='black', linewidth=2, label='Ball', zorder=10)

        # Title with verdict
        color = 'green' if verdict == 'GOAL' else 'red'
        ax.set_title(f'GOAL-LINE ANALYSIS: {verdict}', fontsize=16, fontweight='bold', color=color)

        # Labels
        ax.set_xlabel('Distance to Goal (pixels)', fontsize=12)
        ax.set_xlim(-50, DIAGRAM_WIDTH + 50)
        ax.set_ylim(0, DIAGRAM_HEIGHT)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=DIAGRAM_DPI, bbox_inches='tight')
        plt.close()

        return output_path
