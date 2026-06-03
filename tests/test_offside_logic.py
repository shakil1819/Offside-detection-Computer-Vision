"""
Unit tests for offside logic module.
"""

import pytest
import numpy as np
from src.kernel.modules.object_detector import Detection
from src.kernel.modules.offside_logic import OffsideAnalyzer


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 128


@pytest.fixture
def sample_players():
    """Create sample player detections."""
    return [
        Detection(
            class_id=0,
            class_name="person",
            bbox=(100, 100, 150, 250),
            confidence=0.9,
            keypoints=np.array([[120, 250], [140, 250]] + [[0, 0]] * 15),  # feet at bottom
        ),
        Detection(
            class_id=0,
            class_name="person",
            bbox=(200, 100, 250, 250),
            confidence=0.85,
            keypoints=np.array([[220, 250], [240, 250]] + [[0, 0]] * 15),
        ),
        Detection(
            class_id=0,
            class_name="person",
            bbox=(300, 100, 350, 250),
            confidence=0.8,
            keypoints=np.array([[320, 250], [340, 250]] + [[0, 0]] * 15),
        ),
    ]


def test_get_foot_x_with_keypoints(sample_players):
    """Test foot x-coordinate extraction with keypoints."""
    player = sample_players[0]
    foot_x = OffsideAnalyzer.get_foot_x(player)
    assert foot_x == 140  # Right foot (max of the two)


def test_get_foot_x_without_keypoints():
    """Test foot x-coordinate falls back to center without keypoints."""
    player = Detection(
        class_id=0,
        class_name="person",
        bbox=(100, 100, 150, 250),
        confidence=0.9,
        keypoints=None,
    )
    foot_x = OffsideAnalyzer.get_foot_x(player)
    assert foot_x == 125  # Center x


def test_classify_teams(sample_frame, sample_players):
    """Test team classification using K-means."""
    teams = OffsideAnalyzer.classify_teams(sample_players, sample_frame)

    assert len(teams) == 3
    assert all(team_id in [0, 1] for team_id in teams.values())

    # Check that teams are assigned (0=attacking, 1=defending)
    # Note: actual assignment depends on K-means, just verify it's valid


def test_analyze_offside_clear_offside(sample_players):
    """Test clear offside position."""
    attackers = [sample_players[2]]  # Most advanced (x=340)
    defenders = [sample_players[0], sample_players[1]]  # x=120, x=220
    teams = {0: 1, 1: 1, 2: 0}  # Attackers, defenders

    verdict, confidence, data = OffsideAnalyzer.analyze_offside(
        attackers,
        defenders,
        teams,
    )

    assert verdict == "OFFSIDE"
    assert confidence > 0.5
    assert data["attacker_x"] == 340
    assert data["defender_x"] == 220


def test_analyze_offside_clear_onside(sample_players):
    """Test clear onside position."""
    attackers = [sample_players[0]]  # Least advanced (x=120)
    defenders = [sample_players[1], sample_players[2]]  # x=220, x=340
    teams = {0: 0, 1: 1, 2: 1}

    verdict, confidence, data = OffsideAnalyzer.analyze_offside(
        attackers,
        defenders,
        teams,
    )

    assert verdict == "ONSIDE"
    assert confidence > 0.5


def test_analyze_offside_no_attackers(sample_players):
    """Test with no attackers."""
    attackers = []
    defenders = [sample_players[0], sample_players[1]]

    verdict, confidence, data = OffsideAnalyzer.analyze_offside(
        attackers,
        defenders,
        {},
    )

    assert verdict == "UNCERTAIN"
    assert confidence == 0.0


def test_analyze_offside_not_enough_defenders(sample_players):
    """Test with insufficient defenders."""
    attackers = [sample_players[2]]
    defenders = [sample_players[0]]  # Only 1 defender

    verdict, confidence, data = OffsideAnalyzer.analyze_offside(
        attackers,
        defenders,
        {},
    )

    assert verdict == "UNCERTAIN"
    assert confidence < 0.5
