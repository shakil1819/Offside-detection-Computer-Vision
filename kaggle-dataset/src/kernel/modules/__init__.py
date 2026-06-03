"""Kernel modules package."""

from .object_detector import ObjectDetector, Detection
from .offside_logic import OffsideAnalyzer
from .goal_logic import GoalAnalyzer
from .ball_detector import BallDetector
from .video_processor import VideoProcessor, load_video
from .visual_generator import VisualGenerator
from .annotator import Annotator

__all__ = [
    'ObjectDetector', 'Detection',
    'OffsideAnalyzer',
    'GoalAnalyzer',
    'BallDetector',
    'VideoProcessor', 'load_video',
    'VisualGenerator',
    'Annotator',
]
