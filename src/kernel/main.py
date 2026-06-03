"""
Main entry point for Atlético Intelligence Kaggle Kernel processing.

Orchestrates video processing pipeline:
1. Load video
2. Extract clip around specified frame
3. Detect players, ball, referee
4. Analyze offside or goal position
5. Generate verdict and visualization
6. Save results
"""

import cv2
import json
import sys
from pathlib import Path
from typing import Tuple

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernel.config import get_config
from src.kernel.modules.logger import get_logger
from src.kernel.modules.validators import (
    validate_video_file,
    validate_frame_number,
    validate_incident_type,
)
from src.kernel.modules.video_processor import load_video
from src.kernel.modules.object_detector import ObjectDetector
from src.kernel.modules.offside_logic import OffsideAnalyzer
from src.kernel.modules.goal_logic import GoalAnalyzer
from src.kernel.modules.ball_detector import BallDetector
from src.kernel.modules.visual_generator import VisualGenerator
from src.kernel.modules.annotator import Annotator


def process_single_incident(
    video_path: str,
    frame_number: int,
    incident_type: str,
    output_dir: Path,
    config,
    logger,
) -> dict:
    """
    Process a single incident (offside or goal).

    Args:
        video_path: Path to input video
        frame_number: Frame to analyze
        incident_type: 'offside' or 'goal'
        output_dir: Directory to save results
        config: Configuration object
        logger: Logger instance

    Returns:
        Result dictionary with verdict, confidence, paths
    """
    result = {
        'status': 'pending',
        'verdict': None,
        'confidence': None,
        'clip_path': None,
        'diagram_path': None,
        'error': None,
    }

    try:
        # Validate inputs
        logger.info(f"Processing incident: {incident_type} at frame {frame_number}")

        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            result['error'] = error_msg
            result['status'] = 'failed'
            logger.error(error_msg)
            return result

        # Load video
        logger.info(f"Loading video: {video_path}")
        video = load_video(video_path)

        # Validate frame number
        is_valid, error_msg = validate_frame_number(frame_number, video.total_frames)
        if not is_valid:
            result['error'] = error_msg
            result['status'] = 'failed'
            logger.error(error_msg)
            return result

        # Extract clip
        logger.info(f"Extracting clip around frame {frame_number}")
        clip_path = str(output_dir / f'clip_{incident_type}.mp4')
        video.extract_clip(output_path=clip_path)
        result['clip_path'] = clip_path
        logger.info(f"Clip saved: {clip_path}")

        # Get frozen frame for analysis
        logger.info(f"Extracting frozen frame at {frame_number}")
        frame = video.get_frame(frame_number)
        if frame is None:
            result['error'] = f"Cannot extract frame {frame_number}"
            result['status'] = 'failed'
            return result

        # Detect objects
        logger.info("Detecting players and ball")
        detector = ObjectDetector(config.DETECTION_MODEL)
        detections = detector.detect(frame, confidence_threshold=config.DETECTION_CONFIDENCE)
        logger.info(f"Detected {len(detections)} players")

        # Process based on incident type
        if incident_type.lower() == 'offside':
            logger.info("Analyzing offside position")

            players = detections
            teams = OffsideAnalyzer.classify_teams(players, frame)
            attackers = [p for i, p in enumerate(players) if teams.get(i, 0) == 0]
            defenders = [p for i, p in enumerate(players) if teams.get(i, 0) == 1]

            verdict, confidence, analysis_data = OffsideAnalyzer.analyze_offside(
                attackers,
                defenders,
                teams,
            )

            result['verdict'] = verdict
            result['confidence'] = float(confidence)

            # Annotated freeze frame (overlays applied to the incident frame)
            offside_line_x = analysis_data.get('defender_x')
            annotated_frame = Annotator.annotate_offside_frame(
                frame=frame,
                players=players,
                teams=teams,
                offside_line_x=offside_line_x,
                verdict=verdict,
                confidence=confidence,
            )

            # Save annotated freeze frame as PNG
            freeze_path = str(output_dir / f'freeze_{incident_type}.png')
            cv2.imwrite(freeze_path, annotated_frame)
            result['freeze_path'] = freeze_path
            logger.info(f"Annotated freeze frame saved: {freeze_path}")

            # Generate top-down 3D diagram
            if config.ENABLE_VISUALIZATION:
                logger.info("Generating offside diagram")
                diagram_path = str(output_dir / f'diagram_{incident_type}.png')
                VisualGenerator.offside_3d_diagram(
                    defender_bbox=analysis_data.get('defender_bbox'),
                    attacker_bbox=analysis_data.get('attacker_bbox'),
                    ball_x=0,
                    verdict=verdict,
                    analysis_data=analysis_data,
                    output_path=diagram_path,
                )
                result['diagram_path'] = diagram_path
                logger.info(f"Diagram saved: {diagram_path}")

        elif incident_type.lower() == 'goal':
            logger.info("Analyzing goal-line crossing")

            ball_detector = BallDetector()
            ball_pos = ball_detector.detect(frame)

            if ball_pos is None:
                logger.warning("Ball not detected for goal analysis")
                verdict, confidence, analysis_data = 'UNCERTAIN', 0.0, {}
            else:
                verdict, confidence, analysis_data = GoalAnalyzer.analyze_goal(
                    ball=ball_pos,
                    frame_height=video.frame_height,
                )

            result['verdict'] = verdict
            result['confidence'] = float(confidence)

            # Annotated freeze frame
            goal_line_x = video.frame_width * 0.95
            annotated_frame = Annotator.annotate_goal_frame(
                frame=frame,
                ball_pos=ball_pos,
                goal_line_x=goal_line_x,
                verdict=verdict,
                confidence=confidence,
            )

            freeze_path = str(output_dir / f'freeze_{incident_type}.png')
            cv2.imwrite(freeze_path, annotated_frame)
            result['freeze_path'] = freeze_path
            logger.info(f"Annotated freeze frame saved: {freeze_path}")

            if config.ENABLE_VISUALIZATION:
                logger.info("Generating goal-line diagram")
                diagram_path = str(output_dir / f'diagram_{incident_type}.png')
                VisualGenerator.goal_crossing_diagram(
                    ball_x=ball_pos[0] if ball_pos else 0,
                    goal_line_x=goal_line_x,
                    verdict=verdict,
                    analysis_data=analysis_data,
                    output_path=diagram_path,
                )
                result['diagram_path'] = diagram_path
                logger.info(f"Diagram saved: {diagram_path}")

        result['status'] = 'completed'
        logger.info(f"Processing complete: {result['verdict']} (confidence: {result['confidence']:.2f})")

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Processing failed: {e}", exc_info=True)

    return result


def main():
    """Main entry point for Kaggle Kernel processing."""
    config = get_config()
    logger = get_logger('Athletic Intelligence', level=config.LOG_LEVEL)

    logger.info(f"Starting Atlético Intelligence processing")
    logger.info(f"Environment: {'Kaggle' if config.IS_KAGGLE else 'Local'}")
    logger.info(f"Input dir: {config.INPUT_DIR}")
    logger.info(f"Output dir: {config.OUTPUT_DIR}")

    # Example: Process first video in input directory
    input_dir = Path(config.INPUT_DIR)
    video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))

    if not video_files:
        logger.warning(f"No videos found in {input_dir}")
        return

    results = []

    for video_file in video_files[:1]:  # Process first video
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_file.name}")
        logger.info(f"{'='*60}")

        # Create output subdir
        match_dir = config.OUTPUT_DIR / video_file.stem
        match_dir.mkdir(parents=True, exist_ok=True)

        # Process offside (example: frame 450, incident_type='offside')
        # In production, these would come from user input or config file
        result = process_single_incident(
            video_path=str(video_file),
            frame_number=450,
            incident_type='offside',
            output_dir=match_dir,
            config=config,
            logger=logger,
        )

        results.append({
            'video': video_file.name,
            'incident_type': 'offside',
            'result': result,
        })

        # Save results to JSON
        results_json_path = match_dir / 'results.json'
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved: {results_json_path}")

    logger.info("\nProcessing complete!")


if __name__ == '__main__':
    main()
