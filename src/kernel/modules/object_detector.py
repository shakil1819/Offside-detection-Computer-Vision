"""
Player and object detection using YOLOv11x-pose model.
"""

import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """A detected object in a frame."""

    class_id: int
    class_name: str  # "player", "ball"
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[np.ndarray] = None  # (17, 2) for pose
    track_id: Optional[int] = None  # Tracking ID across frames

    @property
    def center(self) -> Tuple[float, float]:
        """Get bbox center coordinates."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get bbox width."""
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> float:
        """Get bbox height."""
        _, y1, _, y2 = self.bbox
        return y2 - y1


class ObjectDetector:
    """Detect players and objects using YOLOv11x-pose model."""

    def __init__(self, model_name: str = "yolo11x-pose"):
        """
        Initialize detector with YOLO model.

        Args:
            model_name: YOLO model name (e.g., 'yolo11x-pose', 'yolo11s-pose')
        """
        self.model = YOLO(f"{model_name}.pt")
        self.model_name = model_name

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        verbose: bool = False,
    ) -> List[Detection]:
        """
        Detect objects in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for detections
            verbose: Print verbose output

        Returns:
            List of detected objects
        """
        results = self.model.predict(frame, conf=confidence_threshold, verbose=verbose)
        detections = []

        result = results[0]
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        keypoints_data = (
            result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
        )

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            cls_id = int(cls_id)
            class_name = result.names[cls_id]

            # Only keep "person" class (class 0 in COCO)
            if class_name != "person":
                continue

            keypoints = keypoints_data[i] if keypoints_data is not None else None

            detection = Detection(
                class_id=cls_id,
                class_name=class_name,
                bbox=tuple(box),
                confidence=float(conf),
                keypoints=keypoints,
                track_id=None,  # Set by tracking pipeline
            )
            detections.append(detection)

        return detections

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        tracker_yaml: str = "bytetrack.yaml",
    ) -> List[Detection]:
        """
        Detect objects with tracking across frames.

        Args:
            frame: Input frame
            confidence_threshold: Minimum confidence
            tracker_yaml: Path to tracker config

        Returns:
            List of detected objects with track IDs
        """
        results = self.model.track(
            frame,
            conf=confidence_threshold,
            persist=True,
            tracker=tracker_yaml,
            verbose=False,
        )

        detections = []
        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        keypoints_data = (
            result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
        )

        for i, (box, conf, track_id) in enumerate(zip(boxes, confidences, track_ids)):
            keypoints = keypoints_data[i] if keypoints_data is not None else None

            detection = Detection(
                class_id=0,
                class_name="person",
                bbox=tuple(box),
                confidence=float(conf),
                keypoints=keypoints,
                track_id=int(track_id),
            )
            detections.append(detection)

        return detections
