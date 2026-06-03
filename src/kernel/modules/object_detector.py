"""
Player and object detection using YOLOv11x-pose model.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Absolute path to tracker config — works regardless of working directory
_BYTETRACK_CONFIG = Path(__file__).parents[4] / "bytetrack.yaml"

# COCO person class ID
_PERSON_CLASS_ID = 0

# Inference image size: 1280px for soccer footage
# Soccer stadiums have lots of players far from camera — 640 loses them
_INFERENCE_IMGSZ = 1280


@dataclass
class Detection:
    """A detected object in a frame."""

    class_id: int
    class_name: str  # "person"
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[np.ndarray] = None  # (17, 2) ankle/wrist etc.
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get bbox center (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox
        return y2 - y1

    @property
    def foot_center(self) -> Tuple[float, float]:
        """Approximate foot position: bottom-center of bbox."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class ObjectDetector:
    """Detect players using YOLOv11x-pose model with ByteTrack tracking."""

    def __init__(self, model_name: str = "yolo11x-pose"):
        """Initialize detector.

        Args:
            model_name: YOLO model name without extension.
                        Ultralytics auto-downloads if not cached.
                        Options: yolo11n-pose, yolo11s-pose, yolo11x-pose
        """
        self.model = YOLO(f"{model_name}.pt")
        self.model_name = model_name

        # Verify tracker config exists
        if not _BYTETRACK_CONFIG.exists():
            raise FileNotFoundError(
                f"ByteTrack config not found: {_BYTETRACK_CONFIG}. "
                "Ensure bytetrack.yaml is at project root."
            )
        self.tracker_config = str(_BYTETRACK_CONFIG)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        verbose: bool = False,
    ) -> List[Detection]:
        """Detect persons in a single frame (no tracking).

        Args:
            frame: BGR frame from OpenCV
            confidence_threshold: Minimum detection confidence
            verbose: Print YOLO output

        Returns:
            List of Detection objects (persons only)
        """
        results = self.model.predict(
            frame,
            conf=confidence_threshold,
            classes=[_PERSON_CLASS_ID],  # Only detect persons
            imgsz=_INFERENCE_IMGSZ,
            agnostic_nms=True,  # Suppress cross-class NMS conflicts
            verbose=verbose,
        )

        return self._parse_result(results[0])

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.45,
        verbose: bool = False,
    ) -> List[Detection]:
        """Detect and track persons across frames using ByteTrack.

        Args:
            frame: BGR frame from OpenCV
            confidence_threshold: Minimum detection confidence
            verbose: Print YOLO output

        Returns:
            List of Detection objects with track_id set
        """
        results = self.model.track(
            frame,
            conf=confidence_threshold,
            classes=[_PERSON_CLASS_ID],
            imgsz=_INFERENCE_IMGSZ,
            agnostic_nms=True,
            persist=True,  # Keep track state between calls
            tracker=self.tracker_config,
            verbose=verbose,
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
                class_id=_PERSON_CLASS_ID,
                class_name="person",
                bbox=tuple(box),
                confidence=float(conf),
                keypoints=keypoints,
                track_id=int(track_id),
            )
            detections.append(detection)

        return detections

    def _parse_result(self, result) -> List[Detection]:
        """Parse a single YOLO result into Detection objects.

        Only returns person class detections.
        """
        detections = []

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
            if cls_id != _PERSON_CLASS_ID:
                continue

            keypoints = keypoints_data[i] if keypoints_data is not None else None

            detections.append(Detection(
                class_id=cls_id,
                class_name="person",
                bbox=tuple(box),
                confidence=float(conf),
                keypoints=keypoints,
                track_id=None,
            ))

        return detections
