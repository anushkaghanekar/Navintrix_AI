"""YOLO detector wrapper.

Owns: loading trained weights, running inference on a frame, returning
structured detections. Training/fine-tuning itself lives in
training/train_detector.py (see scripts/ for dataset prep) — this module
is inference-only so the backend and evaluation code share one code path.
"""

from dataclasses import dataclass


@dataclass
class Detection:
    cls: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class VehicleDetector:
    def __init__(self, weights_path: str, confidence_threshold: float, iou_threshold: float):
        """Load a YOLO model. TODO: wire up ultralytics.YOLO(weights_path)."""
        raise NotImplementedError

    def detect(self, frame) -> list[Detection]:
        """Run inference on a single frame, return one Detection per object.

        TODO: run model inference, filter by confidence/IoU, map class
        indices to names from configs/model.yaml, return Detection list.
        """
        raise NotImplementedError
