"""YOLO detector wrapper.

Owns: loading trained weights, running inference on a frame, returning
structured detections. Training/fine-tuning itself lives in the scripts/
pipeline (see scripts/prepare_detrac.py, scripts/prepare_emergency.py and
scripts/validate_dataset.py). This module is inference-only so the backend
and evaluation code share one code path.

Class-index contract
--------------------
configs/model.yaml defines the full class set as ``classes.normal`` followed
by ``classes.emergency``. Those names, in that exact order, are the model's
class IDs (0..N): the detector maps the raw YOLO class index back to that
name list, so the config file is the single source of truth for the label
set. The dataset-prep scripts must write label IDs in the same order for
training to agree with this mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model.yaml"


@dataclass
class Detection:
    cls: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def load_class_names(config_path: str = DEFAULT_CONFIG_PATH) -> list[str]:
    """Return the id-ordered model class list from configs/model.yaml.

    The order is ``classes.normal + classes.emergency`` (see the module
    docstring for why that order matters).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        normal = cfg["classes"]["normal"]
        emergency = cfg["classes"]["emergency"]
    except (KeyError, TypeError):
        raise ValueError(
            f"config {config_path} must define 'classes.normal' and "
            "'classes.emergency'"
        )
    if not isinstance(normal, list) or not normal:
        raise ValueError(f"config {config_path}: 'classes.normal' must be a non-empty list")
    if not isinstance(emergency, list):
        raise ValueError(f"config {config_path}: 'classes.emergency' must be a list")
    return [str(name) for name in normal] + [str(name) for name in emergency]


def load_detector_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Return model runtime settings from configs/model.yaml.

    Keeps weights/thresholds config-driven so backend, scripts, and
    evaluation code do not repeat defaults in application logic.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"config {config_path} must define a 'model' mapping")

    required = ("weights_path", "confidence_threshold", "iou_threshold")
    missing = [key for key in required if key not in model_cfg]
    if missing:
        raise ValueError(
            f"config {config_path}: model is missing {', '.join(missing)}"
        )

    try:
        return {
            "weights_path": str(model_cfg["weights_path"]),
            "confidence_threshold": float(model_cfg["confidence_threshold"]),
            "iou_threshold": float(model_cfg["iou_threshold"]),
            "class_names": load_class_names(config_path),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"config {config_path}: confidence_threshold and iou_threshold "
            "must be numeric"
        ) from exc


def build_detector(config_path: str = DEFAULT_CONFIG_PATH) -> "VehicleDetector":
    """Build a VehicleDetector using weights and thresholds from config."""
    detector_cfg = load_detector_config(config_path)
    return VehicleDetector(**detector_cfg)


def _result_to_detections(result, class_names: list[str], min_confidence: float) -> list[Detection]:
    """Convert one ultralytics inference Result into Detection objects.

    ``result.boxes`` exposes three parallel sequences: ``xyxy`` (Nx4 corner
    boxes), ``conf``, and ``cls`` (class indices). Any object exposing those
    attributes works here — tests use a lightweight stub that mirrors the
    ultralytics API so this adapter can be tested without torch installed.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes.cls) == 0:
        return []

    detections: list[Detection] = []
    for i in range(len(boxes.cls)):
        confidence = float(boxes.conf[i])
        if confidence < min_confidence:
            continue
        class_id = int(boxes.cls[i])
        if 0 <= class_id < len(class_names):
            name = class_names[class_id]
        else:
            name = f"class_{class_id}"
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
        detections.append(Detection(name, confidence, x1, y1, x2, y2))
    return detections


class VehicleDetector:
    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        class_names: list[str] | None = None,
    ):
        """Load a YOLO model via ultralytics.

        ``class_names`` defaults to the id-ordered list from
        configs/model.yaml. ultralytics is imported lazily so importing this
        module (and running unit tests against the adapter) does not require
        the heavy torch stack.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = list(class_names) if class_names else load_class_names()

        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        trained_names = getattr(self.model, "names", None)
        if trained_names and len(trained_names) != len(self.class_names):
            logger.warning(
                "model weights declare %d classes but configs/model.yaml "
                "declares %d — check that the training data.yaml order matches",
                len(trained_names),
                len(self.class_names),
            )

    def detect(self, frame) -> list[Detection]:
        """Run inference on a single frame, return one Detection per object.

        Accepts whatever ultralytics can ingest (BGR numpy array from cv2,
        or an image path). NMS and the confidence cut-off are applied by
        ultralytics, then results are mapped to the config class list and
        sorted by confidence descending.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results if results is not None else []:
            detections.extend(
                _result_to_detections(result, self.class_names, self.confidence_threshold)
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
