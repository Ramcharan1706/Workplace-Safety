"""YOLOv8 object detector service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import DEFAULT_MODEL_PATH, DETECTION_CONFIDENCE, DETECTION_IOU, SUPPORTED_CLASSES
from core.schemas import Detection


@dataclass(slots=True)
class DetectionModelInfo:
    model_path: str
    classes_available: list[str]


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO with normalized output records."""

    _ALIASES: dict[str, str] = {
        # Helmet aliases - all colors
        "hard_hat": "hardhat",
        "hardhats": "hardhat",
        "hard_hats": "hardhat",
        "safety_helmet": "helmet",
        "safetyhelm": "helmet",
        "helmets": "helmet",
        # Color-specific helmet mappings
        "yellow_hard_hat": "yellow_helmet",
        "white_hard_hat": "white_helmet",
        "red_hard_hat": "red_helmet",
        "orange_hard_hat": "orange_helmet",
        "blue_hard_hat": "blue_helmet",
        "green_hard_hat": "green_helmet",
        "black_hard_hat": "black_helmet",
        "grey_hard_hat": "grey_helmet",
        # Vest aliases - all colors
        "safety_vest": "safety_vest",
        "safetyvest": "safety_vest",
        "safety_jacket": "safety_vest",
        "hi_vis": "orange_vest",
        "hi_vis_vest": "orange_vest",
        "high_visibility_vest": "orange_vest",
        "high_vis": "orange_vest",
        "reflective_vest": "reflective_vest",
        "reflective_jacket": "reflective_jacket",
        "vests": "vest",
        "yellow_reflective_vest": "yellow_vest",
        "white_reflective_vest": "white_vest",
        "red_reflective_vest": "red_vest",
        "green_reflective_vest": "green_vest",
        # Machinery aliases - heavy equipment
        "bulldozer": "excavator",
        "loader": "excavator",
        "vehicle": "truck",
        "car": "truck",
        "bus": "truck",
        "lift": "scissor_lift",
        "boom": "boom_lift",
        "cherry_picker": "cherry_picker",
        "manift": "man_lift",
        "man_lift": "man_lift",
        "pallet_truck": "pallet_jack",
        # Equipment aliases
        "saw": "circular_saw",
        "welder": "welding_machine",
        "cable": "safety_barrier",
        "fence": "safety_barrier",
        "barrier": "safety_barrier",
        "railing": "guardrail",
        "handrail": "guardrail",
        "scaffolds": "scaffolding",
        "ladder": "ladder",
    }

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._names: dict[int, str] = {}
        self._available_classes: set[str] = set()

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize label by applying aliases and standard transformations."""
        normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
        return YoloDetector._ALIASES.get(normalized, normalized)

    @staticmethod
    def _is_ppe_related(label: str) -> bool:
        """Check if a label relates to PPE (helmet, vest) or machinery."""
        normalized = label.lower()
        ppe_keywords = {
            "helmet", "hard_hat", "hardhat", "vest", "safety", "ppe",
            "machinery", "excavator", "forklift", "truck", "crane", 
            "lift", "bulldozer", "loader", "vehicle", "equipment"
        }
        return any(keyword in normalized for keyword in ppe_keywords)

    def available_classes(self) -> list[str]:
        if self._model is None:
            self.load()
        return sorted(self._available_classes)

    def supports_any_class(self, labels: set[str]) -> bool:
        if self._model is None:
            self.load()
        normalized_targets = {self._normalize_label(label) for label in labels}
        return bool(self._available_classes.intersection(normalized_targets))

    def load(self) -> DetectionModelInfo:
        if self._model is not None:
            return DetectionModelInfo(model_path=self.model_path, classes_available=sorted(set(self._names.values())))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Add ultralytics to requirements.") from exc

        model = YOLO(self.model_path)
        names = model.names
        if isinstance(names, list):
            self._names = {idx: name for idx, name in enumerate(names)}
        else:
            self._names = dict(names)
        self._available_classes = {self._normalize_label(str(name)) for name in self._names.values()}
        self._model = model
        return DetectionModelInfo(model_path=self.model_path, classes_available=sorted(set(self._names.values())))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load()

        results = self._model.predict(frame_bgr, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls.item())
            raw_label = str(self._names.get(cls_id, cls_id))
            normalized_label = self._normalize_label(raw_label)
            
            # Keep detections if:
            # 1. They match SUPPORTED_CLASSES exactly, OR
            # 2. They're PPE/machinery-related (more flexible for different models)
            if normalized_label not in SUPPORTED_CLASSES and not self._is_ppe_related(raw_label):
                continue
            
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(label=normalized_label, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return detections


