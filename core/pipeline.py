"""Frame processing orchestration for detection, rules, scoring, and event generation."""

from __future__ import annotations

from datetime import datetime, timezone

from core.association import has_associated_item, is_too_close_to_machinery, nearest_machinery_distance
from core.detector import YoloDetector
from core.explainability import explain
from core.rules import evaluate_rules
from core.scoring import compute_safety_score
from core.schemas import FrameResult, PersonAssessment, ViolationEvent


class SafetyPipeline:
    def __init__(self, detector: YoloDetector) -> None:
        self.detector = detector

    def process_frame(self, frame_bgr, camera_id: str, frame_index: int, fps: float) -> FrameResult:
        timestamp = datetime.now(timezone.utc)
        detections = self.detector.detect(frame_bgr)

        persons = [item for item in detections if item.label == "person"]
        helmets = [item for item in detections if item.label in {"helmet", "hardhat"}]
        vests = [item for item in detections if item.label in {"vest", "safety_vest"}]
        machinery = [item for item in detections if item.label in {"machinery", "forklift", "truck", "excavator"}]

        assessments: list[PersonAssessment] = []
        violations: list[ViolationEvent] = []
        helmet_check_enabled = self.detector.supports_any_class({"helmet", "hardhat", "hard_hat"})
        vest_check_enabled = self.detector.supports_any_class({"vest", "safety_vest", "safety-vest"})

        for idx, person in enumerate(persons, start=1):
            person_id = f"{camera_id}-p{idx}"
            has_helmet = has_associated_item(person.bbox, helmets) if helmet_check_enabled else True
            has_vest = has_associated_item(person.bbox, vests) if vest_check_enabled else True
            nearest_distance = nearest_machinery_distance(person.bbox, machinery)
            too_close = is_too_close_to_machinery(nearest_distance)

            decision = evaluate_rules(
                has_helmet=has_helmet,
                has_vest=has_vest,
                near_machinery=too_close,
                helmet_check_enabled=helmet_check_enabled,
                vest_check_enabled=vest_check_enabled,
            )
            score = compute_safety_score(
                has_helmet=has_helmet,
                has_vest=has_vest,
                safe_distance=not too_close,
                risk_level=decision.risk_level,
            )
            confidence_context = person.confidence
            reason = explain(
                status=decision.status,
                reason=decision.reason,
                confidence=confidence_context,
                rule_id=decision.rule_id,
            )

            assessment = PersonAssessment(
                person_id=person_id,
                person_box=person.bbox,
                person_confidence=person.confidence,
                has_helmet=has_helmet,
                has_vest=has_vest,
                nearest_machinery_distance=nearest_distance,
                rule_triggered=decision.rule_id,
                reason=reason,
                status=decision.status,
                score=score,
                risk_level=decision.risk_level,
                confidence_context=confidence_context,
            )
            assessments.append(assessment)

            if decision.status != "Safe":
                violations.append(
                    ViolationEvent(
                        timestamp=timestamp,
                        camera_id=camera_id,
                        person_id=person_id,
                        severity=decision.risk_level,
                        status=decision.status,
                        rule_triggered=decision.rule_id,
                        reason=decision.reason,
                        score=score,
                        confidence=confidence_context,
                    )
                )

        return FrameResult(
            camera_id=camera_id,
            timestamp=timestamp,
            frame_index=frame_index,
            fps=fps,
            detections=detections,
            assessments=assessments,
            violations=violations,
        )
