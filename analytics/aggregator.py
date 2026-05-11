"""Real-time analytics aggregation for safety metrics."""

from __future__ import annotations

from collections import Counter

from core.config import MACHINERY_DANGER_DISTANCE_PX
from core.schemas import FrameResult


class SafetyAnalytics:
    def __init__(self) -> None:
        self.total_workers_detected = 0
        self.total_violations = 0
        self.total_safe_assessments = 0
        self.total_limited_assessments = 0
        self.total_warning_assessments = 0
        self.total_helmet_checks = 0
        self.total_helmet_ok = 0
        self.total_vest_checks = 0
        self.total_vest_ok = 0
        self.total_machinery_exposure = 0
        self.total_machinery_danger = 0
        self.violation_counter: Counter[str] = Counter()
        self.score_trend: list[tuple[str, float]] = []
        self.compliance_trend: list[tuple[str, float]] = []

    def update(self, result: FrameResult) -> None:
        assessments = result.assessments
        if not assessments:
            return

        self.total_workers_detected += len(assessments)
        self.total_violations += len(result.violations)
        for item in assessments:
            if item.status == "Safe":
                self.total_safe_assessments += 1
            elif item.status == "Limited":
                self.total_limited_assessments += 1
            elif item.status == "Warning":
                self.total_warning_assessments += 1
        self.total_helmet_checks += len(assessments)
        self.total_helmet_ok += sum(1 for item in assessments if item.has_helmet)
        self.total_vest_checks += len(assessments)
        self.total_vest_ok += sum(1 for item in assessments if item.has_vest)
        machinery_distances = [item.nearest_machinery_distance for item in assessments if item.nearest_machinery_distance is not None]
        self.total_machinery_exposure += len(machinery_distances)
        self.total_machinery_danger += sum(1 for distance in machinery_distances if distance < MACHINERY_DANGER_DISTANCE_PX)
        for item in result.violations:
            self.violation_counter[item.rule_triggered] += 1

        safe_count = sum(1 for item in assessments if item.status == "Safe")
        compliance = safe_count / max(1, len(assessments))
        avg_score = sum(item.score for item in assessments) / len(assessments)

        ts = result.timestamp.isoformat()
        self.compliance_trend.append((ts, compliance))
        self.score_trend.append((ts, avg_score))

    def compliance_rate(self) -> float:
        if self.total_workers_detected == 0:
            return 0.0
        return 1.0 - (self.total_violations / self.total_workers_detected)

    def live_metrics(self) -> dict:
        helmet_rate = self.total_helmet_ok / max(1, self.total_helmet_checks)
        vest_rate = self.total_vest_ok / max(1, self.total_vest_checks)
        return {
            "total_workers_detected": self.total_workers_detected,
            "total_violations": self.total_violations,
            "compliance_rate": self.compliance_rate(),
            "helmet_compliance_rate": helmet_rate,
            "vest_compliance_rate": vest_rate,
            "machinery_exposure_count": self.total_machinery_exposure,
            "machinery_danger_count": self.total_machinery_danger,
        }

    def summary(self) -> dict:
        helmet_rate = self.total_helmet_ok / max(1, self.total_helmet_checks)
        vest_rate = self.total_vest_ok / max(1, self.total_vest_checks)
        if self.total_violations > 0:
            session_verdict = "Unsafe"
            session_reason = "Active red-level violations were detected"
        elif self.total_workers_detected > 0 and self.total_limited_assessments > 0 and self.total_safe_assessments == 0:
            session_verdict = "Safe"
            session_reason = "No active hazards were detected; PPE verification is limited by the current model"
        elif self.total_workers_detected > 0:
            session_verdict = "Safe"
            session_reason = "No active hazards were detected"
        else:
            session_verdict = "Inconclusive"
            session_reason = "No frames were processed"
        return {
            "total_workers_detected": self.total_workers_detected,
            "total_violations": self.total_violations,
            "total_safe_assessments": self.total_safe_assessments,
            "total_limited_assessments": self.total_limited_assessments,
            "total_warning_assessments": self.total_warning_assessments,
            "violation_distribution": dict(self.violation_counter),
            "compliance_rate": self.compliance_rate(),
            "helmet_compliance_rate": helmet_rate,
            "vest_compliance_rate": vest_rate,
            "machinery_exposure_count": self.total_machinery_exposure,
            "machinery_danger_count": self.total_machinery_danger,
            "session_verdict": session_verdict,
            "session_reason": session_reason,
            "score_trend": list(self.score_trend),
            "compliance_trend": list(self.compliance_trend),
        }
