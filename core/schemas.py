"""Typed contracts for detections, safety decisions, and frame outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "green"
    WARNING = "yellow"
    DANGER = "red"


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class PersonAssessment:
    person_id: str
    person_box: tuple[int, int, int, int]
    person_confidence: float
    has_helmet: bool
    has_vest: bool
    nearest_machinery_distance: float | None
    rule_triggered: str
    reason: str
    status: str
    score: int
    risk_level: RiskLevel
    confidence_context: float


@dataclass(slots=True)
class ViolationEvent:
    timestamp: datetime
    camera_id: str
    person_id: str
    severity: RiskLevel
    status: str
    rule_triggered: str
    reason: str
    score: int
    confidence: float


@dataclass(slots=True)
class FrameResult:
    camera_id: str
    timestamp: datetime
    frame_index: int
    fps: float
    detections: list[Detection] = field(default_factory=list)
    assessments: list[PersonAssessment] = field(default_factory=list)
    violations: list[ViolationEvent] = field(default_factory=list)
