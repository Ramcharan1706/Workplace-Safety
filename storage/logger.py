"""Structured event logging for violations and frame-level outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from core.schemas import FrameResult


class EventLogger:
    _FIELDNAMES = [
        "timestamp",
        "camera_id",
        "person_id",
        "status",
        "severity",
        "rule_triggered",
        "reason",
        "score",
        "confidence",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self._FIELDNAMES)
                writer.writeheader()

    def log(self, result: FrameResult) -> None:
        if not result.violations:
            return
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self._FIELDNAMES)
            for event in result.violations:
                writer.writerow(
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "camera_id": event.camera_id,
                        "person_id": event.person_id,
                        "status": event.status,
                        "severity": event.severity.value,
                        "rule_triggered": event.rule_triggered,
                        "reason": event.reason,
                        "score": event.score,
                        "confidence": f"{event.confidence:.4f}",
                    }
                )
