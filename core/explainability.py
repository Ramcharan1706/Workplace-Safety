"""Generate explainable, human-readable model decisions."""

from __future__ import annotations


def explain(status: str, reason: str, confidence: float, rule_id: str) -> str:
    pct = int(round(confidence * 100))
    return f"{status} - {reason} (Confidence: {pct}%, Rule: {rule_id})"
