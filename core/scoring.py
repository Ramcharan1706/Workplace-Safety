"""Dynamic safety scoring model."""

from __future__ import annotations

from core.config import (
    SCORE_HELMET,
    SCORE_VEST,
    SCORE_SAFE_DISTANCE,
    SCORE_PENALTY_DANGER,
    SCORE_PENALTY_WARNING,
)
from core.schemas import RiskLevel


def compute_safety_score(has_helmet: bool, has_vest: bool, safe_distance: bool, risk_level: RiskLevel) -> int:
    score = 0
    if has_helmet:
        score += SCORE_HELMET
    if has_vest:
        score += SCORE_VEST
    if safe_distance:
        score += SCORE_SAFE_DISTANCE

    if risk_level == RiskLevel.DANGER:
        score -= SCORE_PENALTY_DANGER
    elif risk_level == RiskLevel.WARNING:
        score -= SCORE_PENALTY_WARNING

    return max(0, min(100, score))

