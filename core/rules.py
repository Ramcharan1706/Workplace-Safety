"""Deterministic safety rule engine."""

from __future__ import annotations

from dataclasses import dataclass

from core.schemas import RiskLevel


@dataclass(slots=True)
class RuleDecision:
    rule_id: str
    status: str
    reason: str
    risk_level: RiskLevel


def evaluate_rules(
    has_helmet: bool,
    has_vest: bool,
    near_machinery: bool,
    helmet_check_enabled: bool = True,
    vest_check_enabled: bool = True,
) -> RuleDecision:
    if near_machinery:
        return RuleDecision(
            rule_id="R3_MACHINERY_DISTANCE",
            status="Dangerous",
            reason="Person too close to machinery",
            risk_level=RiskLevel.DANGER,
        )

    if helmet_check_enabled and not has_helmet:
        return RuleDecision(
            rule_id="R1_NO_HELMET",
            status="Unsafe",
            reason="Helmet not detected",
            risk_level=RiskLevel.DANGER,
        )

    if vest_check_enabled and not has_vest:
        return RuleDecision(
            rule_id="R2_NO_VEST",
            status="Warning",
            reason="Safety vest not detected",
            risk_level=RiskLevel.WARNING,
        )

    if not helmet_check_enabled or not vest_check_enabled:
        limited_checks = []
        if not helmet_check_enabled:
            limited_checks.append("helmet")
        if not vest_check_enabled:
            limited_checks.append("vest")
        return RuleDecision(
            rule_id="R5_LIMITED_MODEL_CLASSES",
            status="Safe",
            reason=f"No active high-risk condition. PPE check unavailable for: {', '.join(limited_checks)}",
            risk_level=RiskLevel.SAFE,
        )

    return RuleDecision(
        rule_id="R4_FULL_COMPLIANCE",
        status="Safe",
        reason="Helmet and vest detected with safe machinery distance",
        risk_level=RiskLevel.SAFE,
    )
