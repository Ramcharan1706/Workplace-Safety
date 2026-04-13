"""Enhanced rule engine with confidence-weighted soft rules."""

from __future__ import annotations

from dataclasses import dataclass

from core.config import (
    HELMET_CONFIDENCE_PASS,
    HELMET_CONFIDENCE_WARN,
    VEST_CONFIDENCE_PASS,
    VEST_CONFIDENCE_WARN,
)
from core.schemas import RiskLevel


@dataclass(slots=True)
class RuleDecision:
    rule_id: str
    status: str
    reason: str
    risk_level: RiskLevel
    details: str = ""


def evaluate_rules_enhanced(
    has_helmet: bool,
    has_vest: bool,
    near_machinery: bool,
    helmet_confidence: float | None = None,
    vest_confidence: float | None = None,
) -> RuleDecision:
    """Evaluate safety rules with confidence weighting."""
    
    if near_machinery:
        return RuleDecision(
            rule_id="R3_MACHINERY_DISTANCE",
            status="Dangerous",
            reason="Person too close to machinery",
            risk_level=RiskLevel.DANGER,
            details="Machinery proximity is highest-priority danger condition",
        )

    if not has_helmet:
        return RuleDecision(
            rule_id="R1_NO_HELMET",
            status="Unsafe",
            reason="Helmet not detected",
            risk_level=RiskLevel.DANGER,
            details="No helmet detection in any confidence range",
        )

    if helmet_confidence is not None and helmet_confidence < HELMET_CONFIDENCE_PASS:
        if helmet_confidence >= HELMET_CONFIDENCE_WARN:
            return RuleDecision(
                rule_id="R1_HELMET_LOW_CONFIDENCE",
                status="Warning",
                reason=f"Helmet detected with low confidence ({helmet_confidence:.2f})",
                risk_level=RiskLevel.WARNING,
                details=f"Helmet confidence {helmet_confidence:.2f} below pass threshold {HELMET_CONFIDENCE_PASS}",
            )

    if not has_vest:
        return RuleDecision(
            rule_id="R2_NO_VEST",
            status="Warning",
            reason="Safety vest not detected",
            risk_level=RiskLevel.WARNING,
            details="No vest detection in any confidence range",
        )

    if vest_confidence is not None and vest_confidence < VEST_CONFIDENCE_PASS:
        if vest_confidence >= VEST_CONFIDENCE_WARN:
            return RuleDecision(
                rule_id="R2_VEST_LOW_CONFIDENCE",
                status="Warning",
                reason=f"Vest detected with low confidence ({vest_confidence:.2f})",
                risk_level=RiskLevel.WARNING,
                details=f"Vest confidence {vest_confidence:.2f} below pass threshold {VEST_CONFIDENCE_PASS}",
            )

    helmet_conf_str = f" ({helmet_confidence:.2f})" if helmet_confidence else ""
    vest_conf_str = f" ({vest_confidence:.2f})" if vest_confidence else ""
    return RuleDecision(
        rule_id="R4_FULL_COMPLIANCE",
        status="Safe",
        reason=f"Helmet{helmet_conf_str} and vest{vest_conf_str} detected with safe machinery distance",
        risk_level=RiskLevel.SAFE,
        details="All safety conditions met with adequate confidence levels",
    )

