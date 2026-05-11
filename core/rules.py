"""Industrial-grade PPE safety rule engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.schemas import RiskLevel


# Minimum confidence required for valid PPE detection.
MIN_CONFIDENCE = 0.40


@dataclass(slots=True)
class RuleDecision:
    person_id: int = 0
    rule_id: str = ""
    status: str = ""
    reason: str = ""
    risk_level: RiskLevel = RiskLevel.SAFE
    details: str = ""


@dataclass(slots=True)
class PersonSafetyContext:
    person_id: int
    has_helmet: bool
    has_vest: bool
    near_machinery: bool
    helmet_conf: float = 0.0
    vest_conf: float = 0.0


def _evaluate_context(context: PersonSafetyContext) -> list[RuleDecision]:
    decisions: list[RuleDecision] = []

    # Highest-priority hazard first.
    if context.near_machinery:
        decisions.append(
            RuleDecision(
                person_id=context.person_id,
                rule_id="R1_MACHINERY_PROXIMITY",
                status="Danger",
                reason="Worker is too close to machinery",
                risk_level=RiskLevel.DANGER,
                details="Machinery proximity overrides PPE compliance",
            )
        )

    helmet_valid = context.has_helmet and context.helmet_conf >= MIN_CONFIDENCE
    if not helmet_valid:
        decisions.append(
            RuleDecision(
                person_id=context.person_id,
                rule_id="R2_NO_HELMET",
                status="Unsafe",
                reason="Helmet not detected",
                risk_level=RiskLevel.DANGER,
                details=f"Helmet confidence {context.helmet_conf:.2f} below {MIN_CONFIDENCE:.2f}",
            )
        )

    vest_valid = context.has_vest and context.vest_conf >= MIN_CONFIDENCE
    if not vest_valid:
        decisions.append(
            RuleDecision(
                person_id=context.person_id,
                rule_id="R3_NO_VEST",
                status="Warning",
                reason="Safety vest not detected",
                risk_level=RiskLevel.WARNING,
                details=f"Vest confidence {context.vest_conf:.2f} below {MIN_CONFIDENCE:.2f}",
            )
        )

    if helmet_valid and vest_valid and not context.near_machinery:
        decisions.append(
            RuleDecision(
                person_id=context.person_id,
                rule_id="R4_FULL_COMPLIANCE",
                status="Safe",
                reason="Worker follows all safety protocols",
                risk_level=RiskLevel.SAFE,
                details="All safety conditions met with adequate confidence levels",
            )
        )

    return decisions


def _evaluate_legacy_rules(
    has_helmet: bool,
    has_vest: bool,
    near_machinery: bool,
    helmet_check_enabled: bool = True,
    vest_check_enabled: bool = True,
) -> RuleDecision:
    if near_machinery:
        return RuleDecision(
            rule_id="R1_MACHINERY_PROXIMITY",
            status="Danger",
            reason="Person too close to machinery",
            risk_level=RiskLevel.DANGER,
            details="Machinery proximity overrides PPE compliance",
        )

    if helmet_check_enabled and not has_helmet:
        return RuleDecision(
            rule_id="R2_NO_HELMET",
            status="Unsafe",
            reason="Helmet not detected",
            risk_level=RiskLevel.DANGER,
            details="Helmet was expected but not found",
        )

    if vest_check_enabled and not has_vest:
        return RuleDecision(
            rule_id="R3_NO_VEST",
            status="Warning",
            reason="Safety vest not detected",
            risk_level=RiskLevel.WARNING,
            details="Vest was expected but not found",
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
            details="Fallback mode because the active model does not expose all PPE classes",
        )

    return RuleDecision(
        rule_id="R4_FULL_COMPLIANCE",
        status="Safe",
        reason="Helmet and vest detected with safe machinery distance",
        risk_level=RiskLevel.SAFE,
        details="All active checks passed",
    )


def evaluate_person_rules(contexts: Sequence[PersonSafetyContext]) -> list[RuleDecision]:
    decisions: list[RuleDecision] = []
    for context in contexts:
        decisions.extend(_evaluate_context(context))
    return decisions


def evaluate_rules(*args: Any, **kwargs: Any):
    """Evaluate safety rules in either legacy or industrial context mode.

    Legacy mode preserves the current pipeline API:
    evaluate_rules(has_helmet=..., has_vest=..., near_machinery=...)

    Industrial mode accepts a sequence of PersonSafetyContext objects and
    returns a list of RuleDecision objects.
    """
    if len(args) == 1 and not kwargs and isinstance(args[0], Sequence) and not isinstance(args[0], (str, bytes)):
        return evaluate_person_rules(args[0])

    if args:
        has_helmet = bool(args[0])
        has_vest = bool(args[1]) if len(args) > 1 else False
        near_machinery = bool(args[2]) if len(args) > 2 else False
        helmet_check_enabled = bool(args[3]) if len(args) > 3 else True
        vest_check_enabled = bool(args[4]) if len(args) > 4 else True
    else:
        has_helmet = bool(kwargs.get("has_helmet", False))
        has_vest = bool(kwargs.get("has_vest", False))
        near_machinery = bool(kwargs.get("near_machinery", False))
        helmet_check_enabled = bool(kwargs.get("helmet_check_enabled", True))
        vest_check_enabled = bool(kwargs.get("vest_check_enabled", True))

    return _evaluate_legacy_rules(
        has_helmet=has_helmet,
        has_vest=has_vest,
        near_machinery=near_machinery,
        helmet_check_enabled=helmet_check_enabled,
        vest_check_enabled=vest_check_enabled,
    )
