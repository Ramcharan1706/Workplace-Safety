"""Before vs after analytics utilities."""

from __future__ import annotations


def compare_periods(before: dict, after: dict) -> dict:
    before_violations = int(before.get("total_violations", 0))
    after_violations = int(after.get("total_violations", 0))
    before_compliance = float(before.get("compliance_rate", 0.0))
    after_compliance = float(after.get("compliance_rate", 0.0))

    return {
        "violation_delta": after_violations - before_violations,
        "violation_improvement": before_violations - after_violations,
        "compliance_delta": after_compliance - before_compliance,
        "before": before,
        "after": after,
    }
