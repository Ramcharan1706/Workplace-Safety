"""CSV export helpers for analytics snapshots."""

from __future__ import annotations

import csv
from pathlib import Path


def export_summary_to_csv(summary: dict, target_path: str | Path) -> Path:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_workers_detected", summary.get("total_workers_detected", 0)])
        writer.writerow(["total_violations", summary.get("total_violations", 0)])
        writer.writerow(["compliance_rate", f"{float(summary.get('compliance_rate', 0.0)):.6f}"])

    return target
