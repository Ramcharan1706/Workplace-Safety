"""Frame annotation helpers for detections and person-level assessments."""

from __future__ import annotations

import cv2

from core.schemas import FrameResult, RiskLevel


RISK_COLORS = {
    RiskLevel.SAFE: (46, 204, 113),
    RiskLevel.WARNING: (0, 215, 255),
    RiskLevel.DANGER: (39, 39, 255),
}


def annotate_frame(frame_bgr, result: FrameResult):
    canvas = frame_bgr.copy()

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (96, 96, 96), 1)
        cv2.putText(
            canvas,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )

    for assessment in result.assessments:
        x1, y1, x2, y2 = assessment.person_box
        color = RISK_COLORS[assessment.risk_level]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        title = f"{assessment.person_id} | {assessment.status} | Score {assessment.score}"
        subtitle = assessment.reason
        cv2.putText(canvas, title, (x1, max(20, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"FPS: {result.fps:.2f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas
