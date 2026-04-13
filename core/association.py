"""Associate PPE and machinery proximity with each detected person."""

from __future__ import annotations

import math

from core.config import MACHINERY_DANGER_DISTANCE_PX, PPE_ASSOCIATION_IOU, PPE_CONTAINMENT_RATIO
from core.schemas import Detection


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(a_area + b_area - inter_area)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _inside_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Return how much of inner lies within outer as a ratio of inner area."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / float(inner_area)


def nearest_machinery_distance(person_box: tuple[int, int, int, int], machinery: list[Detection]) -> float | None:
    if not machinery:
        return None
    px, py = _center(person_box)
    best_sq = None
    for item in machinery:
        mx, my = _center(item.bbox)
        distance_sq = (px - mx) ** 2 + (py - my) ** 2
        if best_sq is None or distance_sq < best_sq:
            best_sq = distance_sq
    return math.sqrt(best_sq) if best_sq is not None else None


def has_associated_item(person_box: tuple[int, int, int, int], items: list[Detection]) -> bool:
    """Check if person has associated PPE.
    
    Returns True if:
    - PPE item has >= 10% IoU with person box, OR
    - PPE item is >= 35% contained within person box (for items at edges like helmets)
    - This handles helmets at head-top and vests at shoulder edges properly.
    """
    for item in items:
        iou_score = _iou(person_box, item.bbox)
        inside_person_ratio = _inside_ratio(item.bbox, person_box)
        # Higher IoU threshold (0.10) or lower containment ratio (0.35 instead of 0.60)
        if iou_score >= PPE_ASSOCIATION_IOU or inside_person_ratio >= PPE_CONTAINMENT_RATIO:
            return True
    return False


def is_too_close_to_machinery(distance: float | None) -> bool:
    if distance is None:
        return False
    return distance < MACHINERY_DANGER_DISTANCE_PX

