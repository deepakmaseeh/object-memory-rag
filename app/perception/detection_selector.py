from __future__ import annotations

from typing import Literal, Optional

from app.schemas import BBox, Detection

PresentationStrategy = Literal["largest_area", "highest_confidence"]


def bbox_area(bbox: BBox) -> float:
    w = max(0.0, bbox.x2 - bbox.x1)
    h = max(0.0, bbox.y2 - bbox.y1)
    return w * h


def select_primary_detection(
    detections: list[Detection],
    *,
    strategy: str = "largest_area",
) -> Optional[Detection]:
    """Pick the detection used for presentation crops (auction / bg preview)."""
    if not detections:
        return None
    if strategy == "highest_confidence":
        return max(detections, key=lambda d: d.confidence)
    # default: largest_area — tie-break by confidence
    return max(detections, key=lambda d: (bbox_area(d.bbox), d.confidence))
