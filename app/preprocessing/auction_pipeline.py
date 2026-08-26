from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.perception.detection_selector import bbox_area, select_primary_detection
from app.perception.yolo_detector import YOLODetector
from app.perception.sam_segmenter import SAMSegmenter
from app.schemas import BBox
from app.schemas.processing import ProcessingStrength


class AuctionPipeline:
    """Human-presentation pipeline — separate from AI recognition optimization."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.preprocessing
        self._detector: Optional[YOLODetector] = None
        self._segmenter: Optional[SAMSegmenter] = None

    def render(
        self,
        image: Image.Image,
        strength: ProcessingStrength = ProcessingStrength.AUTO,
    ) -> tuple[Image.Image, dict[str, Any]]:
        level = strength.value if strength != ProcessingStrength.AUTO else self.cfg.default_strength
        arr = np.array(image.convert("RGB"))
        meta: dict[str, Any] = {"strength": level}

        det = self._best_detection(arr)
        if det is None:
            meta["fallback"] = "full_frame"
            return self._full_frame_auction(arr, level), meta

        bbox = det["bbox"]
        mask = self._segment_mask(arr, bbox)
        meta["class_name"] = det.get("class_name")
        meta["confidence"] = det.get("confidence")
        meta["selection"] = det.get("selection")
        meta["bbox_area"] = det.get("bbox_area")

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = arr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = arr[y1:y2, x1:x2].copy()
        m = mask[y1:y2, x1:x2] if mask is not None else np.ones(crop.shape[:2], dtype=bool)

        fg = crop[m] if m.any() else crop.reshape(-1, 3)
        if fg.size:
            mean = fg.mean(axis=0)
            scale = 128.0 / max(mean.mean(), 1.0)
            crop = np.clip(crop.astype(np.float32) * min(scale, 1.3), 0, 255).astype(np.uint8)

        rgba = np.dstack([crop, (m.astype(np.uint8) * 255)])
        pad = self.cfg.auction_padding_ratio
        ch, cw = rgba.shape[:2]
        canvas_size = int(max(ch, cw) * (1.0 + pad * 2))
        canvas = np.full((canvas_size, canvas_size, 4), 0, dtype=np.uint8)
        bg = self.cfg.auction_background_rgb
        canvas[:, :, :3] = np.array(bg, dtype=np.uint8)
        canvas[:, :, 3] = 255

        y_off = (canvas_size - ch) // 2
        x_off = (canvas_size - cw) // 2
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        region = canvas[y_off : y_off + ch, x_off : x_off + cw]
        region[:, :, :3] = (
            rgba[:, :, :3].astype(np.float32) * alpha
            + region[:, :, :3].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

        out = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_RGB2BGR)
        factor = min(
            self.cfg.upscale_factor.get(level, 1.25),
            float(self.cfg.max_upscale_factor),
        )
        if factor > 1.01:
            oh, ow = out.shape[:2]
            out = cv2.resize(out, (int(ow * factor), int(oh * factor)), interpolation=cv2.INTER_LANCZOS4)

        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), meta

    def _best_detection(self, arr: np.ndarray) -> Optional[dict[str, Any]]:
        if self._detector is None:
            self._detector = YOLODetector(self.settings)
        dets = self._detector.detect(arr, image_id="auction")
        if not dets:
            return None
        strategy = self.cfg.presentation_detection_strategy
        best = select_primary_detection(dets, strategy=strategy)
        if best is None:
            return None
        return {
            "bbox": best.bbox.as_list(),
            "class_name": best.class_name,
            "confidence": best.confidence,
            "selection": strategy,
            "bbox_area": bbox_area(best.bbox),
        }

    def _segment_mask(self, arr: np.ndarray, bbox: list[float]) -> Optional[np.ndarray]:
        if self._segmenter is None:
            self._segmenter = SAMSegmenter(self.settings)
        try:
            return self._segmenter.segment(arr, bbox)
        except Exception:
            h, w = arr.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
            return mask

    def _full_frame_auction(self, arr: np.ndarray, level: str) -> Image.Image:
        bg = np.array(self.cfg.auction_background_rgb, dtype=np.uint8)
        canvas = np.full_like(arr, bg)
        blend = cv2.addWeighted(arr, 0.92, canvas, 0.08, 0)
        return Image.fromarray(blend)
