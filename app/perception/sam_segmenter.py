from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.perception.base import ImageLike, Segmenter


def _to_ndarray(image: ImageLike) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return np.array(Image.open(image).convert("RGB"))


class SAMSegmenter(Segmenter):
    """SAM 2 segmenter with bbox prompts (Ultralytics). Falls back to bbox mask if SAM unavailable."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._failed = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from ultralytics import SAM

            weights = self.settings.models.segmenter
            model_path = self.settings.resolve_path(self.settings.paths.models) / weights
            path = str(model_path) if model_path.exists() else weights
            self._model = SAM(path)
            return True
        except Exception:
            self._failed = True
            return False

    def segment(self, image: ImageLike, bbox: list[float]) -> np.ndarray:
        arr = _to_ndarray(image)
        h, w = arr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if self._load() and self._model is not None:
            try:
                results = self._model.predict(
                    source=arr,
                    bboxes=[bbox],
                    verbose=False,
                    device=self.settings.resolve_device(),
                )
                if results and results[0].masks is not None and len(results[0].masks.data):
                    mask = results[0].masks.data[0].cpu().numpy()
                    # Resize mask to image if needed
                    if mask.shape[0] != h or mask.shape[1] != w:
                        mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
                        mask_img = mask_img.resize((w, h), Image.NEAREST)
                        mask = np.array(mask_img) > 127
                    else:
                        mask = mask > 0.5
                    return mask.astype(bool)
            except Exception:
                pass

        # Fallback: filled bbox mask
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask
