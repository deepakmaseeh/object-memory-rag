from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.config import Settings, get_settings
from app.schemas.processing import ProcessingOptions, ProcessingStrength


class AIEnhancer:
    """Conservative ML-oriented enhancement — preserves text, logos, fine detail."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.preprocessing

    def enhance(
        self,
        image: Image.Image,
        options: ProcessingOptions,
        strength: ProcessingStrength = ProcessingStrength.AUTO,
    ) -> Image.Image:
        if not (
            options.enhance_for_ai
            or options.remove_noise
            or options.improve_resolution
        ):
            return image.convert("RGB")

        level = self._resolve_strength(strength)
        arr = np.array(image.convert("RGB"))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if options.remove_noise or options.enhance_for_ai:
            h = self.cfg.denoise_strength.get(level, 3)
            arr = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)

        if options.enhance_for_ai:
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip = self.cfg.clahe_clip.get(level, 1.5)
            tile = self.cfg.clahe_tile
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            l = clahe.apply(l)
            arr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            arr = self._unsharp(arr, amount=self.cfg.sharpen_amount.get(level, 0.3))

        if options.improve_resolution:
            factor = min(
                self.cfg.upscale_factor.get(level, 1.25),
                float(self.cfg.max_upscale_factor),
            )
            if factor > 1.01:
                h, w = arr.shape[:2]
                arr = cv2.resize(
                    arr,
                    (int(w * factor), int(h * factor)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _resolve_strength(self, strength: ProcessingStrength) -> str:
        if strength == ProcessingStrength.AUTO:
            return self.cfg.default_strength
        return strength.value

    @staticmethod
    def _unsharp(bgr: np.ndarray, amount: float = 0.3) -> np.ndarray:
        blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)
