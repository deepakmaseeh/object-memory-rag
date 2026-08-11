from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from app.schemas import Detection


ImageLike = Union[str, Path, Image.Image, np.ndarray]


class Detector(ABC):
    """Object detector interface — YOLO today, custom model later."""

    @abstractmethod
    def detect(self, image: ImageLike, image_id: str = "") -> list[Detection]:
        """Return detections for a single image."""


class Segmenter(ABC):
    """Instance segmenter interface — bbox-prompted mask generation."""

    @abstractmethod
    def segment(
        self,
        image: ImageLike,
        bbox: list[float],
    ) -> np.ndarray:
        """Return a boolean mask for the given bbox [x1,y1,x2,y2]."""
