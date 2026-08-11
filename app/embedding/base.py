from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


ImageLike = Union[str, Path, Image.Image, np.ndarray]


class Embedder(ABC):
    """Visual encoder interface — CLIP today, custom encoder later."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        ...

    @abstractmethod
    def encode(self, crop: ImageLike) -> list[float]:
        """Return a unit (or raw) embedding vector for an object crop."""

    def encode_batch(self, crops: list[ImageLike]) -> list[list[float]]:
        return [self.encode(c) for c in crops]
