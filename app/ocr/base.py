from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

from pathlib import Path

ImageLike = Union[str, Path]


class OCRReader(ABC):
    """Optional text extraction from object crops."""

    @abstractmethod
    def extract_text(self, image: ImageLike) -> dict[str, Any]:
        """
        Return:
          text, tokens, confidence, regions
        """
        ...

    @property
    def available(self) -> bool:
        return True
