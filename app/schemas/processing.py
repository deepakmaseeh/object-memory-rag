from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProcessingStrength(str, Enum):
    AUTO = "auto"
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"


class ProcessingOptions(BaseModel):
    enhance_for_ai: bool = False
    remove_background: bool = False
    clean_for_auction: bool = False
    remove_noise: bool = False
    improve_resolution: bool = False

    def any_enabled(self) -> bool:
        return any(
            (
                self.enhance_for_ai,
                self.remove_background,
                self.clean_for_auction,
                self.remove_noise,
                self.improve_resolution,
            )
        )


class RecognitionSource(str, Enum):
    ORIGINAL = "original"
    AI_ENHANCED = "ai_enhanced"
    AUCTION = "auction"


class ImageDerivatives(BaseModel):
    original_path: str
    ai_enhanced_path: Optional[str] = None
    auction_path: Optional[str] = None
    transparent_preview_path: Optional[str] = None
    processing_meta: dict[str, Any] = Field(default_factory=dict)


class PrepareImageResult(BaseModel):
    image_id: str
    width: int
    height: int
    derivatives: ImageDerivatives
    preview_urls: dict[str, str] = Field(default_factory=dict)
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    strength: ProcessingStrength = ProcessingStrength.AUTO
    preprocess_ms: float = 0.0
    auction_ms: float = 0.0


class RecognizeImageRequest(BaseModel):
    image_id: str
    recognition_source: RecognitionSource = RecognitionSource.ORIGINAL
    location_name: Optional[str] = None
    force_vlm: bool = False
    remove_background: bool = False
    scene_id: Optional[str] = None
