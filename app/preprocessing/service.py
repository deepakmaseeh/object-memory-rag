from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import BlobStore
from app.preprocessing.ai_enhancer import AIEnhancer
from app.preprocessing.auction_pipeline import AuctionPipeline
from app.schemas import ImageRecord
from app.schemas.processing import (
    ImageDerivatives,
    ProcessingOptions,
    ProcessingStrength,
)


class PreprocessingService:
    def __init__(
        self,
        blob_store: Optional[BlobStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.blob_store = blob_store or BlobStore(self.settings)
        self.ai = AIEnhancer(self.settings)
        self.auction = AuctionPipeline(self.settings)

    def prepare(
        self,
        image: ImageRecord,
        options: ProcessingOptions,
        strength: ProcessingStrength = ProcessingStrength.AUTO,
    ) -> tuple[ImageDerivatives, float, float]:
        t0 = time.perf_counter()
        original = self.blob_store.load_image(image.original_path)
        ai_ms = 0.0
        auction_ms = 0.0

        derivatives = ImageDerivatives(
            original_path=str(image.original_path),
            processing_meta={
                "options": options.model_dump(),
                "strength": strength.value,
            },
        )

        ai_path: Optional[Path] = None
        if options.enhance_for_ai or options.remove_noise or options.improve_resolution:
            t_ai = time.perf_counter()
            enhanced = self.ai.enhance(original, options, strength)
            ai_path = self.blob_store.save_image_derivative(
                image.image_id,
                "ai",
                enhanced,
                meta={"kind": "ai_enhanced", "options": options.model_dump(mode="json")},
            )
            ai_ms = (time.perf_counter() - t_ai) * 1000.0
            derivatives.ai_enhanced_path = str(ai_path)

        if options.clean_for_auction:
            t_a = time.perf_counter()
            auction_img, auction_meta = self.auction.render(original, strength)
            auction_path = self.blob_store.save_image_derivative(
                image.image_id,
                "auction",
                auction_img,
                meta={"kind": "auction", **auction_meta},
            )
            auction_ms = (time.perf_counter() - t_a) * 1000.0
            derivatives.auction_path = str(auction_path)
            derivatives.processing_meta["auction"] = auction_meta

        derivatives.processing_meta["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        derivatives.processing_meta["ai_ms"] = ai_ms
        derivatives.processing_meta["auction_ms"] = auction_ms
        return derivatives, ai_ms, auction_ms

    def resolve_recognition_path(
        self,
        image: ImageRecord,
        derivatives: ImageDerivatives,
        source: str,
    ) -> Path:
        src = (source or "original").lower()
        if src == "ai_enhanced" and derivatives.ai_enhanced_path:
            return Path(derivatives.ai_enhanced_path)
        if src == "auction" and derivatives.auction_path:
            return Path(derivatives.auction_path)
        return Path(image.original_path)
