from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import BlobStore, new_id
from app.schemas import BBox, Observation, SegmentedDetection, utc_now


class ObjectProcessor:
    """Convert raw detections into structured Observation records."""

    def __init__(
        self,
        blob_store: Optional[BlobStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.blob_store = blob_store or BlobStore(self.settings)

    def extract_crop(
        self,
        image: np.ndarray,
        bbox: BBox,
        mask: Optional[np.ndarray] = None,
        *,
        masked: bool = True,
    ) -> Image.Image:
        h, w = image.shape[:2]
        x1 = max(0, int(round(bbox.x1)))
        y1 = max(0, int(round(bbox.y1)))
        x2 = min(w, int(round(bbox.x2)))
        y2 = min(h, int(round(bbox.y2)))
        if x2 <= x1 or y2 <= y1:
            return Image.fromarray(image).convert("RGB")
        crop = image[y1:y2, x1:x2].copy()
        if mask is not None and masked:
            m = mask[y1:y2, x1:x2]
            if m.shape[:2] == crop.shape[:2]:
                crop = crop.copy()
                crop[~m] = 0
        return Image.fromarray(crop).convert("RGB")

    def extract_transparent_crop(
        self,
        image: np.ndarray,
        bbox: BBox,
        mask: np.ndarray,
    ) -> Image.Image:
        h, w = image.shape[:2]
        x1 = max(0, int(round(bbox.x1)))
        y1 = max(0, int(round(bbox.y1)))
        x2 = min(w, int(round(bbox.x2)))
        y2 = min(h, int(round(bbox.y2)))
        crop = image[y1:y2, x1:x2].copy()
        m = mask[y1:y2, x1:x2]
        alpha = (m.astype(np.uint8) * 255) if m.shape[:2] == crop.shape[:2] else np.full(crop.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([crop, alpha])
        return Image.fromarray(rgba, mode="RGBA")

    def to_observations(
        self,
        segmented: list[SegmentedDetection],
        image_id: str,
        scene_id: Optional[str] = None,
    ) -> list[Observation]:
        default_scene = scene_id or self.settings.default_scene.scene_id
        observations: list[Observation] = []
        for item in segmented:
            det = item.detection
            obs_id = (
                det.detection_id
                if det.detection_id.startswith("obs_")
                else new_id("obs")
            )
            observations.append(
                Observation(
                    observation_id=obs_id,
                    object_id=None,
                    image_id=image_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    timestamp=utc_now(),
                    scene_id=default_scene,
                    mask_path=item.mask_path,
                    crop_path=item.crop_path,
                    transparent_path=item.transparent_path,
                    crop_original_path=item.crop_original_path,
                    crop_enhanced_path=item.crop_enhanced_path,
                    attributes={},
                )
            )
        return observations
