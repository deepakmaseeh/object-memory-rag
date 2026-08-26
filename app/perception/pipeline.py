from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import BlobStore, new_id
from app.perception.base import Detector, Segmenter
from app.perception.detection_selector import select_primary_detection
from app.perception.object_processor import ObjectProcessor
from app.perception.processing_context import ProcessingContext
from app.perception.sam_segmenter import SAMSegmenter
from app.perception.yolo_detector import YOLODetector
from app.schemas import Detection, Observation, SegmentedDetection


class PerceptionPipeline:
    """Image → YOLO → SAM → crops/masks → Observations (object_id pending)."""

    def __init__(
        self,
        detector: Optional[Detector] = None,
        segmenter: Optional[Segmenter] = None,
        processor: Optional[ObjectProcessor] = None,
        blob_store: Optional[BlobStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.blob_store = blob_store or BlobStore(self.settings)
        self.detector = detector or YOLODetector(self.settings)
        self.segmenter = segmenter or SAMSegmenter(self.settings)
        self.processor = processor or ObjectProcessor(self.blob_store, self.settings)
        self.last_latencies: dict[str, float] = {}
        self.last_transparent_preview: Optional[str] = None

    def process_image_path(self, image_path: str | Path, image_id: str) -> list[Observation]:
        observations, _ = self.process_image_path_timed(image_path, image_id)
        return observations

    def process_image_path_timed(
        self,
        image_path: str | Path,
        image_id: str,
        *,
        processing: Optional[ProcessingContext] = None,
    ) -> tuple[list[Observation], dict[str, float]]:
        image = Image.open(image_path).convert("RGB")
        return self.process_image_timed(image, image_id, processing=processing)

    def process_image(self, image: Image.Image, image_id: str) -> list[Observation]:
        observations, _ = self.process_image_timed(image, image_id)
        return observations

    def process_image_timed(
        self,
        image: Image.Image,
        image_id: str,
        *,
        processing: Optional[ProcessingContext] = None,
    ) -> tuple[list[Observation], dict[str, float]]:
        ctx = processing or ProcessingContext()
        self.last_transparent_preview = None
        t0 = time.perf_counter()
        detections = self.detector.detect(image, image_id=image_id)
        t1 = time.perf_counter()
        yolo_ms = (t1 - t0) * 1000.0

        strategy = self.settings.preprocessing.presentation_detection_strategy
        primary_det = (
            select_primary_detection(detections, strategy=strategy)
            if ctx.remove_background and detections
            else None
        )

        segmented: list[SegmentedDetection] = []
        arr = np.array(image)
        sam_ms = 0.0
        for det in detections:
            ts = time.perf_counter()
            mask = self.segmenter.segment(arr, det.bbox.as_list())
            sam_ms += (time.perf_counter() - ts) * 1000.0
            observation_id = new_id("obs")
            mask_path = self.blob_store.save_mask(observation_id, mask)

            crop_original = self.processor.extract_crop(arr, det.bbox, mask, masked=False)
            crop_original_path = self.blob_store.save_crop(
                observation_id,
                crop_original,
                suffix="_original.jpg",
                meta={"kind": "crop_original"},
            )
            crop = self.processor.extract_crop(arr, det.bbox, mask, masked=True)
            crop_path = self.blob_store.save_crop(observation_id, crop)

            transparent_path = None
            if ctx.remove_background:
                transparent = self.processor.extract_transparent_crop(arr, det.bbox, mask)
                transparent_path = self.blob_store.save_crop(
                    observation_id,
                    transparent,
                    suffix="_transparent.png",
                    meta={"kind": "transparent"},
                )
                if primary_det is not None and det is primary_det:
                    preview_dest = self.blob_store.save_image_derivative(
                        image_id,
                        "transparent_preview",
                        transparent,
                        suffix=".png",
                        meta={
                            "observation_id": observation_id,
                            "selection": strategy,
                            "class_name": det.class_name,
                        },
                    )
                    ctx.transparent_preview_saved = True
                    self.last_transparent_preview = str(preview_dest)

            segmented.append(
                SegmentedDetection(
                    detection=Detection(
                        detection_id=observation_id,
                        image_id=image_id,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        timestamp=det.timestamp,
                    ),
                    mask_path=str(mask_path),
                    crop_path=str(crop_path),
                    transparent_path=str(transparent_path) if transparent_path else None,
                    crop_original_path=str(crop_original_path),
                )
            )

        observations = self.processor.to_observations(segmented, image_id=image_id)
        perception_ms = (time.perf_counter() - t0) * 1000.0
        latencies = {
            "yolo_ms": yolo_ms,
            "sam_ms": sam_ms,
            "perception_ms": perception_ms,
        }
        self.last_latencies = latencies
        return observations, latencies
