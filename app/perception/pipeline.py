from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import BlobStore, new_id
from app.perception.base import Detector, Segmenter
from app.perception.object_processor import ObjectProcessor
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

    def process_image_path(self, image_path: str | Path, image_id: str) -> list[Observation]:
        observations, _ = self.process_image_path_timed(image_path, image_id)
        return observations

    def process_image_path_timed(
        self, image_path: str | Path, image_id: str
    ) -> tuple[list[Observation], dict[str, float]]:
        image = Image.open(image_path).convert("RGB")
        return self.process_image_timed(image, image_id)

    def process_image(self, image: Image.Image, image_id: str) -> list[Observation]:
        observations, _ = self.process_image_timed(image, image_id)
        return observations

    def process_image_timed(
        self, image: Image.Image, image_id: str
    ) -> tuple[list[Observation], dict[str, float]]:
        t0 = time.perf_counter()
        detections = self.detector.detect(image, image_id=image_id)
        t1 = time.perf_counter()
        yolo_ms = (t1 - t0) * 1000.0

        segmented: list[SegmentedDetection] = []
        arr = np.array(image)
        sam_ms = 0.0
        for det in detections:
            ts = time.perf_counter()
            mask = self.segmenter.segment(arr, det.bbox.as_list())
            sam_ms += (time.perf_counter() - ts) * 1000.0
            observation_id = new_id("obs")
            mask_path = self.blob_store.save_mask(observation_id, mask)
            crop = self.processor.extract_crop(arr, det.bbox, mask)
            crop_path = self.blob_store.save_crop(observation_id, crop)
            segmented.append(
                SegmentedDetection(
                    detection=Detection(
                        detection_id=observation_id,  # reuse as observation id
                        image_id=image_id,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        timestamp=det.timestamp,
                    ),
                    mask_path=str(mask_path),
                    crop_path=str(crop_path),
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
