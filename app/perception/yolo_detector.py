from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import new_id
from app.perception.base import Detector, ImageLike
from app.schemas import BBox, Detection, utc_now


def _to_ndarray(image: ImageLike) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    path = Path(image)
    return np.array(Image.open(path).convert("RGB"))


class YOLODetector(Detector):
    """Ultralytics YOLO11n detector behind the Detector interface."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._names: dict[int, str] = {}

    def _load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        weights = self.settings.models.detector
        model_path = self.settings.resolve_path(self.settings.paths.models) / weights
        # Ultralytics downloads weights if path doesn't exist; prefer models/ cache
        path = str(model_path) if model_path.exists() else weights
        self._model = YOLO(path)
        self._names = dict(self._model.names)

    def detect(self, image: ImageLike, image_id: str = "") -> list[Detection]:
        self._load()
        conf = self.settings.perception.conf_threshold
        iou = self.settings.perception.iou_threshold
        device = self.settings.resolve_device()
        results = self._model.predict(
            source=_to_ndarray(image),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            max_det=self.settings.perception.max_detections,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            class_id = int(clss[i])
            detections.append(
                Detection(
                    detection_id=new_id("det"),
                    image_id=image_id,
                    class_id=class_id,
                    class_name=str(self._names.get(class_id, f"class_{class_id}")),
                    confidence=float(confs[i]),
                    bbox=BBox.from_list(xyxy[i].tolist()),
                    timestamp=utc_now(),
                )
            )
        return detections
