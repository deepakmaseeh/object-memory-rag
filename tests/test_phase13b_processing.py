"""Phase 13B user-controlled image processing tests."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config.settings import Settings, get_settings
from app.ingestion.storage import BlobStore
from app.memory.memory_updater import PipelineService
from app.perception.detection_selector import bbox_area, select_primary_detection
from app.perception.pipeline import PerceptionPipeline
from app.perception.processing_context import ProcessingContext
from app.preprocessing.auction_pipeline import AuctionPipeline
from app.preprocessing.service import PreprocessingService
from app.schemas import BBox, Detection, Observation, utc_now
from app.schemas.processing import (
    ProcessingOptions,
    ProcessingStrength,
    RecognitionSource,
    RecognizeImageRequest,
)
from main import app


def _tmp_settings(tmp_path: Path) -> Settings:
    s = get_settings().model_copy(deep=True)
    for name in ("raw", "processed", "crops", "masks", "embeddings", "models", "storage", "objects"):
        setattr(s.paths, name, str(tmp_path / name))
    s.qdrant.prefer_local = True
    s.qdrant.local_path = str(tmp_path / "qdrant")
    s.neo4j.backend = "local"
    s.neo4j.local_path = str(tmp_path / "graph.json")
    return s


def _make_jpeg(size: tuple[int, int] = (128, 96), color: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _fake_observation(image_id: str) -> Observation:
    return Observation(
        observation_id="obs_test001",
        image_id=image_id,
        class_id=39,
        class_name="bottle",
        bbox=BBox(x1=10, y1=10, x2=60, y2=80),
        confidence=0.92,
        timestamp=utc_now(),
    )


class FakePerception:
    last_transparent_preview: Optional[str] = None

    def process_image_path_timed(self, image_path, image_id, *, processing=None):
        self.last_path = str(image_path)
        self.last_remove_bg = bool(processing and processing.remove_background)
        if self.last_remove_bg:
            self.last_transparent_preview = str(Path(image_path).parent / "fake_transparent.png")
        return [_fake_observation(image_id)], {
            "yolo_ms": 1.0,
            "sam_ms": 2.0,
            "perception_ms": 3.0,
        }


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    settings = _tmp_settings(tmp_path)
    svc = PipelineService(settings)
    svc.initialize()
    fake_perc = FakePerception()
    svc.perception = fake_perc  # type: ignore[assignment]
    svc.memory.process_observations = MagicMock(return_value=([], []))  # type: ignore[method-assign]
    svc.memory.last_match_results = []
    svc.memory.last_latencies = MagicMock(
        embedding_ms=0,
        ocr_ms=0,
        cluster_lookup_ms=0,
        identity_resolution_ms=0,
        identity_scoring_ms=0,
        neo4j_update_ms=0,
        vlm_ms=0,
        vlm_verify_ms=0,
    )
    return svc


def test_original_preserved(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions(enhance_for_ai=True))
    raw_path = Path(prep.derivatives.original_path)
    before = raw_path.read_bytes()
    from app.schemas import ImageRecord

    image = ImageRecord(
        image_id=prep.image_id,
        original_path=str(raw_path),
        width=prep.width,
        height=prep.height,
    )
    pipeline.preprocessing.prepare(image, ProcessingOptions(enhance_for_ai=True, remove_noise=True))
    assert raw_path.read_bytes() == before


def test_enhancement_optional_off(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions())
    assert prep.derivatives.ai_enhanced_path is None
    assert pipeline.blob_store.get_image_derivative_path(prep.image_id, "ai") is None


def test_enhancement_on(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(
        data, options=ProcessingOptions(enhance_for_ai=True), strength=ProcessingStrength.MEDIUM
    )
    assert prep.derivatives.ai_enhanced_path
    ai_path = Path(prep.derivatives.ai_enhanced_path)
    raw_path = Path(prep.derivatives.original_path)
    assert ai_path.exists()
    assert ai_path.read_bytes() != raw_path.read_bytes()


def test_auction_optional_off(pipeline):
    prep = pipeline.prepare_image_bytes(_make_jpeg(), options=ProcessingOptions())
    assert prep.derivatives.auction_path is None


def test_auction_optional_on(pipeline, monkeypatch):
    monkeypatch.setattr(
        "app.preprocessing.auction_pipeline.AuctionPipeline.render",
        lambda self, image, strength: (image, {"mock": True}),
    )
    prep = pipeline.prepare_image_bytes(
        _make_jpeg(), options=ProcessingOptions(clean_for_auction=True)
    )
    assert prep.derivatives.auction_path
    assert Path(prep.derivatives.auction_path).exists()


def test_background_removal_optional(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions())
    assert prep.derivatives.transparent_preview_path is None

    prep_bg = pipeline.prepare_image_bytes(
        data, filename="bg.jpg", options=ProcessingOptions(remove_background=True)
    )
    rec = pipeline.recognize_image(
        RecognizeImageRequest(image_id=prep_bg.image_id, remove_background=True)
    )
    assert pipeline.perception.last_remove_bg is True  # type: ignore[attr-defined]
    assert rec.detection_count >= 0


def test_recognition_source_original(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions(enhance_for_ai=True))
    pipeline.recognize_image(
        RecognizeImageRequest(image_id=prep.image_id, recognition_source=RecognitionSource.ORIGINAL)
    )
    assert Path(pipeline.perception.last_path) == Path(prep.derivatives.original_path)  # type: ignore[attr-defined]


def test_recognition_source_enhanced(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions(enhance_for_ai=True))
    pipeline.recognize_image(
        RecognizeImageRequest(
            image_id=prep.image_id, recognition_source=RecognitionSource.AI_ENHANCED
        )
    )
    assert Path(pipeline.perception.last_path) == Path(prep.derivatives.ai_enhanced_path)  # type: ignore[attr-defined]


def test_recognition_source_auction_warn(pipeline, monkeypatch):
    monkeypatch.setattr(
        "app.preprocessing.auction_pipeline.AuctionPipeline.render",
        lambda self, image, strength: (image, {}),
    )
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions(clean_for_auction=True))
    result = pipeline.recognize_image(
        RecognizeImageRequest(image_id=prep.image_id, recognition_source=RecognitionSource.AUCTION)
    )
    assert result.processing_options.get("auction_recognition_warning") is True
    assert Path(pipeline.perception.last_path) == Path(prep.derivatives.auction_path)  # type: ignore[attr-defined]


def test_all_options_off_default(pipeline):
    data = _make_jpeg()
    prep = pipeline.prepare_image_bytes(data, options=ProcessingOptions())
    assert prep.derivatives.ai_enhanced_path is None
    assert prep.derivatives.auction_path is None
    result = pipeline.recognize_image(RecognizeImageRequest(image_id=prep.image_id))
    assert result.recognition_source == "original"
    assert Path(pipeline.perception.last_path) == Path(prep.derivatives.original_path)  # type: ignore[attr-defined]


def test_multiple_options_simultaneous(pipeline, monkeypatch):
    monkeypatch.setattr(
        "app.preprocessing.auction_pipeline.AuctionPipeline.render",
        lambda self, image, strength: (image, {}),
    )
    prep = pipeline.prepare_image_bytes(
        _make_jpeg(),
        options=ProcessingOptions(
            enhance_for_ai=True,
            clean_for_auction=True,
            remove_noise=True,
        ),
    )
    assert prep.derivatives.ai_enhanced_path
    assert prep.derivatives.auction_path
    assert Path(prep.derivatives.ai_enhanced_path) != Path(prep.derivatives.auction_path)


def test_processing_disabled_backward_compat(pipeline):
    data = _make_jpeg()
    result = pipeline.process_image_bytes(data, location_name="Desk")
    assert result.image_id
    assert result.recognition_source == "original"


def test_blob_derivative_helpers(tmp_path):
    settings = _tmp_settings(tmp_path)
    store = BlobStore(settings)
    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    ai = store.save_image_derivative("img_test", "ai", img)
    assert ai.name == "img_test_ai.jpg"
    rgba = Image.new("RGBA", (32, 32), color=(10, 20, 30, 200))
    tp = store.save_image_derivative("img_test", "transparent_preview", rgba, suffix=".png")
    assert tp.suffix == ".png"
    obj = store.save_object_derivative("obj_test", "transparent", rgba)
    assert obj.parent.name == "obj_test"


def test_api_prepare_and_recognize_routes(tmp_path, monkeypatch):
    from app.api.routes import get_pipeline

    get_pipeline.cache_clear()
    settings = _tmp_settings(tmp_path)
    svc = PipelineService(settings)
    svc.initialize()
    svc.perception = FakePerception()  # type: ignore[assignment]
    svc.memory.process_observations = MagicMock(return_value=([], []))  # type: ignore[method-assign]
    svc.memory.last_match_results = []
    svc.memory.last_latencies = MagicMock(
        embedding_ms=0,
        ocr_ms=0,
        cluster_lookup_ms=0,
        identity_resolution_ms=0,
        identity_scoring_ms=0,
        neo4j_update_ms=0,
        vlm_ms=0,
        vlm_verify_ms=0,
    )
    monkeypatch.setattr("app.api.routes.get_pipeline", lambda: svc)

    client = TestClient(app)
    data = _make_jpeg()
    files = {"file": ("test.jpg", data, "image/jpeg")}
    prep_resp = client.post("/process/prepare", files=files, data={"enhance_for_ai": "true"})
    assert prep_resp.status_code == 200
    prep = prep_resp.json()
    assert prep["image_id"]
    assert prep["preview_urls"]["original"]

    rec_resp = client.post(
        "/process/recognize",
        json={"image_id": prep["image_id"], "recognition_source": "original"},
    )
    assert rec_resp.status_code == 200


def _det(class_name: str, bbox: list[float], confidence: float) -> Detection:
    return Detection(
        detection_id=f"det_{class_name}_{confidence}",
        image_id="img_test",
        class_id=1,
        class_name=class_name,
        confidence=confidence,
        bbox=BBox.from_list(bbox),
        timestamp=utc_now(),
    )


def test_select_primary_detection_largest_area():
    dial = _det("clock", [100, 100, 200, 200], 0.95)
    full = _det("clock", [50, 50, 350, 350], 0.70)
    picked = select_primary_detection([dial, full], strategy="largest_area")
    assert picked is full
    assert bbox_area(full.bbox) > bbox_area(dial.bbox)


def test_select_primary_detection_highest_confidence():
    dial = _det("clock", [100, 100, 200, 200], 0.95)
    full = _det("clock", [50, 50, 350, 350], 0.70)
    picked = select_primary_detection([dial, full], strategy="highest_confidence")
    assert picked is dial


def test_auction_uses_largest_detection(tmp_path):
    settings = _tmp_settings(tmp_path)
    pipe = AuctionPipeline(settings)
    dial = _det("clock", [100, 100, 200, 200], 0.95)
    full = _det("handbag", [20, 20, 380, 380], 0.60)

    class FakeDet:
        def detect(self, arr, image_id=""):
            return [dial, full]

    pipe._detector = FakeDet()  # type: ignore[assignment]
    det = pipe._best_detection(np.zeros((400, 400, 3), dtype=np.uint8))
    assert det is not None
    assert det["class_name"] == "handbag"
    assert det["selection"] == "largest_area"
    assert det["bbox_area"] > bbox_area(dial.bbox)


def test_transparent_preview_uses_primary_not_first(tmp_path):
    settings = _tmp_settings(tmp_path)
    store = BlobStore(settings)
    dial = _det("clock", [100, 100, 200, 200], 0.95)
    full = _det("clock", [10, 10, 390, 390], 0.70)

    class FakeDet:
        def detect(self, image, image_id=""):
            return [dial, full]

    class FakeSeg:
        def segment(self, arr, bbox):
            h, w = arr.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True
            return mask

    pipeline = PerceptionPipeline(
        detector=FakeDet(),  # type: ignore[arg-type]
        segmenter=FakeSeg(),  # type: ignore[arg-type]
        blob_store=store,
        settings=settings,
    )
    img = Image.new("RGB", (400, 400), color=(128, 128, 128))
    ctx = ProcessingContext(remove_background=True)
    pipeline.process_image_timed(img, "img_preview_test", processing=ctx)

    assert pipeline.last_transparent_preview
    preview = Image.open(pipeline.last_transparent_preview)
    dial_size = 200 - 100
    assert max(preview.size) > dial_size + 50

    meta_path = Path(pipeline.last_transparent_preview).with_suffix(
        Path(pipeline.last_transparent_preview).suffix + ".meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["selection"] == "largest_area"


def test_api_process_image_still_works(tmp_path, monkeypatch):
    from app.api.routes import get_pipeline

    get_pipeline.cache_clear()
    settings = _tmp_settings(tmp_path)
    svc = PipelineService(settings)
    svc.initialize()
    svc.perception = FakePerception()  # type: ignore[assignment]
    svc.memory.process_observations = MagicMock(return_value=([], []))  # type: ignore[method-assign]
    svc.memory.last_match_results = []
    svc.memory.last_latencies = MagicMock(
        embedding_ms=0,
        ocr_ms=0,
        cluster_lookup_ms=0,
        identity_resolution_ms=0,
        identity_scoring_ms=0,
        neo4j_update_ms=0,
        vlm_ms=0,
        vlm_verify_ms=0,
    )
    monkeypatch.setattr("app.api.routes.get_pipeline", lambda: svc)

    client = TestClient(app)
    resp = client.post(
        "/process/image",
        files={"file": ("test.jpg", _make_jpeg(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["image_id"]
