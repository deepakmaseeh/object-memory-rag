from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config.settings import Settings, get_settings
from app.ingestion.storage import BlobStore, new_id
from app.retrieval.query_parser import QueryParser
from app.schemas import BBox, Detection, Observation, utc_now


def _tmp_settings(tmp_path: Path) -> Settings:
    base = get_settings()
    s = base.model_copy(deep=True)
    s.paths.raw = str(tmp_path / "raw")
    s.paths.processed = str(tmp_path / "processed")
    s.paths.crops = str(tmp_path / "crops")
    s.paths.masks = str(tmp_path / "masks")
    s.paths.embeddings = str(tmp_path / "embeddings")
    s.paths.models = str(tmp_path / "models")
    s.paths.storage = str(tmp_path / "storage")
    return s


def test_settings_load():
    s = get_settings()
    assert s.app.name == "object-memory"
    assert s.embedding.vector_size == 512
    assert s.memory.match_threshold == 0.90
    assert s.memory.known_threshold == 0.90
    assert s.memory.uncertain_threshold == 0.70


def test_bbox_roundtrip():
    b = BBox.from_list([1, 2, 3, 4])
    assert b.as_list() == [1, 2, 3, 4]


def test_new_id_prefix():
    assert new_id("obj").startswith("obj_")


def test_blob_store_raw_and_crop(tmp_path):
    settings = _tmp_settings(tmp_path)
    store = BlobStore(settings)
    img = Image.new("RGB", (64, 48), color=(255, 0, 0))
    src = tmp_path / "src.jpg"
    img.save(src)

    image_id, dest = store.save_raw(src, suffix=".jpg")
    assert dest.exists()
    assert (dest.with_suffix(dest.suffix + ".meta.json")).exists()

    crop = Image.new("RGB", (16, 16), color=(0, 255, 0))
    crop_path = store.save_crop("obs_test001", crop)
    assert crop_path.exists()

    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 10:30] = True
    mask_path = store.save_mask("obs_test001", mask)
    assert mask_path.exists()

    try:
        store.save_raw(src, image_id=image_id, suffix=".jpg")
        assert False, "expected FileExistsError"
    except FileExistsError:
        pass


def test_query_parser_phone():
    p = QueryParser()
    mq = p.parse("Where did I last see my black phone?")
    assert mq.class_name == "cell phone"


def test_query_parser_chair():
    p = QueryParser()
    mq = p.parse("Show me the chair near the desk")
    assert mq.class_name == "chair"


def test_observation_schema():
    obs = Observation(
        observation_id="obs_1",
        image_id="img_1",
        class_id=67,
        class_name="cell phone",
        bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        confidence=0.9,
        timestamp=utc_now(),
    )
    assert obs.object_id is None
    data = obs.model_dump()
    assert data["class_name"] == "cell phone"


def test_detection_not_object():
    det = Detection(
        detection_id="det_1",
        image_id="img_1",
        class_id=0,
        class_name="person",
        confidence=0.99,
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
    )
    assert not hasattr(det, "observation_count")
