"""Degraded-mode unit tests without live services."""

from __future__ import annotations

from app.config import get_settings
from app.graph.local_store import LocalGraphStore
from app.health import HealthService
from app.memory.qdrant_store import QdrantVectorStore
from app.rag.service import OllamaClient, RAGService
from app.schemas import BBox, ImageRecord, Observation, utc_now


def test_local_qdrant_without_docker(tmp_path):
    settings = get_settings().model_copy(deep=True)
    settings.qdrant.prefer_local = True
    settings.qdrant.local_path = str(tmp_path / "qdrant")
    store = QdrantVectorStore(settings, force_local=True)
    store.ensure_collections()
    vec = [0.0] * settings.embedding.vector_size
    vec[0] = 1.0
    store.upsert_observation(
        "obs_x",
        vec,
        {"object_id": "obj_x", "class_name": "cup"},
    )
    hits = store.search_similar(vec, top_k=1, class_name="cup")
    assert hits
    assert hits[0]["payload"]["object_id"] == "obj_x"
    assert store.health() is True


def test_rag_without_ollama_uses_memory_fallback(tmp_path, monkeypatch):
    from app.graph.local_store import LocalGraphStore
    from app.schemas import MemoryObject

    settings = get_settings().model_copy(deep=True)
    graph = LocalGraphStore(settings, path=tmp_path / "g.json")
    graph.ensure_schema()
    graph.upsert_object(
        MemoryObject(
            object_id="obj_phone",
            class_id=67,
            class_name="cell phone",
            observation_count=1,
        )
    )
    obs = Observation(
        observation_id="obs1",
        image_id="img1",
        class_id=67,
        class_name="cell phone",
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
        confidence=0.9,
        timestamp=utc_now(),
        scene_id=settings.default_scene.scene_id,
    )
    graph.create_observation(obs, object_id="obj_phone", location_name="Desk")

    # Fake vector store that no-ops
    class EmptyVS:
        def health(self):
            return True

        def ensure_collections(self):
            return None

        def upsert_observation(self, *a, **k):
            return None

        def search_similar(self, *a, **k):
            return []

        def upsert_cluster(self, *a, **k):
            return None

        def search_clusters(self, *a, **k):
            return []

    class DownOllama(OllamaClient):
        def health(self):
            return False

        def model_status(self):
            return {"reachable": False, "models": [], "required": {}, "available": {}}

    rag = RAGService(EmptyVS(), graph, settings)
    rag.llm = DownOllama(settings)
    resp = rag.answer("Where did I last see my phone?")
    assert "obj_phone" in resp.answer or "cell phone" in resp.answer.lower() or "memory" in resp.answer.lower()
    assert resp.context


def test_health_check_shapes():
    report = HealthService(get_settings()).check(load_models=False)
    assert report.status in {"READY", "DEGRADED", "FAILED", "NOT_CONFIGURED"}
    assert report.components
    names = {c.name for c in report.components}
    assert "blob_store" in names
    assert "ollama" in names
