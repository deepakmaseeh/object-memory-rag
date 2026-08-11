"""Unit tests for second-loop identity, cluster hot path, VLM gates (no GPU/DB)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.memory.base import VectorStore
from app.memory.centroid_index import InMemoryCentroidIndex
from app.memory.identity_resolver import ClusterIdentityResolver
from app.perception.vlm import ConditionalVLM


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.obs: list[dict[str, Any]] = []
        self.clusters: list[dict[str, Any]] = []

    def ensure_collections(self) -> None:
        return None

    def upsert_observation(self, observation_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.obs.append({"observation_id": observation_id, "vector": vector, "payload": payload})

    def search_similar(
        self,
        vector: list[float],
        top_k: int = 10,
        class_name: Optional[str] = None,
        object_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        hits = []
        for item in self.obs:
            p = item["payload"]
            if class_name and p.get("class_name") != class_name:
                continue
            if object_ids and p.get("object_id") not in object_ids:
                continue
            v = np.asarray(item["vector"], dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            score = float(np.dot(q, v))
            hits.append({"id": item["observation_id"], "score": score, "payload": p})
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]

    def upsert_cluster(self, cluster_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.clusters.append({"cluster_id": cluster_id, "vector": vector, "payload": payload})

    def search_clusters(
        self,
        vector: list[float],
        top_k: int = 3,
        class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return []

    def health(self) -> bool:
        return True


def _mem_settings(**kwargs):
    settings = get_settings().model_copy(deep=True)
    settings.memory.known_threshold = kwargs.get("known", 0.90)
    settings.memory.uncertain_threshold = kwargs.get("uncertain", 0.70)
    settings.memory.match_threshold = settings.memory.known_threshold
    settings.memory.uncertain_as_new = kwargs.get("uncertain_as_new", True)
    return settings


def test_second_loop_reuses_object_id():
    settings = _mem_settings(known=0.90)
    store = FakeVectorStore()
    index = InMemoryCentroidIndex(settings)
    resolver = ClusterIdentityResolver(store, settings, centroid_index=index)

    vec = [0.1, 0.2, 0.3, 0.4]
    first = resolver.resolve(vec, class_name="cell phone")
    assert first.is_new is True
    assert first.decision == "NEW"
    store.upsert_observation(
        "obs_1",
        vec,
        {
            "observation_id": "obs_1",
            "object_id": first.object_id,
            "class_name": "cell phone",
            "cluster_id": "cluster_cell phone_0",
        },
    )
    index.upsert(
        "cluster_cell phone_0",
        vec,
        class_name="cell phone",
        object_ids=[first.object_id],
    )

    second = resolver.resolve(vec, class_name="cell phone")
    assert second.is_new is False
    assert second.decision == "KNOWN"
    assert second.object_id == first.object_id
    assert second.similarity >= 0.90


def test_duplicate_prevention_when_similar():
    settings = _mem_settings(known=0.90)
    store = FakeVectorStore()
    resolver = ClusterIdentityResolver(store, settings)
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store.upsert_observation(
        "obs_old",
        base.tolist(),
        {"object_id": "obj_keep", "class_name": "cup"},
    )
    noisy = (base + 0.01).tolist()
    match = resolver.resolve(noisy, class_name="cup")
    assert match.is_new is False
    assert match.decision == "KNOWN"
    assert match.object_id == "obj_keep"


def test_uncertain_band_creates_new_by_default():
    settings = _mem_settings(known=0.95, uncertain=0.50, uncertain_as_new=True)
    store = FakeVectorStore()
    resolver = ClusterIdentityResolver(store, settings)
    # Force score ~0.75 by constructing orthogonal-ish vectors with controlled similarity
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # angle ~ cos^-1(0.8)
    b = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    store.upsert_observation(
        "obs_old",
        a.tolist(),
        {"object_id": "obj_maybe", "class_name": "bottle"},
    )
    match = resolver.resolve(b.tolist(), class_name="bottle")
    # cos([1,0],[0.8,0.6]) = 0.8 → UNCERTAIN band → NEW
    assert 0.50 <= match.similarity < 0.95
    assert match.decision == "UNCERTAIN"
    assert match.is_new is True
    assert match.object_id != "obj_maybe"
    assert match.candidate_object_id == "obj_maybe"


def test_low_similarity_is_new():
    settings = _mem_settings(known=0.90, uncertain=0.70)
    store = FakeVectorStore()
    resolver = ClusterIdentityResolver(store, settings)
    store.upsert_observation(
        "obs_old",
        [1.0, 0.0, 0.0, 0.0],
        {"object_id": "obj_other", "class_name": "bottle"},
    )
    match = resolver.resolve([0.0, 1.0, 0.0, 0.0], class_name="bottle")
    assert match.decision == "NEW"
    assert match.is_new is True


def test_centroid_index_hot_path():
    settings = get_settings().model_copy(deep=True)
    idx = InMemoryCentroidIndex(settings)
    idx.upsert("c1", [1, 0, 0, 0], class_name="phone", object_ids=["obj_a"])
    idx.upsert("c2", [0, 1, 0, 0], class_name="phone", object_ids=["obj_b"])
    hits = idx.search([0.99, 0.01, 0, 0], top_k=1, class_name="phone")
    assert len(hits) == 1
    assert hits[0]["payload"]["cluster_id"] == "c1"
    assert "obj_a" in hits[0]["payload"]["object_ids"]


def test_vlm_conditional_not_every_frame():
    settings = get_settings().model_copy(deep=True)
    settings.ollama.vision_enabled = True
    settings.ollama.vision_on_new_object = True
    settings.ollama.vision_on_uncertain = True
    settings.ollama.vision_low_confidence_threshold = 0.90
    vlm = ConditionalVLM(settings)

    assert vlm.should_invoke(is_new=False, similarity=0.97, decision="KNOWN") is False
    assert vlm.should_invoke(is_new=True, similarity=0.0, decision="NEW") is True
    assert vlm.should_invoke(is_new=True, similarity=0.75, decision="UNCERTAIN") is True
    assert vlm.should_invoke(is_new=False, similarity=0.99, force=True) is True
    settings.ollama.vision_enabled = False
    vlm2 = ConditionalVLM(settings)
    assert vlm2.should_invoke(is_new=True, similarity=0.0, decision="NEW") is False
    assert vlm2.should_invoke(is_new=True, similarity=0.0, decision="NEW", force=True) is True


def test_local_graph_second_observation_same_object(tmp_path):
    from app.graph.local_store import LocalGraphStore
    from app.schemas import BBox, MemoryObject, Observation, utc_now

    settings = get_settings().model_copy(deep=True)
    store = LocalGraphStore(settings, path=tmp_path / "graph.json")
    store.ensure_schema()
    obj = MemoryObject(
        object_id="obj_1",
        class_id=67,
        class_name="cell phone",
        observation_count=0,
    )
    store.upsert_object(obj)
    store.set_object_attributes("obj_1", {"color": "black", "material": "metal"})
    for i in range(2):
        obs = Observation(
            observation_id=f"obs_{i}",
            image_id=f"img_{i}",
            class_id=67,
            class_name="cell phone",
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
            confidence=0.9,
            timestamp=utc_now(),
        )
        store.create_observation(obs, object_id="obj_1", location_name="Desk")
    history = store.get_object_history("obj_1")
    assert len(history["observations"]) == 2
    assert history["object"]["observation_count"] == 2
    assert "Desk" in history["locations"]
    assert history["attributes"]["color"] == "black"
