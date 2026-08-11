"""Identity resolver tests using an in-memory fake vector store."""

from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.memory.base import VectorStore
from app.memory.identity_resolver import ClusterIdentityResolver
from app.schemas import ObjectMatch


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
        hits = []
        for item in self.obs:
            p = item["payload"]
            if class_name and p.get("class_name") != class_name:
                continue
            if object_ids and p.get("object_id") not in object_ids:
                continue
            # Cosine-ish: 1.0 if same first components pattern
            score = 0.99 if p.get("object_id") == "obj_known" else 0.5
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
        out = []
        for c in self.clusters:
            p = c["payload"]
            if class_name and p.get("class_name") != class_name:
                continue
            out.append({"id": c["cluster_id"], "score": 0.95, "payload": p})
        return out[:top_k]

    def health(self) -> bool:
        return True


def test_match_existing_object():
    settings = get_settings().model_copy(deep=True)
    settings.memory.known_threshold = 0.85
    settings.memory.match_threshold = 0.85
    settings.memory.uncertain_threshold = 0.70
    store = FakeVectorStore()
    store.clusters.append(
        {
            "cluster_id": "cluster_phone_0",
            "vector": [1.0] * 8,
            "payload": {
                "cluster_id": "cluster_phone_0",
                "class_name": "cell phone",
                "object_ids": ["obj_known"],
            },
        }
    )
    store.obs.append(
        {
            "observation_id": "obs_old",
            "vector": [1.0] * 8,
            "payload": {
                "observation_id": "obs_old",
                "object_id": "obj_known",
                "class_name": "cell phone",
            },
        }
    )
    resolver = ClusterIdentityResolver(store, settings)
    match = resolver.resolve([1.0] * 8, class_name="cell phone")
    assert isinstance(match, ObjectMatch)
    assert match.is_new is False
    assert match.object_id == "obj_known"
    assert match.similarity >= 0.85


def test_create_new_object_when_below_threshold():
    settings = get_settings().model_copy(deep=True)
    # Fake returns 0.99 for known object; thresholds above that force NEW
    settings.memory.known_threshold = 0.995
    settings.memory.match_threshold = 0.995
    settings.memory.uncertain_threshold = 0.994  # score 0.99 → NEW (< uncertain)
    store = FakeVectorStore()
    store.obs.append(
        {
            "observation_id": "obs_old",
            "vector": [1.0] * 8,
            "payload": {
                "observation_id": "obs_old",
                "object_id": "obj_known",
                "class_name": "cell phone",
            },
        }
    )
    resolver = ClusterIdentityResolver(store, settings)
    match = resolver.resolve([1.0] * 8, class_name="cell phone")
    assert match.is_new is True
    assert match.decision == "NEW"
    assert match.object_id.startswith("obj_")
