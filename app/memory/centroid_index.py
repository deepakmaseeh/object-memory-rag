from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Settings, get_settings
from app.schemas import Cluster


class InMemoryCentroidIndex:
    """RAM-resident cluster centroid index for the second-loop hot path."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        # cluster_id -> {centroid: np.ndarray, class_name, object_ids, payload}
        self._clusters: dict[str, dict] = {}

    def clear(self) -> None:
        self._clusters.clear()

    def upsert(
        self,
        cluster_id: str,
        centroid: list[float] | np.ndarray,
        class_name: Optional[str] = None,
        object_ids: Optional[list[str]] = None,
        payload: Optional[dict] = None,
    ) -> None:
        vec = np.asarray(centroid, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec)) + 1e-12
        vec = vec / norm
        self._clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "centroid": vec,
            "class_name": class_name,
            "object_ids": list(object_ids or []),
            "payload": payload or {},
        }

    def upsert_cluster_model(self, cluster: Cluster) -> None:
        if not cluster.centroid:
            return
        self.upsert(
            cluster_id=cluster.cluster_id,
            centroid=cluster.centroid,
            class_name=cluster.class_name,
            object_ids=cluster.object_ids,
            payload={
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "class_name": cluster.class_name,
                "object_ids": cluster.object_ids,
                "object_count": cluster.object_count,
            },
        )

    def search(
        self,
        vector: list[float] | np.ndarray,
        top_k: Optional[int] = None,
        class_name: Optional[str] = None,
    ) -> list[dict]:
        top_k = top_k or self.settings.memory.cluster_search_top_k
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) + 1e-12)
        scored: list[tuple[float, dict]] = []
        for c in self._clusters.values():
            if class_name and c.get("class_name") and c["class_name"] != class_name:
                continue
            score = float(np.dot(q, c["centroid"]))
            payload = {
                "cluster_id": c["cluster_id"],
                "class_name": c.get("class_name"),
                "object_ids": c.get("object_ids") or [],
                **(c.get("payload") or {}),
            }
            scored.append((score, {"id": c["cluster_id"], "score": score, "payload": payload}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def size(self) -> int:
        return len(self._clusters)

    def load_from_clusters(self, clusters: list[Cluster]) -> None:
        for c in clusters:
            self.upsert_cluster_model(c)
