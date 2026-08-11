from __future__ import annotations

import time
from typing import Any, Optional

from app.config import Settings, get_settings
from app.ingestion.storage import new_id
from app.memory.base import IdentityResolver, VectorStore
from app.memory.centroid_index import InMemoryCentroidIndex
from app.schemas import ObjectMatch


class ClusterIdentityResolver(IdentityResolver):
    """
    Second-loop memory lookup with NEW / KNOWN / UNCERTAIN bands:

      similarity >= known_threshold      → KNOWN (reuse object)
      uncertain_threshold ≤ sim < known  → UNCERTAIN (default: create NEW, anti-merge bias)
      similarity < uncertain_threshold   → NEW

    Thresholds are config-driven for later calibration.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        settings: Optional[Settings] = None,
        centroid_index: Optional[InMemoryCentroidIndex] = None,
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.centroid_index = centroid_index or InMemoryCentroidIndex(self.settings)
        self.last_latencies_ms: dict[str, float] = {}

    def resolve(
        self,
        vector: list[float],
        class_name: str,
        class_id: int = 0,
    ) -> ObjectMatch:
        cfg = self.settings.memory
        known_t = cfg.effective_known_threshold()
        uncertain_t = float(cfg.uncertain_threshold)
        t0 = time.perf_counter()

        if self.centroid_index.size() > 0:
            clusters = self.centroid_index.search(
                vector,
                top_k=cfg.cluster_search_top_k,
                class_name=class_name,
            )
            cluster_source = "ram"
        else:
            clusters = self.vector_store.search_clusters(
                vector,
                top_k=cfg.cluster_search_top_k,
                class_name=class_name,
            )
            cluster_source = "qdrant"

        t1 = time.perf_counter()
        self.last_latencies_ms["cluster_lookup_ms"] = (t1 - t0) * 1000.0
        self.last_latencies_ms["cluster_source"] = cluster_source  # type: ignore[assignment]

        candidate_object_ids: list[str] = []
        best_cluster_id: Optional[str] = None
        candidate_scores: list[dict[str, Any]] = []
        for ch in clusters:
            payload = ch.get("payload") or {}
            if best_cluster_id is None:
                best_cluster_id = payload.get("cluster_id")
            oids = payload.get("object_ids") or []
            for oid in oids:
                if oid and oid not in candidate_object_ids:
                    candidate_object_ids.append(oid)

        if candidate_object_ids:
            hits = self.vector_store.search_similar(
                vector,
                top_k=cfg.object_search_top_k,
                class_name=class_name,
                object_ids=candidate_object_ids,
            )
        else:
            hits = self.vector_store.search_similar(
                vector,
                top_k=cfg.object_search_top_k,
                class_name=class_name,
            )

        t2 = time.perf_counter()
        self.last_latencies_ms["identity_resolution_ms"] = (t2 - t1) * 1000.0

        best_score = 0.0
        best_object_id: Optional[str] = None
        for h in hits:
            score = float(h.get("score") or 0.0)
            payload = h.get("payload") or {}
            oid = payload.get("object_id")
            if oid:
                candidate_scores.append(
                    {"object_id": oid, "score": score, "cluster_id": payload.get("cluster_id")}
                )
            if oid and score > best_score:
                best_score = score
                best_object_id = oid
                if not best_cluster_id:
                    best_cluster_id = payload.get("cluster_id")

        candidate_scores.sort(key=lambda x: x["score"], reverse=True)

        # --- Decision bands ---
        if best_object_id and best_score >= known_t:
            return ObjectMatch(
                object_id=best_object_id,
                is_new=False,
                similarity=best_score,
                cluster_id=best_cluster_id,
                decision="KNOWN",
                candidate_object_id=best_object_id,
                candidate_scores=candidate_scores[:5],
            )

        if best_object_id and best_score >= uncertain_t:
            # UNCERTAIN band: default create NEW to avoid corrupted merges
            if cfg.uncertain_as_new:
                return ObjectMatch(
                    object_id=new_id("obj"),
                    is_new=True,
                    similarity=best_score,
                    cluster_id=best_cluster_id,
                    decision="UNCERTAIN",
                    candidate_object_id=best_object_id,
                    candidate_scores=candidate_scores[:5],
                )
            return ObjectMatch(
                object_id=best_object_id,
                is_new=False,
                similarity=best_score,
                cluster_id=best_cluster_id,
                decision="KNOWN",
                candidate_object_id=best_object_id,
                candidate_scores=candidate_scores[:5],
            )

        # No confident match
        return ObjectMatch(
            object_id=new_id("obj"),
            is_new=True,
            similarity=best_score,
            cluster_id=best_cluster_id,
            decision="NEW",
            candidate_object_id=best_object_id,
            candidate_scores=candidate_scores[:5],
        )
