from __future__ import annotations

import time
from typing import Any, Optional

from app.config import Settings, get_settings
from app.ingestion.storage import new_id
from app.memory.base import IdentityResolver, VectorStore
from app.memory.centroid_index import InMemoryCentroidIndex
from app.memory.identity_scorer import IdentityScorer
from app.memory.signature_builder import signature_from_stored
from app.schemas import ObjectMatch
from app.schemas.identity import ObjectSignature


class ClusterIdentityResolver(IdentityResolver):
    """
    Cluster-first identity with optional Phase 13 multi-signal scoring.

    Visual-only path (no signature): backward compatible with phases 0–12.
    Multi-signal path: OCR + semantic + brand/text conflict rules.
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
        self.scorer = IdentityScorer(self.settings)
        self.last_latencies_ms: dict[str, float] = {}

    def resolve(
        self,
        vector: list[float],
        class_name: str,
        class_id: int = 0,
        *,
        new_signature: Optional[ObjectSignature] = None,
        graph_store: Any = None,
    ) -> ObjectMatch:
        cfg = self.settings.memory
        known_t = cfg.effective_known_threshold()
        uncertain_t = float(cfg.uncertain_threshold)
        t0 = time.perf_counter()

        clusters, cluster_source, best_cluster_id, candidate_object_ids, hits = (
            self._retrieve_candidates(vector, class_name, cfg)
        )

        t2 = time.perf_counter()
        self.last_latencies_ms["cluster_lookup_ms"] = (t2 - t0) * 1000.0
        self.last_latencies_ms["cluster_source"] = cluster_source  # type: ignore[assignment]
        self.last_latencies_ms["identity_resolution_ms"] = 0.0

        candidate_scores: list[dict[str, Any]] = []
        best_score = 0.0
        best_object_id: Optional[str] = None
        for h in hits:
            score = float(h.get("score") or 0.0)
            payload = h.get("payload") or {}
            oid = payload.get("object_id")
            if oid:
                candidate_scores.append(
                    {
                        "object_id": oid,
                        "score": score,
                        "visual_score": score,
                        "cluster_id": payload.get("cluster_id"),
                    }
                )
            if oid and score > best_score:
                best_score = score
                best_object_id = oid
                if not best_cluster_id:
                    best_cluster_id = payload.get("cluster_id")

        candidate_scores.sort(key=lambda x: x["score"], reverse=True)

        use_multi = (
            bool(self.settings.identity.enable_multi_signal)
            and new_signature is not None
            and graph_store is not None
            and candidate_scores
        )

        if use_multi:
            t_score0 = time.perf_counter()
            match = self._resolve_multi_signal(
                new_signature=new_signature,
                graph_store=graph_store,
                candidate_scores=candidate_scores,
                best_cluster_id=best_cluster_id,
                cfg=cfg,
            )
            self.last_latencies_ms["identity_scoring_ms"] = (
                time.perf_counter() - t_score0
            ) * 1000.0
            self.last_latencies_ms["identity_resolution_ms"] = (
                self.last_latencies_ms.get("identity_scoring_ms", 0.0)
            )
            match.similarity = match.similarity or best_score
            if new_signature:
                match.object_signature = new_signature.model_dump(mode="json")
            return match

        # --- Visual-only decision bands (phases 0–12) ---
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

        return ObjectMatch(
            object_id=new_id("obj"),
            is_new=True,
            similarity=best_score,
            cluster_id=best_cluster_id,
            decision="NEW",
            candidate_object_id=best_object_id,
            candidate_scores=candidate_scores[:5],
        )

    def _retrieve_candidates(self, vector, class_name, cfg):
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

        self.last_latencies_ms["cluster_lookup_ms"] = (time.perf_counter() - t0) * 1000.0

        candidate_object_ids: list[str] = []
        best_cluster_id: Optional[str] = None
        for ch in clusters:
            payload = ch.get("payload") or {}
            if best_cluster_id is None:
                best_cluster_id = payload.get("cluster_id")
            for oid in payload.get("object_ids") or []:
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
        return clusters, cluster_source, best_cluster_id, candidate_object_ids, hits

    def _resolve_multi_signal(
        self,
        *,
        new_signature: ObjectSignature,
        graph_store: Any,
        candidate_scores: list[dict[str, Any]],
        best_cluster_id: Optional[str],
        cfg: Any,
    ) -> ObjectMatch:
        scored: list[tuple[dict[str, Any], Any]] = []
        for cand in candidate_scores[:10]:
            oid = cand["object_id"]
            visual = float(cand.get("visual_score") or cand.get("score") or 0.0)
            hist = graph_store.get_object_history(oid) if graph_store else {}
            obj = hist.get("object") or {}
            cand_sig = signature_from_stored(
                class_name=str(obj.get("class_name") or new_signature.class_name or ""),
                attributes=hist.get("attributes") if isinstance(hist.get("attributes"), dict) else obj.get("attributes"),
                signature=obj.get("object_signature"),
                embedding_ref=obj.get("representative_observation_id"),
            )
            obs_count = int(obj.get("observation_count") or 0)
            result = self.scorer.score(
                new_signature,
                cand_sig,
                visual_similarity=visual,
                candidate_object_id=oid,
                observation_count=obs_count,
            )
            enriched = {
                **cand,
                "overall_score": result.overall_score,
                "visual_score": result.visual_score,
                "text_score": result.text_score,
                "brand_score": result.brand_score,
                "semantic_score": result.semantic_score,
                "attribute_score": result.attribute_score,
                "decision": result.decision,
                "brand_conflict": result.brand_conflict,
                "reason_codes": result.reason_codes,
            }
            scored.append((enriched, result))

        if not scored:
            return ObjectMatch(
                object_id=new_id("obj"),
                is_new=True,
                similarity=0.0,
                cluster_id=best_cluster_id,
                decision="NEW",
                candidate_scores=candidate_scores[:5],
                reason_codes=["NO_CANDIDATES"],
                object_signature=new_signature.model_dump(mode="json"),
            )

        scored.sort(key=lambda x: x[1].overall_score, reverse=True)
        best_cand, best_result = scored[0]
        candidate_payload = [s[0] for s in scored[:5]]
        decision = best_result.decision
        reason_codes = list(best_result.reason_codes)

        if decision == "KNOWN":
            return ObjectMatch(
                object_id=best_result.candidate_object_id or best_cand["object_id"],
                is_new=False,
                similarity=best_result.overall_score,
                cluster_id=best_cluster_id,
                decision="KNOWN",
                candidate_object_id=best_result.candidate_object_id,
                candidate_scores=candidate_payload,
                identity_score=best_result.model_dump(mode="json"),
                reason_codes=reason_codes,
            )

        if decision == "UNCERTAIN":
            if cfg.uncertain_as_new:
                return ObjectMatch(
                    object_id=new_id("obj"),
                    is_new=True,
                    similarity=best_result.overall_score,
                    cluster_id=best_cluster_id,
                    decision="UNCERTAIN",
                    candidate_object_id=best_result.candidate_object_id,
                    candidate_scores=candidate_payload,
                    identity_score=best_result.model_dump(mode="json"),
                    reason_codes=reason_codes,
                )
            return ObjectMatch(
                object_id=best_result.candidate_object_id or best_cand["object_id"],
                is_new=False,
                similarity=best_result.overall_score,
                cluster_id=best_cluster_id,
                decision="KNOWN",
                candidate_object_id=best_result.candidate_object_id,
                candidate_scores=candidate_payload,
                identity_score=best_result.model_dump(mode="json"),
                reason_codes=reason_codes,
            )

        return ObjectMatch(
            object_id=new_id("obj"),
            is_new=True,
            similarity=best_result.overall_score,
            cluster_id=best_cluster_id,
            decision="NEW",
            candidate_object_id=best_result.candidate_object_id,
            candidate_scores=candidate_payload,
            identity_score=best_result.model_dump(mode="json"),
            reason_codes=reason_codes,
        )
