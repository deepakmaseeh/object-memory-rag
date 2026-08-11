from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import Settings, get_settings
from app.logging_utils import get_logger
from app.memory.base import VectorStore

log = get_logger(__name__)


def _point_id(key: str) -> str:
    return str(uuid5(NAMESPACE_URL, key))


def _probe_remote(host: str, port: int) -> bool:
    try:
        c = QdrantClient(host=host, port=port, check_compatibility=False, timeout=2.0)
        c.get_collections()
        return True
    except Exception:
        return False


class QdrantVectorStore(VectorStore):
    """
    Vector store over Qdrant.
    Uses remote Docker service when available; otherwise embedded on-disk Qdrant
    so second-loop memory can be verified without Docker.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        force_local: Optional[bool] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.obs_collection = self.settings.qdrant.collections.observations
        self.cluster_collection = self.settings.qdrant.collections.clusters
        self.vector_size = self.settings.embedding.vector_size
        self.mode = "remote"
        self.client = self._connect(force_local=force_local)

    def _connect(self, force_local: Optional[bool] = None) -> QdrantClient:
        prefer_local = (
            force_local
            if force_local is not None
            else self.settings.qdrant.prefer_local
        )
        local_path = self.settings.resolve_path(self.settings.qdrant.local_path)
        if prefer_local:
            local_path.mkdir(parents=True, exist_ok=True)
            self.mode = "local"
            log.info("Qdrant using local path store at %s", local_path)
            return QdrantClient(path=str(local_path))

        if _probe_remote(self.settings.qdrant.host, self.settings.qdrant.port):
            self.mode = "remote"
            log.info(
                "Qdrant using remote %s:%s",
                self.settings.qdrant.host,
                self.settings.qdrant.port,
            )
            return QdrantClient(
                host=self.settings.qdrant.host,
                port=self.settings.qdrant.port,
                check_compatibility=False,
            )

        # Fallback when Docker Qdrant is not running
        local_path.mkdir(parents=True, exist_ok=True)
        self.mode = "local"
        log.warning(
            "Remote Qdrant unreachable; falling back to local path store at %s",
            local_path,
        )
        return QdrantClient(path=str(local_path))

    def ensure_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.obs_collection not in existing:
            self.client.create_collection(
                collection_name=self.obs_collection,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
            for field in ("class_name", "object_id", "observation_id"):
                try:
                    self.client.create_payload_index(
                        collection_name=self.obs_collection,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
        if self.cluster_collection not in existing:
            self.client.create_collection(
                collection_name=self.cluster_collection,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=qm.Distance.COSINE,
                ),
            )
            try:
                self.client.create_payload_index(
                    collection_name=self.cluster_collection,
                    field_name="class_name",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert_observation(
        self,
        observation_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        full = {"observation_id": observation_id, **payload}
        self.client.upsert(
            collection_name=self.obs_collection,
            points=[
                qm.PointStruct(
                    id=_point_id(f"obs:{observation_id}"),
                    vector=vector,
                    payload=full,
                )
            ],
        )

    def search_similar(
        self,
        vector: list[float],
        top_k: int = 10,
        class_name: Optional[str] = None,
        object_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        must: list[qm.FieldCondition] = []
        if class_name:
            must.append(
                qm.FieldCondition(
                    key="class_name",
                    match=qm.MatchValue(value=class_name),
                )
            )
        if object_ids:
            must.append(
                qm.FieldCondition(
                    key="object_id",
                    match=qm.MatchAny(any=object_ids),
                )
            )
        query_filter = qm.Filter(must=must) if must else None
        response = self.client.query_points(
            collection_name=self.obs_collection,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "id": str(h.id),
                "score": float(h.score or 0.0),
                "payload": dict(h.payload or {}),
            }
            for h in (response.points or [])
        ]

    def upsert_cluster(
        self,
        cluster_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        full = {"cluster_id": cluster_id, **payload}
        self.client.upsert(
            collection_name=self.cluster_collection,
            points=[
                qm.PointStruct(
                    id=_point_id(f"cluster:{cluster_id}"),
                    vector=vector,
                    payload=full,
                )
            ],
        )

    def search_clusters(
        self,
        vector: list[float],
        top_k: int = 3,
        class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if class_name:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="class_name",
                        match=qm.MatchValue(value=class_name),
                    )
                ]
            )
        try:
            response = self.client.query_points(
                collection_name=self.cluster_collection,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = response.points or []
        except Exception:
            return []
        return [
            {
                "id": str(h.id),
                "score": float(h.score or 0.0),
                "payload": dict(h.payload or {}),
            }
            for h in hits
        ]

    def list_observations_for_class(
        self, class_name: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=self.obs_collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="class_name",
                        match=qm.MatchValue(value=class_name),
                    )
                ]
            ),
            limit=limit,
            with_vectors=True,
            with_payload=True,
        )
        results = []
        for p in points:
            vec = p.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), [])
            results.append(
                {
                    "id": str(p.id),
                    "vector": list(vec) if vec is not None else [],
                    "payload": dict(p.payload or {}),
                }
            )
        return results

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
