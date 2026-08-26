from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.schemas import Cluster, ObjectMatch


class VectorStore(ABC):
    """Vector memory interface (Qdrant)."""

    @abstractmethod
    def ensure_collections(self) -> None:
        ...

    @abstractmethod
    def upsert_observation(
        self,
        observation_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    def search_similar(
        self,
        vector: list[float],
        top_k: int = 10,
        class_name: Optional[str] = None,
        object_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def upsert_cluster(
        self,
        cluster_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    def search_clusters(
        self,
        vector: list[float],
        top_k: int = 3,
        class_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...


class IdentityResolver(ABC):
    """Second-loop matching: vector → cluster → candidate objects → identity."""

    @abstractmethod
    def resolve(
        self,
        vector: list[float],
        class_name: str,
        class_id: int = 0,
        *,
        new_signature: Any = None,
        graph_store: Any = None,
    ) -> ObjectMatch:
        ...
