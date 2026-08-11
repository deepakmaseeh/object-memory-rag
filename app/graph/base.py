from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.schemas import ImageRecord, MemoryObject, Observation, Scene


class GraphStore(ABC):
    """Graph memory interface (Neo4j)."""

    @abstractmethod
    def ensure_schema(self) -> None:
        ...

    @abstractmethod
    def upsert_image(self, image: ImageRecord) -> None:
        ...

    @abstractmethod
    def upsert_object(self, obj: MemoryObject) -> None:
        ...

    @abstractmethod
    def create_observation(
        self,
        observation: Observation,
        object_id: str,
        scene_id: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> None:
        ...

    @abstractmethod
    def ensure_scene(self, scene: Scene) -> None:
        ...

    @abstractmethod
    def link_object_to_cluster(self, object_id: str, cluster_id: str) -> None:
        ...

    @abstractmethod
    def upsert_cluster_node(
        self,
        cluster_id: str,
        name: str,
        class_name: Optional[str] = None,
        object_count: int = 0,
    ) -> None:
        ...

    @abstractmethod
    def get_object(self, object_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    def get_object_history(self, object_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def latest_observation(self, object_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    def search_objects_by_class(self, class_name: str, limit: int = 20) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
