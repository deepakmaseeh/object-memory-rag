from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BBox(BaseModel):
    """Axis-aligned box as [x1, y1, x2, y2] in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, ...]) -> "BBox":
        return cls(x1=values[0], y1=values[1], x2=values[2], y2=values[3])


class InputSource(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class InputRecord(BaseModel):
    """Raw user input metadata (ingest-time)."""

    input_id: str
    source: InputSource = InputSource.IMAGE
    original_path: str
    content_type: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)


class ImageRecord(BaseModel):
    image_id: str
    original_path: str
    width: int
    height: int
    timestamp: datetime = Field(default_factory=utc_now)
    content_type: Optional[str] = "image/jpeg"
    meta: dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    """Single detector output — an observation candidate, not a persistent identity."""

    detection_id: str
    image_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    timestamp: datetime = Field(default_factory=utc_now)


class SegmentedDetection(BaseModel):
    """Detection after SAM mask + crop extraction."""

    detection: Detection
    mask_path: Optional[str] = None
    crop_path: Optional[str] = None


class Observation(BaseModel):
    """A stored observation of a persistent Object (or pending identity)."""

    observation_id: str
    object_id: Optional[str] = None
    image_id: str
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float
    timestamp: datetime = Field(default_factory=utc_now)
    scene_id: Optional[str] = None
    mask_path: Optional[str] = None
    crop_path: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class MemoryObject(BaseModel):
    """Persistent object identity (not a single detection)."""

    object_id: str
    class_id: int
    class_name: str
    created_at: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)
    observation_count: int = 0
    cluster_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Cluster(BaseModel):
    cluster_id: str
    name: str
    class_name: Optional[str] = None
    object_count: int = 0
    object_ids: list[str] = Field(default_factory=list)
    centroid: Optional[list[float]] = None


class Scene(BaseModel):
    scene_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class Attribute(BaseModel):
    name: str
    value: str
    confidence: Optional[float] = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class ObjectMatch(BaseModel):
    object_id: str
    is_new: bool
    similarity: float
    cluster_id: Optional[str] = None
    # NEW | KNOWN | UNCERTAIN
    decision: str = "NEW"
    candidate_object_id: Optional[str] = None
    candidate_scores: list[dict[str, Any]] = Field(default_factory=list)


class MemoryQuery(BaseModel):
    query: str
    top_k: int = 5
    class_name: Optional[str] = None


class MemoryContextItem(BaseModel):
    object_id: str
    class_name: str
    similarity: Optional[float] = None
    last_scene: Optional[str] = None
    last_location: Optional[str] = None
    last_seen: Optional[datetime] = None
    observation_count: int = 0
    summary: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    locations: list[str] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    query: str
    answer: str
    context: list[MemoryContextItem] = Field(default_factory=list)
    raw_context: Optional[str] = None


class ProcessImageResult(BaseModel):
    image_id: str
    original_path: str
    detection_count: int
    observations: list[Observation] = Field(default_factory=list)
    objects: list[MemoryObject] = Field(default_factory=list)
    matches: list["ObservationMatchResult"] = Field(default_factory=list)
    latencies: "PipelineLatencies" = Field(default_factory=lambda: PipelineLatencies())
    request_id: str = ""
    device: Optional[str] = None
    models: dict[str, Any] = Field(default_factory=dict)


class ObservationMatchResult(BaseModel):
    observation_id: str
    object_id: str
    class_name: str
    confidence: float
    matched_existing_object: bool
    is_new: bool
    # NEW | KNOWN | UNCERTAIN
    decision: str = "NEW"
    cluster_id: Optional[str] = None
    similarity: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = None
    scene_id: Optional[str] = None
    image_id: Optional[str] = None
    crop_path: Optional[str] = None
    mask_path: Optional[str] = None
    memory_saved: bool = True
    embedding_stored: bool = True
    graph_updated: bool = True
    cluster_assigned: bool = False
    candidate_scores: list[dict[str, Any]] = Field(default_factory=list)


class PipelineLatencies(BaseModel):
    yolo_ms: float = 0.0
    sam_ms: float = 0.0
    embedding_ms: float = 0.0
    cluster_lookup_ms: float = 0.0
    identity_resolution_ms: float = 0.0
    neo4j_update_ms: float = 0.0
    vlm_ms: float = 0.0
    perception_ms: float = 0.0
    total_ms: float = 0.0


class ComponentStatus(BaseModel):
    name: str
    status: str  # READY | DEGRADED | NOT_CONFIGURED | FAILED
    detail: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    status: str  # READY | DEGRADED | NOT_CONFIGURED | FAILED
    qdrant: str
    neo4j: str
    ollama: str
    details: dict[str, Any] = Field(default_factory=dict)
    components: list[ComponentStatus] = Field(default_factory=list)
