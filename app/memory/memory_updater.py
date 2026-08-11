from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

from app.clustering.engine import ClusterEngine
from app.config import Settings, get_settings
from app.embedding.base import Embedder
from app.embedding.clip_embedder import CLIPEmbedder
from app.graph.base import GraphStore
from app.graph.local_store import LocalGraphStore
from app.graph.neo4j_store import Neo4jGraphStore
from app.ingestion.image_ingestor import ImageIngestor
from app.ingestion.storage import BlobStore
from app.logging_utils import get_logger, new_request_id, setup_logging
from app.memory.base import IdentityResolver, VectorStore
from app.memory.centroid_index import InMemoryCentroidIndex
from app.memory.identity_resolver import ClusterIdentityResolver
from app.memory.qdrant_store import QdrantVectorStore
from app.perception.pipeline import PerceptionPipeline
from app.perception.vlm import ConditionalVLM
from app.schemas import (
    ImageRecord,
    MemoryObject,
    Observation,
    ObservationMatchResult,
    PipelineLatencies,
    ProcessImageResult,
    Scene,
    utc_now,
)

log = get_logger(__name__)


def create_graph_store(settings: Settings) -> GraphStore:
    backend = (settings.neo4j.backend or "auto").lower()
    if backend == "local":
        path = settings.resolve_path(settings.neo4j.local_path)
        log.info("Using LocalGraphStore at %s", path)
        return LocalGraphStore(settings, path=path)
    if backend == "neo4j":
        return Neo4jGraphStore(settings)

    # auto: prefer Neo4j when reachable
    try:
        neo = Neo4jGraphStore(settings)
        if neo.health():
            log.info("Using Neo4jGraphStore at %s", settings.neo4j.uri)
            return neo
        neo.close()
    except Exception as exc:
        log.warning("Neo4j unavailable (%s); using LocalGraphStore", exc)
    path = settings.resolve_path(settings.neo4j.local_path)
    return LocalGraphStore(settings, path=path)


class MemoryUpdater:
    """Identity resolution + optional VLM attributes + graph/vector updates."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: Embedder,
        identity_resolver: IdentityResolver,
        cluster_engine: Optional[ClusterEngine] = None,
        vlm: Optional[ConditionalVLM] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.identity_resolver = identity_resolver
        self.cluster_engine = cluster_engine
        self.vlm = vlm or ConditionalVLM(settings or get_settings())
        self.settings = settings or get_settings()
        self.last_match_results: list[ObservationMatchResult] = []
        self.last_latencies = PipelineLatencies()

    def process_observations(
        self,
        observations: list[Observation],
        image: ImageRecord,
        location_name: Optional[str] = None,
        force_vlm: bool = False,
    ) -> tuple[list[Observation], list[MemoryObject]]:
        scene_id = (
            (image.meta or {}).get("scene_id")
            or self.settings.default_scene.scene_id
        )
        t_graph0 = time.perf_counter()
        self.graph_store.upsert_image(image)
        self.graph_store.ensure_scene(Scene(scene_id=scene_id, name=scene_id))

        updated_obs: list[Observation] = []
        objects: list[MemoryObject] = []
        seen_classes: set[str] = set()
        match_results: list[ObservationMatchResult] = []
        emb_ms = 0.0
        cluster_ms = 0.0
        identity_ms = 0.0
        vlm_ms = 0.0

        for obs in observations:
            if not obs.crop_path or not Path(obs.crop_path).exists():
                continue

            t_e0 = time.perf_counter()
            vector = self.embedder.encode(obs.crop_path)
            emb_ms += (time.perf_counter() - t_e0) * 1000.0

            match = self.identity_resolver.resolve(
                vector, class_name=obs.class_name, class_id=obs.class_id
            )
            if isinstance(self.identity_resolver, ClusterIdentityResolver):
                cluster_ms += float(
                    self.identity_resolver.last_latencies_ms.get("cluster_lookup_ms", 0.0)
                )
                identity_ms += float(
                    self.identity_resolver.last_latencies_ms.get(
                        "identity_resolution_ms", 0.0
                    )
                )

            decision = getattr(match, "decision", None) or (
                "NEW" if match.is_new else "KNOWN"
            )
            obs.object_id = match.object_id
            obs.scene_id = scene_id

            # VLM attributes for NEW / UNCERTAIN (and forced)
            if self.vlm.should_invoke(
                is_new=match.is_new,
                similarity=match.similarity,
                decision=decision,
                force=force_vlm,
            ):
                t_v0 = time.perf_counter()
                attrs = self.vlm.describe_crop(obs.crop_path, class_name=obs.class_name)
                vlm_ms += (time.perf_counter() - t_v0) * 1000.0
                if attrs:
                    obs.attributes = {**obs.attributes, **attrs}
                    log.info(
                        "VLM attributes for obs=%s decision=%s model=%s keys=%s",
                        obs.observation_id,
                        decision,
                        self.vlm.model,
                        list(attrs.keys()),
                    )

            mem_obj = MemoryObject(
                object_id=match.object_id,
                class_id=obs.class_id,
                class_name=obs.class_name,
                created_at=utc_now(),
                last_seen=obs.timestamp,
                observation_count=0,
                cluster_id=match.cluster_id,
                attributes=dict(obs.attributes or {}),
            )
            self.graph_store.upsert_object(mem_obj)
            if obs.attributes and hasattr(self.graph_store, "set_object_attributes"):
                try:
                    self.graph_store.set_object_attributes(match.object_id, obs.attributes)
                except Exception as exc:
                    log.warning("attribute graph link failed: %s", exc)

            if match.cluster_id:
                self.graph_store.link_object_to_cluster(match.object_id, match.cluster_id)

            self.graph_store.create_observation(
                obs,
                object_id=match.object_id,
                scene_id=scene_id,
                location_name=location_name,
            )

            self.vector_store.upsert_observation(
                observation_id=obs.observation_id,
                vector=vector,
                payload={
                    "observation_id": obs.observation_id,
                    "object_id": match.object_id,
                    "class_name": obs.class_name,
                    "class_id": obs.class_id,
                    "image_id": obs.image_id,
                    "timestamp": obs.timestamp.isoformat(),
                    "cluster_id": match.cluster_id,
                    "confidence": obs.confidence,
                    "matched_existing_object": (not match.is_new),
                    "similarity": match.similarity,
                    "decision": decision,
                    "attributes": obs.attributes,
                    "location": location_name,
                },
            )

            # Persist embedding reference sidecar (lightweight)
            try:
                emb_dir = self.settings.resolve_path(self.settings.paths.embeddings)
                emb_dir.mkdir(parents=True, exist_ok=True)
                import json

                (emb_dir / f"{obs.observation_id}.json").write_text(
                    json.dumps(
                        {
                            "observation_id": obs.observation_id,
                            "object_id": match.object_id,
                            "class_name": obs.class_name,
                            "dim": len(vector),
                            "decision": decision,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

            match_results.append(
                ObservationMatchResult(
                    observation_id=obs.observation_id,
                    object_id=match.object_id,
                    class_name=obs.class_name,
                    confidence=obs.confidence,
                    matched_existing_object=(not match.is_new),
                    is_new=match.is_new,
                    decision=decision,
                    cluster_id=match.cluster_id,
                    similarity=match.similarity,
                    attributes=dict(obs.attributes or {}),
                    location=location_name,
                    scene_id=scene_id,
                    image_id=obs.image_id,
                    crop_path=obs.crop_path,
                    mask_path=obs.mask_path,
                    memory_saved=True,
                    embedding_stored=True,
                    graph_updated=True,
                    cluster_assigned=bool(match.cluster_id)
                    or decision in {"NEW", "UNCERTAIN"},
                    candidate_scores=list(getattr(match, "candidate_scores", []) or []),
                )
            )
            updated_obs.append(obs)
            objects.append(mem_obj)
            seen_classes.add(obs.class_name)

            log.info(
                "memory update image=%s obs=%s object=%s decision=%s sim=%.3f class=%s attrs=%d",
                image.image_id,
                obs.observation_id,
                match.object_id,
                decision,
                match.similarity,
                obs.class_name,
                len(obs.attributes or {}),
            )

        t_graph1 = time.perf_counter()
        if self.cluster_engine:
            for class_name in seen_classes:
                try:
                    self.cluster_engine.rebuild_for_class(class_name)
                except Exception as exc:
                    log.warning("cluster rebuild failed for %s: %s", class_name, exc)

        self.last_match_results = match_results
        self.last_latencies = PipelineLatencies(
            embedding_ms=emb_ms,
            cluster_lookup_ms=cluster_ms,
            identity_resolution_ms=identity_ms,
            neo4j_update_ms=(t_graph1 - t_graph0) * 1000.0,
            vlm_ms=vlm_ms,
        )
        return updated_obs, objects


class PipelineService:
    """End-to-end: ingest → perceive → embed → match → memory."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        setup_logging(self.settings.logging.level)
        self.settings.ensure_dirs()
        self.blob_store = BlobStore(self.settings)
        self.ingestor = ImageIngestor(self.blob_store, self.settings)
        self.vector_store = QdrantVectorStore(self.settings)
        self.graph_store = create_graph_store(self.settings)
        self.centroid_index = InMemoryCentroidIndex(self.settings)
        self.embedder = CLIPEmbedder(self.settings)
        self.identity = ClusterIdentityResolver(
            self.vector_store, self.settings, centroid_index=self.centroid_index
        )
        self.cluster_engine = ClusterEngine(
            self.vector_store,
            self.graph_store,
            self.settings,
            centroid_index=self.centroid_index,
        )
        self.vlm = ConditionalVLM(self.settings)
        self.perception = PerceptionPipeline(
            blob_store=self.blob_store, settings=self.settings
        )
        self.memory = MemoryUpdater(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedder=self.embedder,
            identity_resolver=self.identity,
            cluster_engine=self.cluster_engine,
            vlm=self.vlm,
            settings=self.settings,
        )
        self._initialized = False
        self.last_result: Optional[ProcessImageResult] = None

    def initialize(self) -> None:
        if self._initialized:
            return
        self.vector_store.ensure_collections()
        self.graph_store.ensure_schema()
        self._initialized = True
        log.info(
            "pipeline initialized qdrant_mode=%s graph=%s device=%s",
            getattr(self.vector_store, "mode", "unknown"),
            type(self.graph_store).__name__,
            self.settings.resolve_device(),
        )

    def process_image_path(
        self,
        path: Union[str, Path],
        scene_id: Optional[str] = None,
        location_name: Optional[str] = None,
        force_vlm: bool = False,
    ) -> ProcessImageResult:
        request_id = new_request_id()
        self.initialize()
        t0 = time.perf_counter()
        image = self.ingestor.ingest_path(path, scene_id=scene_id)
        result = self._run(
            image, location_name=location_name, force_vlm=force_vlm, request_id=request_id
        )
        result.latencies.total_ms = (time.perf_counter() - t0) * 1000.0
        self.last_result = result
        return result

    def process_image_bytes(
        self,
        data: bytes,
        filename: str = "upload.jpg",
        scene_id: Optional[str] = None,
        location_name: Optional[str] = None,
        force_vlm: bool = False,
    ) -> ProcessImageResult:
        request_id = new_request_id()
        self.initialize()
        t0 = time.perf_counter()
        image = self.ingestor.ingest_bytes(data, filename=filename, scene_id=scene_id)
        result = self._run(
            image, location_name=location_name, force_vlm=force_vlm, request_id=request_id
        )
        result.latencies.total_ms = (time.perf_counter() - t0) * 1000.0
        self.last_result = result
        return result

    def ingest_only(self, data: bytes, filename: str = "upload.jpg") -> ImageRecord:
        self.initialize()
        image = self.ingestor.ingest_bytes(data, filename=filename)
        self.graph_store.upsert_image(image)
        return image

    def _run(
        self,
        image: ImageRecord,
        location_name: Optional[str] = None,
        force_vlm: bool = False,
        request_id: str = "-",
    ) -> ProcessImageResult:
        t_perc = time.perf_counter()
        # Prefer timed perception if available
        if hasattr(self.perception, "process_image_path_timed"):
            observations, perc_lat = self.perception.process_image_path_timed(
                image.original_path, image.image_id
            )
        else:
            observations = self.perception.process_image_path(
                image.original_path, image.image_id
            )
            perc_lat = {
                "yolo_ms": 0.0,
                "sam_ms": 0.0,
                "perception_ms": (time.perf_counter() - t_perc) * 1000.0,
            }

        for obs in observations:
            if image.meta.get("scene_id"):
                obs.scene_id = image.meta["scene_id"]

        updated, objects = self.memory.process_observations(
            observations, image, location_name=location_name, force_vlm=force_vlm
        )
        mem_lat = self.memory.last_latencies
        latencies = PipelineLatencies(
            yolo_ms=float(perc_lat.get("yolo_ms", 0.0)),
            sam_ms=float(perc_lat.get("sam_ms", 0.0)),
            embedding_ms=mem_lat.embedding_ms,
            cluster_lookup_ms=mem_lat.cluster_lookup_ms,
            identity_resolution_ms=mem_lat.identity_resolution_ms,
            neo4j_update_ms=mem_lat.neo4j_update_ms,
            vlm_ms=mem_lat.vlm_ms,
            perception_ms=float(perc_lat.get("perception_ms", 0.0)),
            total_ms=0.0,
        )
        return ProcessImageResult(
            request_id=request_id,
            image_id=image.image_id,
            original_path=image.original_path,
            detection_count=len(updated),
            observations=updated,
            objects=objects,
            matches=self.memory.last_match_results,
            latencies=latencies,
            device=self.settings.resolve_device(),
            models={
                "detector": self.settings.models.detector,
                "segmenter": self.settings.models.segmenter,
                "embedder": self.settings.models.embedder,
                "rag": self.settings.ollama.rag_model,
                "vision": self.settings.ollama.vision_model,
            },
        )

    def close(self) -> None:
        self.graph_store.close()
