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
from app.memory.signature_builder import (
    build_object_signature,
    derive_product_signature,
    product_display_name,
)
from app.ocr import create_ocr_reader
from app.ocr.base import OCRReader
from app.perception.processing_context import ProcessingContext
from app.preprocessing.service import PreprocessingService
from app.schemas.processing import (
    ImageDerivatives,
    PrepareImageResult,
    ProcessingOptions,
    ProcessingStrength,
    RecognitionSource,
    RecognizeImageRequest,
)
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
        ocr: Optional[OCRReader] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.identity_resolver = identity_resolver
        self.cluster_engine = cluster_engine
        self.vlm = vlm or ConditionalVLM(settings or get_settings())
        self.ocr = ocr or create_ocr_reader(settings or get_settings())
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
        ocr_ms = 0.0
        cluster_ms = 0.0
        identity_ms = 0.0
        identity_scoring_ms = 0.0
        vlm_ms = 0.0
        vlm_verify_ms = 0.0

        for obs in observations:
            if not obs.crop_path or not Path(obs.crop_path).exists():
                continue

            t_ocr0 = time.perf_counter()
            ocr_result = self.ocr.extract_text(obs.crop_path)
            ocr_ms += (time.perf_counter() - t_ocr0) * 1000.0
            if ocr_result.get("text"):
                obs.attributes = {**obs.attributes, "ocr": ocr_result}
                obs.ocr_ref = obs.observation_id

            t_e0 = time.perf_counter()
            vector = self.embedder.encode(obs.crop_path)
            emb_ms += (time.perf_counter() - t_e0) * 1000.0
            obs.embedding_ref = obs.observation_id

            partial_sig = build_object_signature(
                class_name=obs.class_name,
                crop_path=obs.crop_path,
                bbox=obs.bbox.as_list(),
                ocr=ocr_result,
                embedding_ref=obs.observation_id,
            )

            match = self.identity_resolver.resolve(
                vector,
                class_name=obs.class_name,
                class_id=obs.class_id,
                new_signature=partial_sig if self.settings.identity.enable_multi_signal else None,
                graph_store=self.graph_store if self.settings.identity.enable_multi_signal else None,
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
                identity_scoring_ms += float(
                    self.identity_resolver.last_latencies_ms.get("identity_scoring_ms", 0.0)
                )

            decision = getattr(match, "decision", None) or (
                "NEW" if match.is_new else "KNOWN"
            )
            identity_path = "known_fast" if decision == "KNOWN" else "new_uncertain"

            # VLM attributes for NEW / UNCERTAIN only (not routine KNOWN path)
            vlm_attrs: dict[str, Any] = {}
            if self.vlm.should_invoke(
                is_new=match.is_new,
                similarity=match.similarity,
                decision=decision,
                force=force_vlm,
            ):
                t_v0 = time.perf_counter()
                vlm_attrs = self.vlm.describe_crop(obs.crop_path, class_name=obs.class_name)
                vlm_ms += (time.perf_counter() - t_v0) * 1000.0
                if vlm_attrs:
                    obs.attributes = {**obs.attributes, **vlm_attrs}

            full_sig = build_object_signature(
                class_name=obs.class_name,
                crop_path=obs.crop_path,
                bbox=obs.bbox.as_list(),
                ocr=ocr_result,
                vlm_attrs=vlm_attrs or obs.attributes,
                embedding_ref=obs.observation_id,
            )
            obs.object_signature = full_sig.model_dump(mode="json")

            # Re-score with full semantic signature when VLM ran and not already KNOWN
            if (
                self.settings.identity.enable_multi_signal
                and vlm_attrs
                and decision != "KNOWN"
            ):
                match = self.identity_resolver.resolve(
                    vector,
                    class_name=obs.class_name,
                    class_id=obs.class_id,
                    new_signature=full_sig,
                    graph_store=self.graph_store,
                )
                decision = match.decision
                identity_scoring_ms += float(
                    getattr(self.identity_resolver, "last_latencies_ms", {}).get(
                        "identity_scoring_ms", 0.0
                    )
                )

            # VLM verification for UNCERTAIN
            if self.vlm.should_verify(decision) and match.candidate_object_id:
                cand_hist = self.graph_store.get_object_history(match.candidate_object_id)
                cand_obs = (cand_hist.get("observations") or [{}])[0]
                cand_crop = cand_obs.get("crop_path")
                if cand_crop and Path(cand_crop).exists():
                    t_vv0 = time.perf_counter()
                    verify = self.vlm.verify_same_physical_object(
                        obs.crop_path,
                        cand_crop,
                        candidate_metadata=cand_hist.get("attributes")
                        or (cand_hist.get("object") or {}).get("attributes"),
                        ocr_text=ocr_result.get("text") or "",
                    )
                    vlm_verify_ms += (time.perf_counter() - t_vv0) * 1000.0
                    min_conf = float(self.settings.identity.vlm_verify_min_confidence)
                    if (
                        verify.same_physical_object is True
                        and verify.confidence >= min_conf
                    ):
                        match = match.model_copy(
                            update={
                                "object_id": match.candidate_object_id,
                                "is_new": False,
                                "decision": "KNOWN",
                                "reason_codes": list(match.reason_codes or [])
                                + ["VLM_VERIFY_SAME"],
                            }
                        )
                        decision = "KNOWN"
                    elif (
                        verify.same_physical_object is False
                        and verify.confidence >= min_conf
                    ):
                        from app.ingestion.storage import new_id as _new_id

                        match = match.model_copy(
                            update={
                                "object_id": _new_id("obj"),
                                "is_new": True,
                                "decision": "NEW",
                                "reason_codes": list(match.reason_codes or [])
                                + ["VLM_VERIFY_DIFFERENT"],
                            }
                        )
                        decision = "NEW"
                    else:
                        match = match.model_copy(
                            update={
                                "reason_codes": list(match.reason_codes or [])
                                + ["VLM_VERIFY_INCONCLUSIVE"],
                            }
                        )

            product = derive_product_signature(full_sig)
            product_id = product.product_signature_id if product else None
            product_label = product_display_name(product) if product else None
            if product and hasattr(self.graph_store, "upsert_product_signature"):
                self.graph_store.upsert_product_signature(product)

            obs.object_id = match.object_id
            obs.scene_id = scene_id
            obs.product_signature_id = product_id
            obs.identity_state = decision
            obs.identity_score = match.similarity

            mem_obj = MemoryObject(
                object_id=match.object_id,
                class_id=obs.class_id,
                class_name=obs.class_name,
                created_at=utc_now(),
                last_seen=obs.timestamp,
                observation_count=0,
                cluster_id=match.cluster_id,
                attributes=dict(obs.attributes or {}),
                product_signature_id=product_id,
                identity_state=decision,
                identity_confidence=match.similarity,
                representative_observation_id=obs.observation_id,
                object_signature=obs.object_signature,
            )
            self.graph_store.upsert_object(mem_obj)
            if product_id and hasattr(self.graph_store, "link_object_to_product"):
                self.graph_store.link_object_to_product(match.object_id, product_id)
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
                    "product_signature_id": product_id,
                    "identity_score": match.similarity,
                    "object_signature": obs.object_signature,
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
                    product_signature_id=product_id,
                    product_label=product_label,
                    identity_score=getattr(match, "identity_score", None),
                    reason_codes=list(getattr(match, "reason_codes", []) or []),
                    object_signature=obs.object_signature,
                    ocr_text=ocr_result.get("text") or None,
                    identity_path=identity_path,
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
            ocr_ms=ocr_ms,
            cluster_lookup_ms=cluster_ms,
            identity_resolution_ms=identity_ms,
            identity_scoring_ms=identity_scoring_ms,
            neo4j_update_ms=(t_graph1 - t_graph0) * 1000.0,
            vlm_ms=vlm_ms,
            vlm_verify_ms=vlm_verify_ms,
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
        self.preprocessing = PreprocessingService(self.blob_store, self.settings)
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
        self._prepared: dict[str, dict] = {}

    def _processing_state_path(self, image_id: str) -> Path:
        return self.settings.resolve_path(self.settings.paths.processed) / f"{image_id}.processing.json"

    def _save_processing_state(
        self,
        image: ImageRecord,
        derivatives: ImageDerivatives,
        options: ProcessingOptions,
        strength: ProcessingStrength,
    ) -> None:
        import json

        payload = {
            "image_id": image.image_id,
            "original_path": image.original_path,
            "width": image.width,
            "height": image.height,
            "derivatives": derivatives.model_dump(mode="json"),
            "options": options.model_dump(mode="json"),
            "strength": strength.value,
        }
        path = self._processing_state_path(image.image_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._prepared[image.image_id] = payload
        image.meta = {**(image.meta or {}), "processing": payload}
        self.graph_store.upsert_image(image)

    def _load_processing_state(self, image_id: str) -> Optional[dict]:
        if image_id in self._prepared:
            return self._prepared[image_id]
        path = self._processing_state_path(image_id)
        if path.exists():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            self._prepared[image_id] = data
            return data
        return None

    def _preview_urls(self, image_id: str, derivatives: ImageDerivatives) -> dict[str, str]:
        urls: dict[str, str] = {
            "original": f"/media/raw/{image_id}",
        }
        if derivatives.ai_enhanced_path:
            urls["ai_enhanced"] = f"/media/processed/{image_id}/ai"
        if derivatives.auction_path:
            urls["auction"] = f"/media/processed/{image_id}/auction"
        if derivatives.transparent_preview_path:
            urls["background_removed"] = f"/media/processed/{image_id}/transparent_preview"
        elif self.blob_store.get_image_derivative_path(image_id, "transparent_preview"):
            urls["background_removed"] = f"/media/processed/{image_id}/transparent_preview"
        return urls

    def prepare_image_bytes(
        self,
        data: bytes,
        filename: str = "upload.jpg",
        scene_id: Optional[str] = None,
        options: Optional[ProcessingOptions] = None,
        strength: ProcessingStrength = ProcessingStrength.AUTO,
    ) -> PrepareImageResult:
        import time

        self.initialize()
        t0 = time.perf_counter()
        opts = options or ProcessingOptions()
        image = self.ingestor.ingest_bytes(data, filename=filename, scene_id=scene_id)
        self.graph_store.upsert_image(image)

        derivatives, ai_ms, auction_ms = self.preprocessing.prepare(image, opts, strength)

        if opts.remove_background:
            ctx = ProcessingContext(remove_background=True)
            self.perception.process_image_path_timed(
                image.original_path, image.image_id, processing=ctx
            )
            if self.perception.last_transparent_preview:
                derivatives.transparent_preview_path = self.perception.last_transparent_preview

        self._save_processing_state(image, derivatives, opts, strength)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        return PrepareImageResult(
            image_id=image.image_id,
            width=image.width,
            height=image.height,
            derivatives=derivatives,
            preview_urls=self._preview_urls(image.image_id, derivatives),
            options=opts,
            strength=strength,
            preprocess_ms=preprocess_ms,
            auction_ms=auction_ms,
        )

    def recognize_image(
        self,
        request: RecognizeImageRequest,
    ) -> ProcessImageResult:
        import time

        self.initialize()
        t0 = time.perf_counter()
        state = self._load_processing_state(request.image_id)
        if not state:
            raise ValueError(f"No prepared image found for {request.image_id}")

        derivatives = ImageDerivatives.model_validate(state["derivatives"])
        options = ProcessingOptions.model_validate(state.get("options") or {})
        remove_bg = request.remove_background or options.remove_background

        image = ImageRecord(
            image_id=request.image_id,
            original_path=state["original_path"],
            width=int(state.get("width") or 0),
            height=int(state.get("height") or 0),
            meta={"processing": state, "scene_id": request.scene_id},
        )

        rec_path = self.preprocessing.resolve_recognition_path(
            image,
            derivatives,
            request.recognition_source.value,
        )

        request_id = new_request_id()
        result = self._run(
            image,
            location_name=request.location_name,
            force_vlm=request.force_vlm,
            request_id=request_id,
            recognition_path=rec_path,
            remove_background=remove_bg,
            recognition_source=request.recognition_source.value,
            derivatives=derivatives,
            processing_options=options,
        )
        result.latencies.total_ms = (time.perf_counter() - t0) * 1000.0
        if request.recognition_source == RecognitionSource.AUCTION:
            po = dict(result.processing_options or {})
            po["auction_recognition_warning"] = True
            result = result.model_copy(update={"processing_options": po})
        self.last_result = result
        return result

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
        options: Optional[ProcessingOptions] = None,
        strength: ProcessingStrength = ProcessingStrength.AUTO,
        recognition_source: RecognitionSource = RecognitionSource.ORIGINAL,
    ) -> ProcessImageResult:
        """Backward-compatible one-shot: prepare (optional) + recognize."""
        self.initialize()
        opts = options or ProcessingOptions()
        if opts.any_enabled():
            prep = self.prepare_image_bytes(
                data,
                filename=filename,
                scene_id=scene_id,
                options=opts,
                strength=strength,
            )
            return self.recognize_image(
                RecognizeImageRequest(
                    image_id=prep.image_id,
                    recognition_source=recognition_source,
                    location_name=location_name,
                    force_vlm=force_vlm,
                    remove_background=opts.remove_background,
                    scene_id=scene_id,
                )
            )
        request_id = new_request_id()
        t0 = time.perf_counter()
        image = self.ingestor.ingest_bytes(data, filename=filename, scene_id=scene_id)
        result = self._run(
            image,
            location_name=location_name,
            force_vlm=force_vlm,
            request_id=request_id,
            recognition_source=recognition_source.value,
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
        recognition_path: Optional[Path] = None,
        remove_background: bool = False,
        recognition_source: str = "original",
        derivatives: Optional[ImageDerivatives] = None,
        processing_options: Optional[ProcessingOptions] = None,
    ) -> ProcessImageResult:
        t_perc = time.perf_counter()
        perc_path = recognition_path or Path(image.original_path)
        ctx = ProcessingContext(remove_background=remove_background)
        if hasattr(self.perception, "process_image_path_timed"):
            observations, perc_lat = self.perception.process_image_path_timed(
                perc_path, image.image_id, processing=ctx
            )
        else:
            observations = self.perception.process_image_path(perc_path, image.image_id)
            perc_lat = {
                "yolo_ms": 0.0,
                "sam_ms": 0.0,
                "perception_ms": (time.perf_counter() - t_perc) * 1000.0,
            }

        if derivatives is None and self.perception.last_transparent_preview:
            derivatives = ImageDerivatives(
                original_path=str(image.original_path),
                transparent_preview_path=self.perception.last_transparent_preview,
            )

        for obs in observations:
            if image.meta.get("scene_id"):
                obs.scene_id = image.meta["scene_id"]

        updated, objects = self.memory.process_observations(
            observations, image, location_name=location_name, force_vlm=force_vlm
        )
        mem_lat = self.memory.last_latencies
        state = self._load_processing_state(image.image_id) or {}
        latencies = PipelineLatencies(
            yolo_ms=float(perc_lat.get("yolo_ms", 0.0)),
            sam_ms=float(perc_lat.get("sam_ms", 0.0)),
            embedding_ms=mem_lat.embedding_ms,
            ocr_ms=mem_lat.ocr_ms,
            preprocess_ms=float(state.get("derivatives", {}).get("processing_meta", {}).get("preprocess_ms", 0.0)),
            auction_ms=float(state.get("derivatives", {}).get("processing_meta", {}).get("auction_ms", 0.0)),
            cluster_lookup_ms=mem_lat.cluster_lookup_ms,
            identity_resolution_ms=mem_lat.identity_resolution_ms,
            identity_scoring_ms=mem_lat.identity_scoring_ms,
            neo4j_update_ms=mem_lat.neo4j_update_ms,
            vlm_ms=mem_lat.vlm_ms,
            vlm_verify_ms=mem_lat.vlm_verify_ms,
            perception_ms=float(perc_lat.get("perception_ms", 0.0)),
            total_ms=0.0,
        )
        deriv_dict = derivatives.model_dump(mode="json") if derivatives else None
        preview = self._preview_urls(image.image_id, derivatives) if derivatives else {}
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
            recognition_source=recognition_source,
            recognition_path=str(perc_path),
            processing_options=(processing_options.model_dump(mode="json") if processing_options else None),
            derivatives=deriv_dict,
            preview_urls=preview,
        )

    def close(self) -> None:
        self.graph_store.close()
