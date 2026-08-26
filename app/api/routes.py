from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import get_settings
from app.health import HealthService
from app.memory.memory_updater import PipelineService
from app.rag.service import RAGService
from app.schemas import HealthStatus, MemoryResponse, ProcessImageResult
from app.schemas.processing import (
    PrepareImageResult,
    ProcessingOptions,
    ProcessingStrength,
    RecognitionSource,
    RecognizeImageRequest,
)

router = APIRouter()


@lru_cache
def get_pipeline() -> PipelineService:
    return PipelineService(get_settings())


class QueryBody(BaseModel):
    query: str


class RebuildBody(BaseModel):
    class_names: list[str]


def _graph_call(method: str, *args, **kwargs):
    pipeline = get_pipeline()
    pipeline.initialize()
    store = pipeline.graph_store
    fn = getattr(store, method, None)
    if not callable(fn):
        raise HTTPException(status_code=501, detail=f"Graph method not available: {method}")
    return fn(*args, **kwargs)


@router.get("/health", response_model=HealthStatus)
def health(load_models: bool = False) -> HealthStatus:
    return HealthService(get_settings()).check(load_models=load_models)


@router.get("/stats")
def stats() -> dict:
    pipeline = get_pipeline()
    pipeline.initialize()
    try:
        counts = _graph_call("stats")
    except HTTPException:
        counts = {
            "objects": 0,
            "observations": 0,
            "clusters": 0,
            "images": 0,
            "scenes": 0,
        }
    clusters_ram = pipeline.centroid_index.size()
    return {
        **counts,
        "clusters_ram": clusters_ram,
        "qdrant_mode": getattr(pipeline.vector_store, "mode", "unknown"),
        "graph_backend": type(pipeline.graph_store).__name__,
        "device": pipeline.settings.resolve_device(),
    }


@router.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        image = pipeline.ingest_only(data, filename=file.filename or "upload.jpg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "image_id": image.image_id,
        "original_path": image.original_path,
        "width": image.width,
        "height": image.height,
    }


@router.post("/process/prepare", response_model=PrepareImageResult)
async def prepare_image(
    file: UploadFile = File(...),
    scene_id: Optional[str] = Form(default=None),
    enhance_for_ai: bool = Form(default=False),
    remove_background: bool = Form(default=False),
    clean_for_auction: bool = Form(default=False),
    remove_noise: bool = Form(default=False),
    improve_resolution: bool = Form(default=False),
    strength: ProcessingStrength = Form(default=ProcessingStrength.AUTO),
) -> PrepareImageResult:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    options = ProcessingOptions(
        enhance_for_ai=enhance_for_ai,
        remove_background=remove_background,
        clean_for_auction=clean_for_auction,
        remove_noise=remove_noise,
        improve_resolution=improve_resolution,
    )
    pipeline = get_pipeline()
    try:
        return pipeline.prepare_image_bytes(
            data,
            filename=file.filename or "upload.jpg",
            scene_id=scene_id,
            options=options,
            strength=strength,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/recognize", response_model=ProcessImageResult)
def recognize_image(body: RecognizeImageRequest) -> ProcessImageResult:
    pipeline = get_pipeline()
    try:
        return pipeline.recognize_image(body)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/process/image", response_model=ProcessImageResult)
async def process_image(
    file: UploadFile = File(...),
    scene_id: Optional[str] = Form(default=None),
    location_name: Optional[str] = Form(default=None),
    force_vlm: bool = Form(default=False),
) -> ProcessImageResult:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    pipeline = get_pipeline()
    try:
        result = pipeline.process_image_bytes(
            data,
            filename=file.filename or "upload.jpg",
            scene_id=scene_id,
            location_name=location_name,
            force_vlm=force_vlm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/query", response_model=MemoryResponse)
def query_memory(body: QueryBody) -> MemoryResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        rag = RAGService(pipeline.vector_store, pipeline.graph_store, get_settings())
        return rag.answer(body.query)
    except Exception as exc:
        return MemoryResponse(
            query=body.query,
            answer=(
                "Memory services are unavailable, so I cannot look up objects yet. "
                f"Detail: {exc}"
            ),
            context=[],
            raw_context=None,
        )


@router.get("/objects")
def list_objects(limit: int = 200) -> dict:
    try:
        rows = _graph_call("list_objects", limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(rows), "objects": rows}


@router.get("/objects/{object_id}")
def get_object(object_id: str) -> dict:
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        history = pipeline.graph_store.get_object_history(object_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not history:
        raise HTTPException(status_code=404, detail="Object not found")
    return history


@router.get("/objects/{object_id}/observations")
def get_object_observations(object_id: str) -> dict:
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        history = pipeline.graph_store.get_object_history(object_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not history:
        raise HTTPException(status_code=404, detail="Object not found")
    return {
        "object_id": object_id,
        "observations": history.get("observations") or [],
    }


@router.get("/clusters")
def list_clusters() -> dict:
    pipeline = get_pipeline()
    pipeline.initialize()
    rows = []
    try:
        rows = _graph_call("list_clusters")
    except Exception:
        rows = []
    # Merge RAM index
    for c in pipeline.centroid_index._clusters.values():
        cid = c.get("cluster_id")
        if not any(r.get("cluster_id") == cid for r in rows):
            rows.append(
                {
                    "cluster_id": cid,
                    "name": cid,
                    "class_name": c.get("class_name"),
                    "object_count": len(c.get("object_ids") or []),
                    "object_ids": c.get("object_ids") or [],
                    "source": "ram_index",
                }
            )
    return {"count": len(rows), "clusters": rows}


@router.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: str) -> dict:
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        hit = None
        for c in pipeline.centroid_index._clusters.values():
            if c.get("cluster_id") == cluster_id:
                hit = {
                    "cluster_id": cluster_id,
                    "class_name": c.get("class_name"),
                    "object_ids": c.get("object_ids") or [],
                    "source": "ram_index",
                }
                break
        if hit is None:
            try:
                for row in _graph_call("list_clusters"):
                    if row.get("cluster_id") == cluster_id:
                        hit = {**row, "source": "graph"}
                        break
            except Exception:
                pass
        if hit is None:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return hit
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/graph")
def get_graph(object_id: Optional[str] = None, limit: int = 100) -> dict:
    try:
        return _graph_call("export_graph", object_id=object_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/graph/object/{object_id}")
def get_object_graph(object_id: str) -> dict:
    try:
        data = _graph_call("export_graph", object_id=object_id, limit=50)
        if not data.get("nodes"):
            raise HTTPException(status_code=404, detail="Object graph empty / not found")
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/clusters/rebuild")
def rebuild_clusters(body: RebuildBody) -> dict:
    pipeline = get_pipeline()
    try:
        pipeline.initialize()
        result = {}
        for name in body.class_names:
            clusters = pipeline.cluster_engine.rebuild_for_class(name)
            result[name] = [c.model_dump() for c in clusters]
        return {"clusters": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/media/processed/{image_id}/{kind}")
def get_processed_derivative(image_id: str, kind: str):
    pipeline = get_pipeline()
    path = pipeline.blob_store.get_image_derivative_path(image_id, kind)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Processed derivative not found")
    return FileResponse(path)


@router.get("/media/object/{object_id}/{kind}")
def get_object_derivative(object_id: str, kind: str):
    pipeline = get_pipeline()
    path = pipeline.blob_store.get_object_derivative_path(object_id, kind)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Object derivative not found")
    return FileResponse(path)


@router.get("/media/crop/{observation_id}")
def get_crop(observation_id: str):
    settings = get_settings()
    crops = settings.resolve_path(settings.paths.crops)
    path = crops / f"{observation_id}.jpg"
    if not path.exists():
        path = crops / f"{observation_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(path)


@router.get("/media/mask/{observation_id}")
def get_mask(observation_id: str):
    settings = get_settings()
    masks = settings.resolve_path(settings.paths.masks)
    path = masks / f"{observation_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(path)


@router.get("/media/raw/{image_id}")
def get_raw_image(image_id: str):
    settings = get_settings()
    raw = settings.resolve_path(settings.paths.raw)
    candidates = list(raw.glob(f"{image_id}.*"))
    candidates = [p for p in candidates if not p.name.endswith(".meta.json")]
    if not candidates:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidates[0])


@router.get("/system")
def system_info() -> dict:
    settings = get_settings()
    pipeline = get_pipeline()
    health = HealthService(settings).check(load_models=False)
    return {
        "app": settings.app.model_dump(),
        "models": settings.models.model_dump(),
        "ollama": {
            "base_url": settings.ollama.base_url,
            "rag_model": settings.ollama.rag_model,
            "fallback_model": settings.ollama.fallback_model,
            "vision_model": settings.ollama.vision_model,
        },
        "paths": settings.paths.model_dump(),
        "device": settings.resolve_device(),
        "qdrant_mode": getattr(pipeline.vector_store, "mode", "unknown"),
        "graph_backend": type(pipeline.graph_store).__name__,
        "health": health.model_dump(mode="json"),
    }
