from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.graph.local_store import LocalGraphStore
from app.graph.neo4j_store import Neo4jGraphStore
from app.ingestion.storage import BlobStore
from app.memory.qdrant_store import QdrantVectorStore
from app.rag.service import OllamaClient
from app.schemas import ComponentStatus, HealthStatus


def _status(ok: bool, optional: bool = False) -> str:
    if ok:
        return "READY"
    return "NOT_CONFIGURED" if optional else "FAILED"


class HealthService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def check(self, load_models: bool = False) -> HealthStatus:
        components: list[ComponentStatus] = []
        details: dict[str, Any] = {}

        # Python / torch / GPU
        details["python_version"] = sys.version.split()[0]
        torch_status = "NOT_CONFIGURED"
        cuda = False
        try:
            import torch

            details["torch_version"] = torch.__version__
            cuda = bool(torch.cuda.is_available())
            details["cuda_available"] = cuda
            details["cuda_version"] = getattr(torch.version, "cuda", None)
            if cuda:
                details["gpu_name"] = torch.cuda.get_device_name(0)
                details["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                )
            torch_status = "READY" if cuda else "DEGRADED"
            components.append(
                ComponentStatus(
                    name="pytorch",
                    status=torch_status,
                    detail=torch.__version__,
                    meta={
                        "cuda": cuda,
                        "device_config": self.settings.models.device,
                        "resolved_device": self.settings.resolve_device(),
                    },
                )
            )
        except Exception as exc:
            components.append(
                ComponentStatus(name="pytorch", status="FAILED", detail=str(exc))
            )

        # Blob store
        try:
            blob = BlobStore(self.settings)
            raw = blob.raw_dir
            raw.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(raw).free
            components.append(
                ComponentStatus(
                    name="blob_store",
                    status="READY",
                    detail=str(raw),
                    meta={"free_bytes": free},
                )
            )
        except Exception as exc:
            components.append(
                ComponentStatus(name="blob_store", status="FAILED", detail=str(exc))
            )

        # Qdrant
        q_status = "FAILED"
        try:
            if self.settings.qdrant.prefer_local:
                path = self.settings.resolve_path(self.settings.qdrant.local_path)
                path.mkdir(parents=True, exist_ok=True)
                q_status = "READY"
                components.append(
                    ComponentStatus(
                        name="qdrant",
                        status=q_status,
                        detail=f"local:{path}",
                    )
                )
            else:
                vs = QdrantVectorStore(self.settings)
                ok = vs.health()
                q_status = "READY" if ok else "FAILED"
                components.append(
                    ComponentStatus(
                        name="qdrant",
                        status=q_status,
                        detail=getattr(vs, "mode", "unknown"),
                    )
                )
        except Exception as exc:
            components.append(
                ComponentStatus(name="qdrant", status="FAILED", detail=str(exc))
            )

        # Graph / Neo4j
        n_status = "FAILED"
        try:
            if (self.settings.neo4j.backend or "auto").lower() == "local":
                gs = LocalGraphStore(self.settings)
                n_status = "READY" if gs.health() else "FAILED"
                backend = "local"
            else:
                neo = Neo4jGraphStore(self.settings)
                if neo.health():
                    n_status = "READY"
                    backend = "neo4j"
                else:
                    # auto-local fallback counts as DEGRADED but operational
                    gs = LocalGraphStore(self.settings)
                    n_status = "DEGRADED" if gs.health() else "FAILED"
                    backend = "local_fallback"
                neo.close()
            components.append(
                ComponentStatus(name="graph", status=n_status, detail=backend)
            )
        except Exception as exc:
            try:
                gs = LocalGraphStore(self.settings)
                n_status = "DEGRADED" if gs.health() else "FAILED"
                components.append(
                    ComponentStatus(
                        name="graph",
                        status=n_status,
                        detail=f"local_fallback after error: {exc}",
                    )
                )
            except Exception as exc2:
                components.append(
                    ComponentStatus(name="graph", status="FAILED", detail=str(exc2))
                )

        # Ollama + models
        ollama = OllamaClient(self.settings)
        model_info = ollama.model_status()
        o_ok = model_info.get("reachable", False)
        o_status = "READY" if o_ok else "FAILED"
        components.append(
            ComponentStatus(
                name="ollama",
                status=o_status,
                detail=self.settings.ollama.base_url,
                meta=model_info,
            )
        )

        # Optional model load probes
        if load_models:
            components.extend(self._probe_models())

        # Overall
        statuses = {c.name: c.status for c in components}
        core_ok = statuses.get("qdrant") == "READY" and statuses.get("graph") in {
            "READY",
            "DEGRADED",
        }
        if core_ok and o_status == "READY" and statuses.get("pytorch") in {
            "READY",
            "DEGRADED",
        }:
            overall = "READY" if statuses.get("pytorch") == "READY" else "DEGRADED"
        elif core_ok:
            overall = "DEGRADED"
        else:
            overall = "FAILED"

        details["docker_installed"] = shutil.which("docker") is not None
        details["resolved_device"] = self.settings.resolve_device()

        return HealthStatus(
            status=overall,
            qdrant=q_status,
            neo4j=n_status,
            ollama=o_status,
            details=details,
            components=components,
        )

    def _probe_models(self) -> list[ComponentStatus]:
        out: list[ComponentStatus] = []
        try:
            from app.perception.yolo_detector import YOLODetector

            det = YOLODetector(self.settings)
            det._load()
            out.append(ComponentStatus(name="yolo", status="READY", detail=self.settings.models.detector))
        except Exception as exc:
            out.append(ComponentStatus(name="yolo", status="FAILED", detail=str(exc)))
        try:
            from app.perception.sam_segmenter import SAMSegmenter

            seg = SAMSegmenter(self.settings)
            ok = seg._load()
            out.append(
                ComponentStatus(
                    name="sam2",
                    status="READY" if ok else "DEGRADED",
                    detail=self.settings.models.segmenter
                    + (" (bbox fallback active)" if not ok else ""),
                )
            )
        except Exception as exc:
            out.append(ComponentStatus(name="sam2", status="DEGRADED", detail=str(exc)))
        try:
            from app.embedding.clip_embedder import CLIPEmbedder

            emb = CLIPEmbedder(self.settings)
            emb._load()
            out.append(
                ComponentStatus(
                    name="openclip",
                    status="READY",
                    detail=self.settings.models.embedder,
                )
            )
        except Exception as exc:
            out.append(ComponentStatus(name="openclip", status="FAILED", detail=str(exc)))
        return out
