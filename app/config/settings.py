from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    name: str = "object-memory"


class PathsConfig(BaseModel):
    raw: str = "data/raw"
    processed: str = "data/processed"
    crops: str = "data/crops"
    masks: str = "data/masks"
    embeddings: str = "data/embeddings"
    models: str = "models"
    storage: str = "storage"


class ModelsConfig(BaseModel):
    detector: str = "yolo11n.pt"
    segmenter: str = "sam2_b.pt"
    embedder: str = "ViT-B-32"
    embedder_pretrained: str = "openai"
    device: str = "auto"  # auto | cpu | cuda | mps


class PerceptionConfig(BaseModel):
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 50


class EmbeddingConfig(BaseModel):
    vector_size: int = 512
    normalize: bool = True


class QdrantCollections(BaseModel):
    observations: str = "observations"
    clusters: str = "clusters"


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    prefer_local: bool = False
    local_path: str = "storage/qdrant"
    collections: QdrantCollections = Field(default_factory=QdrantCollections)


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "objectmemory"
    backend: str = "auto"  # auto | neo4j | local
    local_path: str = "storage/local_graph.json"


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"  # backward compatible alias for rag_model
    rag_model: str = "qwen3:8b"
    fallback_model: str = "llama3.2:latest"
    vision_model: str = "qwen2.5vl:3b"
    vision_high_quality_model: str = "qwen3-vl:8b-instruct-q4_K_M"
    vision_enabled: bool = True
    vision_on_new_object: bool = True
    vision_on_uncertain: bool = True
    vision_low_confidence_threshold: float = 0.90
    timeout_seconds: int = 120


class MemoryConfig(BaseModel):
    # Identity bands (calibrate later on real data). Prefer not merging aggressively.
    known_threshold: float = 0.90
    uncertain_threshold: float = 0.70
    match_threshold: float = 0.90  # alias of known_threshold for older code
    cluster_search_top_k: int = 3
    object_search_top_k: int = 10
    min_cluster_size: int = 3
    uncertain_as_new: bool = True

    def effective_known_threshold(self) -> float:
        # Prefer explicit known_threshold; fall back to match_threshold
        return float(self.known_threshold or self.match_threshold or 0.90)


class ClusteringConfig(BaseModel):
    n_clusters_per_class: int = 5
    min_samples_for_rebuild: int = 5


class DefaultSceneConfig(BaseModel):
    scene_id: str = "scene_default"
    name: str = "default"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    default_scene: DefaultSceneConfig = Field(default_factory=DefaultSceneConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    root_dir: Path = ROOT_DIR

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def ensure_dirs(self) -> None:
        for key in (
            self.paths.raw,
            self.paths.processed,
            self.paths.crops,
            self.paths.masks,
            self.paths.embeddings,
            self.paths.models,
            self.paths.storage,
        ):
            self.resolve_path(key).mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        device = (self.models.device or "auto").lower()
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


class EnvOverrides(BaseSettings):
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_backend: Optional[str] = None
    qdrant_host: Optional[str] = None
    qdrant_port: Optional[int] = None
    qdrant_prefer_local: Optional[bool] = None
    ollama_base_url: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_rag_model: Optional[str] = None
    ollama_fallback_model: Optional[str] = None
    ollama_vision_model: Optional[str] = None
    app_host: Optional[str] = None
    app_port: Optional[int] = None
    device: Optional[str] = None

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _apply_env(settings: Settings) -> Settings:
    env = EnvOverrides()
    if env.neo4j_uri:
        settings.neo4j.uri = env.neo4j_uri
    if env.neo4j_user:
        settings.neo4j.user = env.neo4j_user
    if env.neo4j_password:
        settings.neo4j.password = env.neo4j_password
    if env.neo4j_backend:
        settings.neo4j.backend = env.neo4j_backend
    if env.qdrant_host:
        settings.qdrant.host = env.qdrant_host
    if env.qdrant_port is not None:
        settings.qdrant.port = env.qdrant_port
    if env.qdrant_prefer_local is not None:
        settings.qdrant.prefer_local = env.qdrant_prefer_local
    if env.ollama_base_url:
        settings.ollama.base_url = env.ollama_base_url
    if env.ollama_rag_model:
        settings.ollama.rag_model = env.ollama_rag_model
        settings.ollama.model = env.ollama_rag_model
    elif env.ollama_model:
        settings.ollama.model = env.ollama_model
        settings.ollama.rag_model = env.ollama_model
    if env.ollama_fallback_model:
        settings.ollama.fallback_model = env.ollama_fallback_model
    if env.ollama_vision_model:
        settings.ollama.vision_model = env.ollama_vision_model
    if env.app_host:
        settings.app.host = env.app_host
    if env.app_port is not None:
        settings.app.port = env.app_port
    if env.device:
        settings.models.device = env.device
    # Keep model alias in sync when only rag_model set in yaml
    if settings.ollama.rag_model and settings.ollama.model == "llama3.2":
        settings.ollama.model = settings.ollama.rag_model
    return settings


@lru_cache
def get_settings(config_path: Optional[str] = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT_DIR / path
    raw = _load_yaml(path)
    settings = Settings.model_validate(raw) if raw else Settings()
    settings.root_dir = ROOT_DIR
    settings = _apply_env(settings)
    return settings


def clear_settings_cache() -> None:
    get_settings.cache_clear()
