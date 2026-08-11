from __future__ import annotations

from typing import Any, Optional

from app.config import Settings, get_settings
from app.graph.base import GraphStore
from app.memory.base import VectorStore
from app.retrieval.query_parser import QueryParser
from app.schemas import MemoryContextItem, MemoryQuery


class MemoryRetriever:
    """Combine graph memory (+ optional class filter) into LLM-ready context."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: Any = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder  # reserved for future embedding-based text retrieval
        self.settings = settings or get_settings()
        self.parser = QueryParser()

    def retrieve(self, query: str | MemoryQuery) -> list[MemoryContextItem]:
        if isinstance(query, str):
            mq = self.parser.parse(query)
        else:
            mq = query

        items: list[MemoryContextItem] = []
        seen: set[str] = set()

        if mq.class_name:
            try:
                rows = self.graph_store.search_objects_by_class(mq.class_name, limit=mq.top_k)
            except Exception:
                rows = []
            for row in rows:
                obj = row.get("object") or {}
                oid = obj.get("object_id")
                if not oid or oid in seen:
                    continue
                seen.add(oid)
                items.append(self._build_item(oid, obj, fallback_scene=row.get("last_scene")))
        else:
            if hasattr(self.graph_store, "list_objects"):
                try:
                    for row in self.graph_store.list_objects(limit=mq.top_k):
                        obj = row.get("object") or {}
                        oid = obj.get("object_id")
                        if not oid or oid in seen:
                            continue
                        seen.add(oid)
                        items.append(self._build_item(oid, obj))
                except Exception:
                    pass

        return items[: mq.top_k]

    def _build_item(
        self,
        object_id: str,
        obj: Optional[dict[str, Any]] = None,
        fallback_scene: Optional[str] = None,
    ) -> MemoryContextItem:
        hist: dict[str, Any] = {}
        try:
            hist = self.graph_store.get_object_history(object_id) or {}
        except Exception:
            hist = {}

        obj = dict(hist.get("object") or obj or {})
        if not obj.get("object_id"):
            obj["object_id"] = object_id

        attrs = hist.get("attributes") if isinstance(hist.get("attributes"), dict) else {}
        if not attrs:
            attrs = obj.get("attributes") if isinstance(obj.get("attributes"), dict) else {}

        locations = list(hist.get("locations") or [])
        observations = list(hist.get("observations") or [])
        latest = observations[0] if observations else None

        last_scene = (
            (latest or {}).get("scene_name")
            or fallback_scene
            or (latest or {}).get("scene_id")
        )
        last_location = locations[-1] if locations else None
        last_seen = obj.get("last_seen") or (latest or {}).get("timestamp")
        class_name = obj.get("class_name") or "unknown"
        count = int(obj.get("observation_count") or len(observations) or 0)

        return MemoryContextItem(
            object_id=object_id,
            class_name=class_name,
            last_scene=last_scene,
            last_location=last_location,
            last_seen=last_seen,
            observation_count=count,
            summary=self._summarize(obj, latest, attrs, locations),
            attributes=dict(attrs),
            locations=list(locations),
        )

    def format_context(self, items: list[MemoryContextItem]) -> str:
        if not items:
            return "No matching objects found in memory."
        lines = []
        for i, item in enumerate(items, 1):
            attrs = item.attributes or {}
            attr_txt = (
                ", ".join(f"{k}={self._fmt_val(v)}" for k, v in list(attrs.items())[:12])
                if attrs
                else "none"
            )
            locs = item.locations or ([item.last_location] if item.last_location else [])
            loc_txt = ", ".join(str(x) for x in locs) if locs else "unknown"
            lines.append(
                f"{i}. Object {item.object_id} ({item.class_name})\n"
                f"   locations: {loc_txt}\n"
                f"   scene: {item.last_scene or 'unknown'}\n"
                f"   last_seen: {item.last_seen or 'unknown'}\n"
                f"   observations: {item.observation_count}\n"
                f"   attributes: {attr_txt}\n"
                f"   note: {item.summary}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_val(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            return "/".join(str(x) for x in v)
        return str(v)

    @staticmethod
    def _summarize(
        obj: dict,
        latest: Optional[dict],
        attrs: Optional[dict] = None,
        locations: Optional[list] = None,
    ) -> str:
        scene = (latest or {}).get("scene_name") or "unknown scene"
        ts = (latest or {}).get("timestamp") or obj.get("last_seen") or "unknown time"
        parts = [f"Most recent observation: scene '{scene}' at {ts}."]
        if locations:
            parts.append(f"Known locations: {', '.join(map(str, locations))}.")
        if attrs:
            attr_bits = ", ".join(
                f"{k}={MemoryRetriever._fmt_val(v)}" for k, v in list(attrs.items())[:12]
            )
            parts.append(f"Stored attributes: {attr_bits}.")
        return " ".join(parts)
