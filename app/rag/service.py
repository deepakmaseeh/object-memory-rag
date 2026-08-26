from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.graph.base import GraphStore
from app.logging_utils import get_logger
from app.memory.base import VectorStore
from app.retrieval.retriever import MemoryRetriever
from app.schemas import MemoryQuery, MemoryResponse

log = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def rag_model(self) -> str:
        return self.settings.ollama.rag_model or self.settings.ollama.model

    @property
    def fallback_model(self) -> str:
        return self.settings.ollama.fallback_model

    @property
    def vision_model(self) -> str:
        return self.settings.ollama.vision_model

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.rag_model
        url = f"{self.settings.ollama.base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Qwen3 thinking models often return empty/visible "think" blocks otherwise
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }
        with httpx.Client(timeout=self.settings.ollama.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            # Older Ollama may reject unknown "think" key — retry once without it
            if resp.status_code >= 400 and "think" in payload:
                payload.pop("think", None)
                resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or data.get("message", {}).get("content") or "").strip()
            text = _THINK_RE.sub("", text).strip()
            return text

    def health(self) -> bool:
        return bool(self.model_status().get("reachable"))

    def list_models(self) -> list[str]:
        url = f"{self.settings.ollama.base_url.rstrip('/')}/api/tags"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def model_status(self) -> dict[str, Any]:
        names = self.list_models()
        reachable = len(names) > 0 or self._ping()
        required = {
            "rag_model": self.rag_model,
            "fallback_model": self.fallback_model,
            "vision_model": self.vision_model,
        }
        available = {}
        for key, model in required.items():
            available[key] = any(
                n == model or n.startswith(model.split(":")[0]) for n in names
            )
        return {
            "reachable": reachable,
            "models": names,
            "required": required,
            "available": available,
        }

    def _ping(self) -> bool:
        url = f"{self.settings.ollama.base_url.rstrip('/')}/api/tags"
        try:
            with httpx.Client(timeout=3.0) as client:
                return client.get(url).status_code == 200
        except Exception:
            return False


class RAGService:
    """LLM is a reasoning/interface layer on top of retrieved memory context."""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = MemoryRetriever(
            vector_store, graph_store, settings=self.settings
        )
        self.llm = OllamaClient(self.settings)

    def _select_model(self, query: str) -> str:
        q = query.lower()
        simple_markers = ("hello", "hi ", "ping", "who are you", "help")
        if any(m in q for m in simple_markers) and len(q) < 40:
            return self.llm.fallback_model
        return self.llm.rag_model

    def _fallback_from_memory(self, context_items: list) -> str:
        if not context_items:
            return (
                "I could not find matching objects in memory. "
                "Process an image first, then ask about a class like bus, phone, bottle, or clock."
            )
        top = context_items[0]
        attrs = getattr(top, "attributes", None) or {}
        attr_bits = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:8]) if attrs else ""
        locs = getattr(top, "locations", None) or []
        loc = (locs[-1] if locs else None) or top.last_location or "unknown location"
        parts = [
            f"This is a {top.class_name} (physical object {top.object_id}).",
        ]
        if getattr(top, "product_label", None):
            parts.append(f"Product identity: {top.product_label}.")
        parts.append(f"It was last seen at {loc}")
        if top.last_scene:
            parts[-1] += f" (scene {top.last_scene})"
        parts[-1] += "."
        if attr_bits:
            parts.append(f"Stored details: {attr_bits}.")
        if top.observation_count:
            parts.append(f"Seen in {top.observation_count} observation(s).")
        return " ".join(parts)

    def answer(self, query: str) -> MemoryResponse:
        mq: MemoryQuery = self.retriever.parser.parse(query)
        context_items = self.retriever.retrieve(mq)
        context_text = self.retriever.format_context(context_items)
        model = self._select_model(query)

        prompt = (
            "You are an object memory assistant. Answer using ONLY the memory context below.\n"
            "Distinguish three concepts:\n"
            "- OBJECT CLASS: what kind of thing (e.g. bottle)\n"
            "- PRODUCT: brand/product identity (e.g. Aquafina 500ml) — may have multiple physical instances\n"
            "- PHYSICAL OBJECT: a specific instance (object_id) you have seen before\n"
            "For 'have I seen this exact X before' answer about the physical object_id.\n"
            "For 'how many of this product' count instances sharing a product signature.\n"
            "For 'which brand' use semantic/OCR attributes.\n"
            "If the context is insufficient, say you do not know. Do not invent objects.\n"
            "Be concise (2-4 sentences).\n\n"
            f"Memory context:\n{context_text}\n\n"
            f"User question: {query}\n\n"
            "Answer:"
        )

        answer = ""
        if not self.llm.health():
            answer = (
                self._fallback_from_memory(context_items)
                if context_items
                else "I could not find matching objects in memory, and Ollama is unavailable."
            )
            if context_items:
                answer = f"(Ollama offline) {answer}"
        else:
            try:
                answer = self.llm.generate(prompt, model=model)
            except Exception as exc:
                log.warning("primary LLM failed model=%s err=%s", model, exc)
                try:
                    answer = self.llm.generate(prompt, model=self.llm.fallback_model)
                    model = self.llm.fallback_model
                except Exception as exc2:
                    log.warning("fallback LLM failed err=%s", exc2)
                    answer = self._fallback_from_memory(context_items)
                    if not context_items:
                        answer = f"Retrieval failed LLM generation: {exc}"

        if not (answer or "").strip():
            answer = self._fallback_from_memory(context_items)

        log.info("RAG answer model=%s context_items=%d", model, len(context_items))
        return MemoryResponse(
            query=query,
            answer=answer.strip(),
            context=context_items,
            raw_context=context_text,
        )
