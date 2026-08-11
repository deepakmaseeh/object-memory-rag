from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.logging_utils import get_logger

log = get_logger(__name__)


class ConditionalVLM:
    """
    Invoke qwen2.5vl (or configured vision model) only when needed.
    Used for NEW / UNCERTAIN objects — not every frame.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.ollama.vision_model
        self.enabled = self.settings.ollama.vision_enabled

    def should_invoke(
        self,
        *,
        is_new: bool,
        similarity: float,
        decision: str = "NEW",
        force: bool = False,
        attributes_changed: bool = False,
        explicit_request: bool = False,
    ) -> bool:
        if force or explicit_request:
            return True
        if not self.enabled:
            return False
        if attributes_changed:
            return True
        decision = (decision or "").upper()
        if decision in {"NEW", "UNCERTAIN"} and self.settings.ollama.vision_on_new_object:
            return True
        if is_new and self.settings.ollama.vision_on_new_object:
            return True
        low = self.settings.ollama.vision_low_confidence_threshold
        if (not is_new) and similarity < low and self.settings.ollama.vision_on_uncertain:
            return True
        return False

    def describe_crop(self, crop_path: str | Path, class_name: str = "") -> dict[str, Any]:
        """Structured attributes for long-term object memory."""
        path = Path(crop_path)
        if not path.exists():
            return {}
        try:
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode("ascii")
            prompt = (
                f"You are describing a single cropped object for a long-term visual memory system. "
                f"Detector class guess: {class_name or 'unknown'}. "
                "Return ONLY a compact JSON object (no markdown) with keys when applicable: "
                "object_type, color, material, shape, cap, brand, condition, visible_text, notes. "
                "Be factual; do not invent brands if text is unreadable."
            )
            url = f"{self.settings.ollama.base_url.rstrip('/')}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }
            with httpx.Client(timeout=self.settings.ollama.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                text = (resp.json().get("response") or "").strip()
            return self._parse_attrs(text)
        except Exception as exc:
            log.warning("VLM describe failed: %s", exc)
            return {}

    @staticmethod
    def _parse_attrs(text: str) -> dict[str, Any]:
        import json
        import re

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # Try to extract JSON object if model added noise
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # Drop empty values
                return {str(k): v for k, v in data.items() if v not in (None, "", [])}
        except Exception:
            pass
        return {"notes": text[:500]} if text else {}
