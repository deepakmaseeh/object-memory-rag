from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.logging_utils import get_logger

from app.schemas.identity import VLMVerificationResult

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
        decision = (decision or "").upper()
        # KNOWN fast path: do not invoke VLM for routine re-observations
        if decision == "KNOWN" and not attributes_changed:
            return False
        if attributes_changed:
            return True
        if decision in {"NEW", "UNCERTAIN"} and self.settings.ollama.vision_on_new_object:
            return True
        if is_new and self.settings.ollama.vision_on_new_object:
            return True
        low = self.settings.ollama.vision_low_confidence_threshold
        if (not is_new) and similarity < low and self.settings.ollama.vision_on_uncertain:
            return True
        return False

    def should_verify(self, decision: str) -> bool:
        if not self.enabled:
            return False
        return (decision or "").upper() == "UNCERTAIN"

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

    def verify_same_physical_object(
        self,
        new_crop_path: str | Path,
        candidate_crop_path: str | Path,
        *,
        candidate_metadata: Optional[dict[str, Any]] = None,
        ocr_text: str = "",
    ) -> VLMVerificationResult:
        """Dual-crop VLM verification for UNCERTAIN identity cases."""
        new_path = Path(new_crop_path)
        cand_path = Path(candidate_crop_path)
        if not new_path.exists() or not cand_path.exists():
            return VLMVerificationResult(
                same_physical_object=None,
                confidence=0.0,
                reason="missing crop image",
            )
        try:
            images = [
                base64.b64encode(new_path.read_bytes()).decode("ascii"),
                base64.b64encode(cand_path.read_bytes()).decode("ascii"),
            ]
            meta = candidate_metadata or {}
            prompt = (
                "You compare two cropped photos of objects for a physical-instance memory system. "
                "Image 1 is the NEW sighting. Image 2 is a CANDIDATE previously stored object.\n"
                f"Candidate metadata: {meta}\n"
                f"OCR on new crop: {ocr_text or 'none'}\n"
                "Return ONLY JSON with keys: "
                "same_physical_object (boolean), confidence (0-1 number), reason (string), "
                "matching_features (array of strings), different_features (array of strings). "
                "same_physical_object means the exact same physical item, not merely same brand/product."
            )
            url = f"{self.settings.ollama.base_url.rstrip('/')}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": images,
                "stream": False,
            }
            with httpx.Client(timeout=self.settings.ollama.timeout_seconds) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                text = (resp.json().get("response") or "").strip()
            return self._parse_verification(text)
        except Exception as exc:
            log.warning("VLM verify failed: %s", exc)
            return VLMVerificationResult(
                same_physical_object=None,
                confidence=0.0,
                reason=str(exc),
            )

    @staticmethod
    def _parse_verification(text: str) -> VLMVerificationResult:
        import json
        import re

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                same = data.get("same_physical_object")
                if isinstance(same, str):
                    same = same.lower() in {"true", "yes", "1"}
                return VLMVerificationResult(
                    same_physical_object=same if same is not None else None,
                    confidence=float(data.get("confidence") or 0.0),
                    reason=str(data.get("reason") or ""),
                    matching_features=list(data.get("matching_features") or []),
                    different_features=list(data.get("different_features") or []),
                )
        except Exception:
            pass
        return VLMVerificationResult(
            same_physical_object=None,
            confidence=0.0,
            reason=text[:300],
        )
