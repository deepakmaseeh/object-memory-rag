from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.config import Settings, get_settings
from app.logging_utils import get_logger
from app.ocr.base import OCRReader, ImageLike

log = get_logger(__name__)


class NoOpOCRReader(OCRReader):
    """Neutral OCR — pipeline continues without text signal."""

    @property
    def available(self) -> bool:
        return False

    def extract_text(self, image: ImageLike) -> dict[str, Any]:
        return {"text": "", "tokens": [], "confidence": 0.0, "regions": []}


class EasyOCRReader(OCRReader):
    """Optional local OCR when easyocr is installed."""

    def __init__(self) -> None:
        import easyocr  # type: ignore

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def extract_text(self, image: ImageLike) -> dict[str, Any]:
        path = Path(image)
        if not path.exists():
            return {"text": "", "tokens": [], "confidence": 0.0, "regions": []}
        try:
            results = self._reader.readtext(str(path))
        except Exception as exc:
            log.warning("EasyOCR failed: %s", exc)
            return {"text": "", "tokens": [], "confidence": 0.0, "regions": []}
        tokens: list[str] = []
        regions: list[dict[str, Any]] = []
        confidences: list[float] = []
        for bbox, text, conf in results:
            t = str(text).strip()
            if not t:
                continue
            tokens.append(t)
            confidences.append(float(conf))
            regions.append({"text": t, "confidence": float(conf), "bbox": bbox})
        raw = " ".join(tokens)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": raw,
            "tokens": tokens,
            "confidence": avg_conf,
            "regions": regions,
        }


class PillowOCRReader(OCRReader):
    """
    Lightweight fallback using pytesseract when installed.
    Requires Tesseract binary on PATH.
    """

    def extract_text(self, image: ImageLike) -> dict[str, Any]:
        import pytesseract  # type: ignore

        path = Path(image)
        if not path.exists():
            return {"text": "", "tokens": [], "confidence": 0.0, "regions": []}
        try:
            img = Image.open(path).convert("RGB")
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            tokens: list[str] = []
            confidences: list[float] = []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                w = str(word).strip()
                if not w:
                    continue
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = 0.0
                if c < 0:
                    continue
                tokens.append(w)
                confidences.append(c / 100.0)
            raw = " ".join(tokens)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return {
                "text": raw,
                "tokens": tokens,
                "confidence": avg_conf,
                "regions": [],
            }
        except Exception as exc:
            log.warning("PillowOCR failed: %s", exc)
            return {"text": "", "tokens": [], "confidence": 0.0, "regions": []}


def create_ocr_reader(settings: Optional[Settings] = None) -> OCRReader:
    settings = settings or get_settings()
    backend = (getattr(settings, "ocr", None) and settings.ocr.backend or "auto").lower()
    if backend in {"none", "noop", "disabled"}:
        return NoOpOCRReader()
    if backend == "easyocr":
        try:
            return EasyOCRReader()
        except Exception as exc:
            log.warning("EasyOCR unavailable (%s); using NoOp OCR", exc)
            return NoOpOCRReader()
    if backend == "pytesseract":
        try:
            import pytesseract  # noqa: F401

            return PillowOCRReader()
        except Exception as exc:
            log.warning("pytesseract unavailable (%s); using NoOp OCR", exc)
            return NoOpOCRReader()
    # auto: try easyocr then pytesseract
    for cls in (EasyOCRReader, PillowOCRReader):
        try:
            if cls is EasyOCRReader:
                import easyocr  # noqa: F401
            else:
                import pytesseract  # noqa: F401
            return cls()
        except Exception:
            continue
    log.info("No OCR backend available; text signal will be neutral")
    return NoOpOCRReader()
