from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from app.schemas.identity import (
    ObjectSignature,
    ProductSignature,
    SemanticSignature,
    TextSignature,
    VisualSignature,
)


def build_object_signature(
    *,
    class_name: str,
    crop_path: Optional[str] = None,
    bbox: Optional[list[float]] = None,
    ocr: Optional[dict[str, Any]] = None,
    vlm_attrs: Optional[dict[str, Any]] = None,
    embedding_ref: Optional[str] = None,
) -> ObjectSignature:
    """Build structured signature from available signals. Never hallucinates missing fields."""
    ocr = ocr or {}
    vlm_attrs = vlm_attrs or {}

    visual = _visual_from_crop(crop_path, bbox, embedding_ref)
    text = TextSignature(
        raw_text=ocr.get("text") or None,
        tokens=list(ocr.get("tokens") or []),
        confidence=float(ocr.get("confidence") or 0.0),
        regions=list(ocr.get("regions") or []),
    )

    brand = _clean(vlm_attrs.get("brand"))
    visible_text = _clean(vlm_attrs.get("visible_text"))
    if not brand and text.tokens:
        brand = text.tokens[0]
    if not brand and visible_text:
        brand = visible_text.split()[0] if visible_text else None

    semantic = SemanticSignature(
        object_type=_clean(vlm_attrs.get("object_type")),
        brand=brand,
        product_name=_clean(vlm_attrs.get("product_name")),
        material=_clean(vlm_attrs.get("material")),
    )

    distinguishing: list[str] = []
    for key in ("color", "cap", "shape", "condition", "notes"):
        val = vlm_attrs.get(key)
        if val:
            distinguishing.append(f"{key}: {val}")
    for tok in text.tokens[:5]:
        if tok and tok not in distinguishing:
            distinguishing.append(str(tok))

    return ObjectSignature(
        class_name=class_name,
        visual=visual,
        semantic=semantic,
        text=text,
        distinguishing_features=distinguishing[:12],
    )


def signature_from_stored(
    *,
    class_name: str,
    attributes: Optional[dict[str, Any]] = None,
    signature: Optional[dict[str, Any]] = None,
    embedding_ref: Optional[str] = None,
) -> ObjectSignature:
    attributes = attributes or {}
    if signature:
        try:
            return ObjectSignature.model_validate(signature)
        except Exception:
            pass
    ocr = attributes.get("ocr") if isinstance(attributes.get("ocr"), dict) else {}
    return build_object_signature(
        class_name=class_name,
        ocr=ocr,
        vlm_attrs=attributes,
        embedding_ref=embedding_ref,
    )


def derive_product_signature(sig: ObjectSignature) -> Optional[ProductSignature]:
    brand = sig.semantic.brand
    product = sig.semantic.product_name or sig.semantic.object_type
    if not brand and not product and not sig.text.tokens:
        return None

    parts = [
        _slug(sig.class_name or "object"),
        _slug(brand or ""),
        _slug(product or ""),
        _slug(sig.text.raw_text or ""),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    pid = f"product_{digest}"

    label_bits = [x for x in [brand, product] if x]
    text_sig = " ".join(label_bits) if label_bits else (sig.text.raw_text or None)

    return ProductSignature(
        product_signature_id=pid,
        class_name=sig.class_name,
        brand=brand,
        product_name=product,
        semantic_attributes={
            k: v
            for k, v in {
                "object_type": sig.semantic.object_type,
                "material": sig.semantic.material,
            }.items()
            if v
        },
        text_signature=text_sig,
    )


def product_display_name(prod: ProductSignature) -> str:
    bits = [prod.brand, prod.product_name]
    label = " ".join(x for x in bits if x)
    return label or prod.text_signature or prod.product_signature_id


def _visual_from_crop(
    crop_path: Optional[str],
    bbox: Optional[list[float]],
    embedding_ref: Optional[str],
) -> VisualSignature:
    aspect = None
    colors: list[str] = []
    shape = None
    if bbox and len(bbox) >= 4:
        w = max(1.0, float(bbox[2]) - float(bbox[0]))
        h = max(1.0, float(bbox[3]) - float(bbox[1]))
        aspect = round(h / w, 3) if w else None
        if aspect and aspect > 1.4:
            shape = "tall"
        elif aspect and aspect < 0.7:
            shape = "wide"
        else:
            shape = "compact"
    if crop_path and Path(crop_path).exists():
        try:
            img = Image.open(crop_path).convert("RGB")
            img.thumbnail((32, 32))
            pixels = list(img.getdata())
            r = sum(p[0] for p in pixels) / len(pixels)
            g = sum(p[1] for p in pixels) / len(pixels)
            b = sum(p[2] for p in pixels) / len(pixels)
            colors.append(_dominant_color_label(r, g, b))
        except Exception:
            pass
    return VisualSignature(
        embedding_ref=embedding_ref,
        dominant_colors=colors,
        shape=shape,
        aspect_ratio=aspect,
    )


def _dominant_color_label(r: float, g: float, b: float) -> str:
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 60 and g < 60 and b < 60:
        return "black"
    if b > r + 30 and b > g + 20:
        return "blue"
    if g > r + 20 and g > b + 20:
        return "green"
    if r > g + 30 and r > b + 30:
        return "red"
    if abs(r - g) < 25 and abs(g - b) < 25:
        return "gray"
    return "mixed"


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"unknown", "none", "n/a", "null"}:
        return None
    return s


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
