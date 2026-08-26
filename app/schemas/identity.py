from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class VisualSignature(BaseModel):
    embedding_ref: Optional[str] = None
    dominant_colors: list[str] = Field(default_factory=list)
    shape: Optional[str] = None
    aspect_ratio: Optional[float] = None


class SemanticSignature(BaseModel):
    object_type: Optional[str] = None
    brand: Optional[str] = None
    product_name: Optional[str] = None
    material: Optional[str] = None


class TextSignature(BaseModel):
    raw_text: Optional[str] = None
    tokens: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    regions: list[dict[str, Any]] = Field(default_factory=list)


class ObjectSignature(BaseModel):
    """Multi-signal identity fingerprint for one observation."""

    class_name: Optional[str] = None
    visual: VisualSignature = Field(default_factory=VisualSignature)
    semantic: SemanticSignature = Field(default_factory=SemanticSignature)
    text: TextSignature = Field(default_factory=TextSignature)
    distinguishing_features: list[str] = Field(default_factory=list)


class ProductSignature(BaseModel):
    """Semantic product identity (brand/product), separate from physical instance."""

    product_signature_id: str
    class_name: Optional[str] = None
    brand: Optional[str] = None
    product_name: Optional[str] = None
    semantic_attributes: dict[str, Any] = Field(default_factory=dict)
    text_signature: Optional[str] = None


class IdentityScoreResult(BaseModel):
    overall_score: float = 0.0
    visual_score: float = 0.0
    text_score: float = 0.0
    semantic_score: float = 0.0
    attribute_score: float = 0.0
    brand_score: float = 0.0
    product_score: float = 0.0
    shape_score: float = 0.0
    historical_score: float = 0.0
    class_match: bool = True
    brand_conflict: bool = False
    decision: str = "NEW"  # KNOWN | UNCERTAIN | NEW
    reason_codes: list[str] = Field(default_factory=list)
    candidate_object_id: Optional[str] = None


class VLMVerificationResult(BaseModel):
    same_physical_object: Optional[bool] = None
    confidence: float = 0.0
    reason: str = ""
    matching_features: list[str] = Field(default_factory=list)
    different_features: list[str] = Field(default_factory=list)
