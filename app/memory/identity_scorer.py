from __future__ import annotations

import re
from typing import Any, Optional

from app.config import IdentityConfig, Settings, get_settings
from app.schemas.identity import IdentityScoreResult, ObjectSignature


def _norm(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _token_jaccard(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.0
    sa = {_norm(t) for t in a if _norm(t)}
    sb = {_norm(t) for t in b if _norm(t)}
    if not sa or not sb:
        return 0.5
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _field_match(a: Optional[str], b: Optional[str]) -> float:
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 0.5
    if not na or not nb:
        return 0.5
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.75
    return 0.0


class IdentityScorer:
    """Weighted multi-signal identity scoring with hard conflict rules."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.cfg: IdentityConfig = self.settings.identity

    def score(
        self,
        new_sig: ObjectSignature,
        candidate_sig: ObjectSignature,
        *,
        visual_similarity: float,
        candidate_object_id: str,
        observation_count: int = 1,
    ) -> IdentityScoreResult:
        w = self.cfg.weights
        reason_codes: list[str] = []

        class_match = _norm(new_sig.class_name) == _norm(candidate_sig.class_name)
        if not class_match and new_sig.class_name and candidate_sig.class_name:
            reason_codes.append("CLASS_MISMATCH")

        visual_score = max(0.0, min(1.0, float(visual_similarity)))
        if visual_score >= 0.85:
            reason_codes.append("HIGH_VISUAL_SIMILARITY")

        text_score = _token_jaccard(
            new_sig.text.tokens or _split_tokens(new_sig.text.raw_text),
            candidate_sig.text.tokens or _split_tokens(candidate_sig.text.raw_text),
        )
        if text_score >= 0.8:
            reason_codes.append("HIGH_TEXT_SIMILARITY")

        brand_score = _field_match(new_sig.semantic.brand, candidate_sig.semantic.brand)
        product_score = _field_match(
            new_sig.semantic.product_name, candidate_sig.semantic.product_name
        )
        semantic_score = _semantic_similarity(new_sig.semantic, candidate_sig.semantic)

        attr_score = _attribute_similarity(new_sig, candidate_sig)
        shape_score = _shape_similarity(new_sig, candidate_sig)

        # Historical: more observations → slightly higher trust in candidate continuity
        historical_score = min(1.0, 0.5 + 0.05 * max(0, observation_count - 1))

        brand_conflict = self._brand_conflict(new_sig, candidate_sig)
        if brand_conflict:
            reason_codes.append("BRAND_CONFLICT")

        product_conflict = self._product_conflict(new_sig, candidate_sig)
        if product_conflict:
            reason_codes.append("PRODUCT_CONFLICT")

        overall = (
            w.visual * visual_score
            + w.text * text_score
            + w.brand * brand_score
            + w.semantic * semantic_score
            + w.attributes * attr_score
            + w.historical * historical_score
            + w.product * product_score * 0.5
            + w.shape * shape_score * 0.5
        )
        overall = max(0.0, min(1.0, overall))

        decision = self._decide(
            overall=overall,
            visual_score=visual_score,
            brand_conflict=brand_conflict,
            product_conflict=product_conflict,
            class_match=class_match,
            reason_codes=reason_codes,
        )

        return IdentityScoreResult(
            overall_score=overall,
            visual_score=visual_score,
            text_score=text_score,
            semantic_score=semantic_score,
            attribute_score=attr_score,
            brand_score=brand_score,
            product_score=product_score,
            shape_score=shape_score,
            historical_score=historical_score,
            class_match=class_match,
            brand_conflict=brand_conflict,
            decision=decision,
            reason_codes=reason_codes,
            candidate_object_id=candidate_object_id,
        )

    def _decide(
        self,
        *,
        overall: float,
        visual_score: float,
        brand_conflict: bool,
        product_conflict: bool,
        class_match: bool,
        reason_codes: list[str],
    ) -> str:
        known_t = float(self.cfg.thresholds.known)
        uncertain_t = float(self.cfg.thresholds.uncertain)

        if brand_conflict or product_conflict:
            if visual_score >= 0.80:
                reason_codes.append("CONFLICT_BLOCKS_KNOWN")
            return "UNCERTAIN"

        if not class_match and overall < known_t:
            return "NEW"

        if overall >= known_t and not brand_conflict and not product_conflict:
            return "KNOWN"
        if overall >= uncertain_t:
            return "UNCERTAIN"
        return "NEW"

    def _brand_conflict(self, a: ObjectSignature, b: ObjectSignature) -> bool:
        min_conf = float(self.cfg.conflict_min_confidence)
        ba, bb = a.semantic.brand, b.semantic.brand
        if not ba or not bb:
            return False
        ta = max(a.text.confidence, 0.0)
        tb = max(b.text.confidence, 0.0)
        if _norm(ba) == _norm(bb):
            return False
        if ta >= min_conf or tb >= min_conf or (a.semantic.brand and b.semantic.brand):
            return True
        return False

    def _product_conflict(self, a: ObjectSignature, b: ObjectSignature) -> bool:
        pa, pb = a.semantic.product_name, b.semantic.product_name
        if not pa or not pb:
            return False
        return _norm(pa) != _norm(pb)


def _split_tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    return [t for t in re.split(r"[\W_]+", text) if t]


def _semantic_similarity(a: Any, b: Any) -> float:
    fields = ["object_type", "brand", "product_name", "material"]
    scores = [_field_match(getattr(a, f, None), getattr(b, f, None)) for f in fields]
    return sum(scores) / len(scores) if scores else 0.5


def _attribute_similarity(a: ObjectSignature, b: ObjectSignature) -> float:
    fa = {str(x).lower() for x in (a.distinguishing_features or [])}
    fb = {str(x).lower() for x in (b.distinguishing_features or [])}
    if not fa and not fb:
        return 0.5
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def _shape_similarity(a: ObjectSignature, b: ObjectSignature) -> float:
    sa = _norm(a.visual.shape)
    sb = _norm(b.visual.shape)
    if not sa and not sb:
        ar_a = a.visual.aspect_ratio
        ar_b = b.visual.aspect_ratio
        if ar_a and ar_b:
            diff = abs(ar_a - ar_b) / max(ar_a, ar_b, 1e-6)
            return max(0.0, 1.0 - diff)
        return 0.5
    if sa and sb:
        return 1.0 if sa == sb else 0.3
    return 0.5
