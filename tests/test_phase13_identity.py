"""Phase 13 multi-signal identity tests."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from app.config import get_settings
from app.memory.identity_scorer import IdentityScorer
from app.memory.identity_resolver import ClusterIdentityResolver
from app.memory.signature_builder import build_object_signature, derive_product_signature
from app.ocr.reader import NoOpOCRReader
from app.schemas.identity import ObjectSignature, SemanticSignature, TextSignature, VisualSignature
from tests.test_second_loop_unit import FakeVectorStore


class FakeGraphStore:
    def __init__(self, objects: dict[str, dict[str, Any]]) -> None:
        self.objects = objects

    def get_object_history(self, object_id: str) -> dict[str, Any]:
        obj = self.objects.get(object_id, {})
        return {
            "object": obj,
            "attributes": obj.get("attributes") or {},
            "observations": obj.get("observations") or [],
        }


def _bottle_sig(brand: str, tokens: Optional[list[str]] = None, visual_ref: str = "obs1") -> ObjectSignature:
    return ObjectSignature(
        class_name="bottle",
        visual=VisualSignature(embedding_ref=visual_ref, shape="tall", aspect_ratio=1.2),
        semantic=SemanticSignature(brand=brand, object_type="water bottle", material="plastic"),
        text=TextSignature(raw_text=" ".join(tokens or [brand]), tokens=tokens or [brand], confidence=0.95),
    )


def _settings():
    s = get_settings().model_copy(deep=True)
    s.identity.enable_multi_signal = True
    s.memory.uncertain_as_new = True
    return s


def test_brand_a_new_object():
    settings = _settings()
    store = FakeVectorStore()
    resolver = ClusterIdentityResolver(store, settings)
    sig = _bottle_sig("Brand A")
    match = resolver.resolve([1.0] * 8, "bottle", new_signature=sig, graph_store=FakeGraphStore({}))
    assert match.decision == "NEW"
    assert match.is_new is True


def test_same_brand_a_known_on_second_sighting():
    settings = _settings()
    store = FakeVectorStore()
    oid = "obj_000001"
    store.obs.append(
        {
            "observation_id": "obs_old",
            "vector": [1.0] * 8,
            "payload": {
                "object_id": oid,
                "class_name": "bottle",
                "cluster_id": "cluster_bottle_0",
            },
        }
    )
    store.clusters.append(
        {
            "cluster_id": "cluster_bottle_0",
            "vector": [1.0] * 8,
            "payload": {
                "cluster_id": "cluster_bottle_0",
                "class_name": "bottle",
                "object_ids": [oid],
            },
        }
    )
    graph = FakeGraphStore(
        {
            oid: {
                "object_id": oid,
                "class_name": "bottle",
                "observation_count": 2,
                "attributes": {"brand": "Brand A"},
                "object_signature": _bottle_sig("Brand A").model_dump(mode="json"),
            }
        }
    )
    resolver = ClusterIdentityResolver(store, settings)
    sig = _bottle_sig("Brand A")
    match = resolver.resolve([1.0] * 8, "bottle", new_signature=sig, graph_store=graph)
    assert match.decision == "KNOWN"
    assert match.object_id == oid
    assert match.is_new is False


def test_brand_b_does_not_match_brand_a_despite_visual_similarity():
    settings = _settings()
    store = FakeVectorStore()
    oid = "obj_000001"
    store.obs.append(
        {
            "observation_id": "obs_old",
            "vector": [0.99] * 8,
            "payload": {"object_id": oid, "class_name": "bottle"},
        }
    )
    store.clusters.append(
        {
            "cluster_id": "cluster_bottle_0",
            "vector": [0.99] * 8,
            "payload": {
                "cluster_id": "cluster_bottle_0",
                "class_name": "bottle",
                "object_ids": [oid],
            },
        }
    )
    graph = FakeGraphStore(
        {
            oid: {
                "object_id": oid,
                "class_name": "bottle",
                "observation_count": 1,
                "attributes": {"brand": "Brand A"},
                "object_signature": _bottle_sig("Brand A").model_dump(mode="json"),
            }
        }
    )
    resolver = ClusterIdentityResolver(store, settings)
    sig = _bottle_sig("Brand B")
    match = resolver.resolve([0.99] * 8, "bottle", new_signature=sig, graph_store=graph)
    assert match.decision in {"UNCERTAIN", "NEW"}
    assert match.object_id != oid or match.is_new is True
    assert "BRAND_CONFLICT" in (match.reason_codes or [])


def test_same_product_different_physical_instances_allowed():
    settings = _settings()
    scorer = IdentityScorer(settings)
    sig_a = _bottle_sig("Brand X", tokens=["Brand", "X", "500ml"])
    sig_b = _bottle_sig("Brand X", tokens=["Brand", "X", "500ml"], visual_ref="obs2")
    prod_a = derive_product_signature(sig_a)
    prod_b = derive_product_signature(sig_b)
    assert prod_a and prod_b
    assert prod_a.product_signature_id == prod_b.product_signature_id
    result = scorer.score(sig_b, sig_a, visual_similarity=0.88, candidate_object_id="obj_a", observation_count=1)
    assert result.decision in {"UNCERTAIN", "NEW", "KNOWN"}


def test_brand_conflict_blocks_known():
    scorer = IdentityScorer(_settings())
    new_sig = _bottle_sig("Coca-Cola")
    cand_sig = _bottle_sig("Pepsi")
    result = scorer.score(
        new_sig, cand_sig, visual_similarity=0.94, candidate_object_id="obj_x", observation_count=3
    )
    assert result.brand_conflict is True
    assert result.decision == "UNCERTAIN"
    assert "BRAND_CONFLICT" in result.reason_codes


def test_ocr_unavailable_neutral():
    reader = NoOpOCRReader()
    out = reader.extract_text("missing.jpg")
    assert out["text"] == ""
    assert out["confidence"] == 0.0
    sig = build_object_signature(class_name="bottle", ocr=out)
    assert sig.text.confidence == 0.0


def test_vlm_should_not_invoke_on_known():
    from app.perception.vlm import ConditionalVLM

    vlm = ConditionalVLM(_settings())
    assert vlm.should_invoke(is_new=False, similarity=0.95, decision="KNOWN") is False


def test_product_vs_object_signature_ids_differ():
    sig1 = _bottle_sig("Brand A")
    sig2 = _bottle_sig("Brand B")
    p1 = derive_product_signature(sig1)
    p2 = derive_product_signature(sig2)
    assert p1 and p2
    assert p1.product_signature_id != p2.product_signature_id


def test_rag_query_parser_modes():
    from app.retrieval.query_parser import QueryParser

    p = QueryParser()
    assert p.parse_extended("Have I seen this exact bottle before?").mode == "instance"
    assert p.parse_extended("How many bottles of this product have I seen?").mode == "count"
    assert p.parse_extended("Which brand is this?").mode == "brand"
    assert p.parse_extended("Where did I last see my phone?").mode == "location"


def test_high_visual_brand_conflict_not_known():
    settings = _settings()
    scorer = IdentityScorer(settings)
    res = scorer.score(
        _bottle_sig("Brand B"),
        _bottle_sig("Brand A"),
        visual_similarity=0.94,
        candidate_object_id="obj_a",
    )
    assert res.decision != "KNOWN"
    assert res.visual_score == pytest.approx(0.94)
