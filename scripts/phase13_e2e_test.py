#!/usr/bin/env python3
"""Phase 13 reproducible multi-signal identity scenario (unit-level, no GPU images required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.memory.identity_resolver import ClusterIdentityResolver
from app.memory.signature_builder import build_object_signature, derive_product_signature
from app.schemas.identity import SemanticSignature, TextSignature
from tests.test_second_loop_unit import FakeVectorStore


class MemGraph:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}

    def get_object_history(self, object_id: str) -> dict:
        obj = self.objects.get(object_id, {})
        return {"object": obj, "attributes": obj.get("attributes") or {}}


def _sig(brand: str, tokens: list[str] | None = None):
    return build_object_signature(
        class_name="bottle",
        ocr={"text": " ".join(tokens or [brand]), "tokens": tokens or [brand], "confidence": 0.95, "regions": []},
        vlm_attrs={"brand": brand, "object_type": "water bottle"},
    )


def main() -> int:
    settings = get_settings().model_copy(deep=True)
    settings.identity.enable_multi_signal = True
    settings.memory.uncertain_as_new = True

    store = FakeVectorStore()
    graph = MemGraph()
    resolver = ClusterIdentityResolver(store, settings)
    vec = [1.0, 0.1, 0.0, 0.0]

    print("=== Phase 13 E2E identity scenario ===\n")

    # 1) Brand A → NEW
    sig_a = _sig("Brand A")
    m1 = resolver.resolve(vec, "bottle", new_signature=sig_a, graph_store=graph)
    print(f"1) Brand A bottle → {m1.decision} {m1.object_id}")
    store.upsert_observation("obs1", vec, {"object_id": m1.object_id, "class_name": "bottle"})
    store.clusters.append(
        {
            "cluster_id": "cluster_bottle_0",
            "vector": vec,
            "payload": {"cluster_id": "cluster_bottle_0", "class_name": "bottle", "object_ids": [m1.object_id]},
        }
    )
    graph.objects[m1.object_id] = {
        "object_id": m1.object_id,
        "class_name": "bottle",
        "observation_count": 1,
        "attributes": {"brand": "Brand A"},
        "object_signature": sig_a.model_dump(mode="json"),
    }

    # 2) Same Brand A physical → KNOWN
    m2 = resolver.resolve(vec, "bottle", new_signature=sig_a, graph_store=graph)
    print(f"2) Same Brand A → {m2.decision} {m2.object_id} (expect same)")

    # 3) Brand B similar shape → NEW/UNCERTAIN, not merged
    sig_b = _sig("Brand B")
    m3 = resolver.resolve(vec, "bottle", new_signature=sig_b, graph_store=graph)
    print(f"3) Brand B similar → {m3.decision} {m3.object_id} reasons={m3.reason_codes}")

    # 4) Different physical Brand A → NEW (same product signature, different object)
    sig_a2 = _sig("Brand A", tokens=["Brand", "A", "500ml"])
    prod = derive_product_signature(sig_a2)
    m4 = resolver.resolve([0.92, 0.12, 0.0, 0.0], "bottle", new_signature=sig_a2, graph_store=graph)
    print(f"4) Different physical Brand A → {m4.decision} {m4.object_id} product={prod.product_signature_id if prod else None}")

    ok = (
        m1.decision == "NEW"
        and m2.decision == "KNOWN"
        and m2.object_id == m1.object_id
        and m3.object_id != m1.object_id
        and m3.decision in {"NEW", "UNCERTAIN"}
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
