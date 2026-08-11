#!/usr/bin/env python
"""
Second-loop identity verification.

Processes the same image twice and asserts that second-pass detections
match existing object_ids (is_new=false, no new identities created).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.memory.memory_updater import PipelineService


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-loop same-object identity test")
    parser.add_argument("path", type=str, help="Path to image file")
    parser.add_argument("--location", type=str, default="Desk")
    parser.add_argument("--min-dets", type=int, default=1)
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="Enable conditional VLM (off by default for speed)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not args.with_vlm:
        settings.ollama.vision_enabled = False

    service = PipelineService(settings)
    try:
        first = service.process_image_path(args.path, location_name=args.location)
        second = service.process_image_path(args.path, location_name=args.location)

        if first.detection_count < args.min_dets:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"Need at least {args.min_dets} detections, got {first.detection_count}",
                    },
                    indent=2,
                )
            )
            return 2

        first_ids = {m.object_id for m in first.matches}
        rows = []
        for m in second.matches:
            reuse_ok = (not m.is_new) and (m.object_id in first_ids)
            rows.append(
                {
                    "class": m.class_name,
                    "second_object_id": m.object_id,
                    "matched_existing_object": m.matched_existing_object,
                    "is_new": m.is_new,
                    "similarity": m.similarity,
                    "reuse_ok": reuse_ok,
                }
            )

        # Second pass must not create new identities for the same scene content
        duplicate_created = any(m.is_new for m in second.matches)
        unknown_ids = any(m.object_id not in first_ids for m in second.matches)
        ok = (not duplicate_created) and (not unknown_ids) and all(r["reuse_ok"] for r in rows)

        print(
            json.dumps(
                {
                    "ok": ok,
                    "duplicate_created": duplicate_created,
                    "first_image_id": first.image_id,
                    "second_image_id": second.image_id,
                    "first_object_ids": sorted(first_ids),
                    "first_latencies_ms": first.latencies.model_dump(),
                    "second_latencies_ms": second.latencies.model_dump(),
                    "rows": rows,
                    "FIRST_RUN": [
                        {
                            "object_id": m.object_id,
                            "new_object": m.is_new,
                            "class": m.class_name,
                        }
                        for m in first.matches
                    ],
                    "SECOND_RUN": [
                        {
                            "object_id": m.object_id,
                            "new_object": m.is_new,
                            "class": m.class_name,
                            "matched_existing_object": m.matched_existing_object,
                        }
                        for m in second.matches
                    ],
                },
                indent=2,
            )
        )
        return 0 if ok else 1
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
