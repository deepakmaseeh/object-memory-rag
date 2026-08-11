#!/usr/bin/env python
"""Process a single image with full latency breakdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.memory.memory_updater import PipelineService


def main() -> int:
    parser = argparse.ArgumentParser(description="Process image end-to-end with latencies")
    parser.add_argument("path", type=str, help="Path to image file")
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--location", type=str, default="Desk")
    parser.add_argument("--force-vlm", action="store_true")
    args = parser.parse_args()

    service = PipelineService()
    try:
        result = service.process_image_path(
            args.path,
            scene_id=args.scene_id,
            location_name=args.location,
            force_vlm=args.force_vlm,
        )
        summary = {
            "image_id": result.image_id,
            "detection_count": result.detection_count,
            "device": result.device,
            "models": result.models,
            "latencies_ms": result.latencies.model_dump(),
            "objects": [
                {
                    "class": m.class_name,
                    "confidence": m.confidence,
                    "object_id": m.object_id,
                    "observation_id": m.observation_id,
                    "matched_existing_object": m.matched_existing_object,
                    "is_new": m.is_new,
                    "cluster_id": m.cluster_id,
                    "similarity": m.similarity,
                }
                for m in result.matches
            ],
        }
        print(json.dumps(summary, indent=2))
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
