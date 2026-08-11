#!/usr/bin/env python
"""Run perception only (YOLO + SAM) on an image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.image_ingestor import ImageIngestor
from app.perception.pipeline import PerceptionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run perception pipeline")
    parser.add_argument("path", type=str, help="Path to image file")
    args = parser.parse_args()

    ingestor = ImageIngestor()
    image = ingestor.ingest_path(args.path)
    pipe = PerceptionPipeline()
    observations = pipe.process_image_path(image.original_path, image.image_id)
    print(
        json.dumps(
            {
                "image_id": image.image_id,
                "count": len(observations),
                "observations": [o.model_dump(mode="json") for o in observations],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
