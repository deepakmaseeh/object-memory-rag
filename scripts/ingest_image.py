#!/usr/bin/env python
"""Ingest an image into immutable raw storage (no perception)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.image_ingestor import ImageIngestor


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest an image")
    parser.add_argument("path", type=str, help="Path to image file")
    parser.add_argument("--scene-id", type=str, default=None)
    args = parser.parse_args()

    ingestor = ImageIngestor()
    record = ingestor.ingest_path(args.path, scene_id=args.scene_id)
    print(json.dumps(record.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
