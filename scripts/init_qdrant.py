#!/usr/bin/env python
"""Initialize Qdrant collections."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.memory.qdrant_store import QdrantVectorStore


def main() -> int:
    settings = get_settings()
    store = QdrantVectorStore(settings)
    store.ensure_collections()
    print(
        f"Qdrant collections ready: "
        f"{settings.qdrant.collections.observations}, "
        f"{settings.qdrant.collections.clusters}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
