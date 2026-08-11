#!/usr/bin/env python
"""Initialize Neo4j constraints and default scene."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.graph.neo4j_store import Neo4jGraphStore


def main() -> int:
    settings = get_settings()
    store = Neo4jGraphStore(settings)
    store.ensure_schema()
    print("Neo4j schema ready")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
