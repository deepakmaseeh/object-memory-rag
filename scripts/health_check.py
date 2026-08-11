#!/usr/bin/env python
"""Comprehensive health check for local runtime readiness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.health import HealthService


def main() -> int:
    settings = get_settings()
    load_models = "--models" in sys.argv
    report = HealthService(settings).check(load_models=load_models)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    # READY or DEGRADED -> success for local bootstrap (GPU optional)
    if report.status in {"READY", "DEGRADED"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
