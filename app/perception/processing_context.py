from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProcessingContext:
    remove_background: bool = False
    save_object_derivatives: bool = False
    object_id_prefix: Optional[str] = None
    transparent_preview_saved: bool = field(default=False, repr=False)
