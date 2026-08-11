from __future__ import annotations

import re
from typing import Optional

from app.schemas import MemoryQuery


COMMON_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

# Map casual words to COCO class names
ALIASES = {
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "sofa": "couch",
    "tv": "tv",
    "television": "tv",
    "computer": "laptop",
    "table": "dining table",
    "plant": "potted plant",
    "people": "person",
    "persons": "person",
    "human": "person",
    "humans": "person",
}


class QueryParser:
    """Lightweight keyword parser for class/location extraction."""

    def parse(self, query: str) -> MemoryQuery:
        q = query.strip()
        lower = q.lower()
        class_name: Optional[str] = None

        # Prefer longer alias matches first
        for alias, mapped in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                class_name = mapped
                break

        if class_name is None:
            for c in sorted(COMMON_CLASSES, key=lambda x: -len(x)):
                if re.search(rf"\b{re.escape(c)}\b", lower):
                    class_name = c
                    break

        return MemoryQuery(query=q, class_name=class_name, top_k=5)
