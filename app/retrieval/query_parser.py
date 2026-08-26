from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass
class ParsedQuery:
    query: str
    class_name: Optional[str] = None
    top_k: int = 5
    mode: str = "general"  # general | instance | product | brand | location | count


class QueryParser:
    """Keyword parser for class/product/instance/location intent."""

    def parse(self, query: str) -> MemoryQuery:
        parsed = self.parse_extended(query)
        return MemoryQuery(query=parsed.query, class_name=parsed.class_name, top_k=parsed.top_k)

    def parse_extended(self, query: str) -> ParsedQuery:
        q = query.strip()
        lower = q.lower()
        class_name = self._extract_class(lower)
        mode = self._extract_mode(lower)
        top_k = 10 if mode in {"product", "count"} else 5
        return ParsedQuery(query=q, class_name=class_name, top_k=top_k, mode=mode)

    def _extract_class(self, lower: str) -> Optional[str]:
        for alias, mapped in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{re.escape(alias)}\b", lower):
                return mapped
        for c in sorted(COMMON_CLASSES, key=lambda x: -len(x)):
            if re.search(rf"\b{re.escape(c)}\b", lower):
                return c
        return None

    @staticmethod
    def _extract_mode(lower: str) -> str:
        if re.search(r"\b(how many|count|number of)\b.*\b(product|brand|bottle|same)\b", lower):
            return "count"
        if re.search(
            r"\b(exact|this exact|same physical|physical instance|seen this .+ before)\b", lower
        ):
            return "instance"
        if re.search(r"\b(where did i last see|last seen|last location)\b", lower):
            return "location"
        if re.search(r"\b(which brand|what brand|brand is)\b", lower):
            return "brand"
        if re.search(
            r"\b(how many bottles|similar products|same product|product have i seen)\b", lower
        ):
            return "product"
        if re.search(r"\bwhat is this\b", lower):
            return "product"
        return "general"
