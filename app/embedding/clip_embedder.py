from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.config import Settings, get_settings
from app.embedding.base import Embedder, ImageLike


class CLIPEmbedder(Embedder):
    """OpenCLIP ViT-B/32 visual encoder."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._preprocess = None
        self._device = self.settings.resolve_device()
        self._size = self.settings.embedding.vector_size

    @property
    def vector_size(self) -> int:
        return self._size

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip
        import torch

        device = self.settings.resolve_device()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.settings.models.embedder,
            pretrained=self.settings.models.embedder_pretrained,
            device=device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    def encode(self, crop: ImageLike) -> list[float]:
        import torch

        self._load()
        image = self._as_pil(crop)
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = feats.float()
            if self.settings.embedding.normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            vec = feats.cpu().numpy().reshape(-1)
        # Pad/truncate if open_clip dim != configured size
        if len(vec) != self._size:
            if len(vec) > self._size:
                vec = vec[: self._size]
            else:
                padded = np.zeros(self._size, dtype=np.float32)
                padded[: len(vec)] = vec
                vec = padded
        return [float(x) for x in vec.tolist()]

    @staticmethod
    def _as_pil(crop: ImageLike) -> Image.Image:
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")
        if isinstance(crop, np.ndarray):
            return Image.fromarray(crop).convert("RGB")
        return Image.open(crop).convert("RGB")
