from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import numpy as np
from PIL import Image

from app.config import Settings, get_settings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class BlobStore:
    """Filesystem blob storage with immutable raw assets and metadata sidecars."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.raw_dir = self.settings.resolve_path(self.settings.paths.raw)
        self.processed_dir = self.settings.resolve_path(self.settings.paths.processed)
        self.crops_dir = self.settings.resolve_path(self.settings.paths.crops)
        self.masks_dir = self.settings.resolve_path(self.settings.paths.masks)
        self.embeddings_dir = self.settings.resolve_path(self.settings.paths.embeddings)
        self.objects_dir = self.settings.resolve_path(self.settings.paths.objects)

    def save_image_derivative(
        self,
        image_id: str,
        kind: str,
        image: Union[Image.Image, np.ndarray, Path, str],
        suffix: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        ext = suffix or (".png" if kind == "transparent" else ".jpg")
        dest = self.processed_dir / f"{image_id}_{kind}{ext}"
        if ext == ".png" and isinstance(image, Image.Image) and image.mode == "RGBA":
            image.save(dest)
        else:
            self._write_image(dest, image)
        self._write_meta(
            dest,
            {
                "image_id": image_id,
                "path": str(dest),
                "kind": kind,
                **(meta or {}),
            },
        )
        return dest

    def save_object_derivative(
        self,
        object_id: str,
        kind: str,
        image: Union[Image.Image, np.ndarray, Path, str],
        suffix: str = ".png",
        meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        obj_dir = self.objects_dir / object_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        dest = obj_dir / f"{kind}{suffix}"
        if suffix == ".png" and isinstance(image, Image.Image) and image.mode == "RGBA":
            image.save(dest)
        else:
            self._write_image(dest, image)
        self._write_meta(
            dest,
            {
                "object_id": object_id,
                "path": str(dest),
                "kind": kind,
                **(meta or {}),
            },
        )
        return dest

    def get_image_derivative_path(self, image_id: str, kind: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.processed_dir / f"{image_id}_{kind}{ext}"
            if p.exists():
                return p
        return None

    def get_object_derivative_path(self, object_id: str, kind: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.objects_dir / object_id / f"{kind}{ext}"
            if p.exists():
                return p
        return None

    def save_raw(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        image_id: Optional[str] = None,
        suffix: str = ".jpg",
        meta: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Path]:
        image_id = image_id or new_id("img")
        dest = self.raw_dir / f"{image_id}{suffix}"
        if dest.exists():
            raise FileExistsError(f"Raw asset already exists (immutable): {dest}")

        if isinstance(source, (str, Path)):
            shutil.copy2(str(source), dest)
        elif isinstance(source, bytes):
            dest.write_bytes(source)
        else:
            data = source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            dest.write_bytes(data)

        sidecar = {
            "image_id": image_id,
            "original_path": str(dest),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        }
        self._write_meta(dest, sidecar)
        return image_id, dest

    def save_processed(
        self,
        image_id: str,
        image: Union[Image.Image, np.ndarray, Path, str],
        suffix: str = ".jpg",
        meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        dest = self.processed_dir / f"{image_id}{suffix}"
        self._write_image(dest, image)
        self._write_meta(
            dest,
            {
                "image_id": image_id,
                "path": str(dest),
                "kind": "processed",
                **(meta or {}),
            },
        )
        return dest

    def save_crop(
        self,
        observation_id: str,
        crop: Union[Image.Image, np.ndarray, Path, str],
        suffix: str = ".jpg",
        meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        dest = self.crops_dir / f"{observation_id}{suffix}"
        if suffix.endswith(".png") and isinstance(crop, Image.Image) and crop.mode == "RGBA":
            crop.save(dest)
        else:
            self._write_image(dest, crop)
        self._write_meta(
            dest,
            {
                "observation_id": observation_id,
                "path": str(dest),
                "kind": "crop",
                **(meta or {}),
            },
        )
        return dest

    def save_mask(
        self,
        observation_id: str,
        mask: np.ndarray,
        meta: Optional[dict[str, Any]] = None,
    ) -> Path:
        dest = self.masks_dir / f"{observation_id}.png"
        # Store binary mask as 0/255 PNG
        arr = (mask.astype(bool).astype(np.uint8)) * 255
        Image.fromarray(arr, mode="L").save(dest)
        self._write_meta(
            dest,
            {
                "observation_id": observation_id,
                "path": str(dest),
                "kind": "mask",
                **(meta or {}),
            },
        )
        return dest

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _write_meta(asset_path: Path, meta: dict[str, Any]) -> None:
        meta_path = asset_path.with_suffix(asset_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _write_image(dest: Path, image: Union[Image.Image, np.ndarray, Path, str]) -> None:
        if isinstance(image, (str, Path)):
            shutil.copy2(str(image), dest)
            return
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                pil = Image.fromarray(image, mode="L")
            elif image.shape[2] == 4:
                pil = Image.fromarray(image, mode="RGBA").convert("RGB")
            else:
                # BGR from OpenCV is possible; assume RGB unless clearly not
                pil = Image.fromarray(image[..., :3])
            pil.save(dest, quality=95)
            return
        if isinstance(image, Image.Image):
            if dest.suffix.lower() == ".png" and image.mode == "RGBA":
                image.save(dest)
            else:
                image.convert("RGB").save(dest, quality=95)
            return
        raise TypeError(f"Unsupported image type: {type(image)}")
