from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union

from PIL import Image

from app.config import Settings, get_settings
from app.ingestion.storage import BlobStore, new_id
from app.schemas import ImageRecord, InputSource, utc_now


class ImageIngestor:
    """Preserve every input immutably and produce an ImageRecord."""

    def __init__(
        self,
        blob_store: Optional[BlobStore] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.blob_store = blob_store or BlobStore(self.settings)

    def ingest_path(
        self,
        path: Union[str, Path],
        image_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        extra_meta: Optional[dict] = None,
    ) -> ImageRecord:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower() or ".jpg"
        with Image.open(path) as im:
            width, height = im.size
            content_type = Image.MIME.get(im.format or "JPEG", "image/jpeg")

        meta = {
            "source": InputSource.IMAGE.value,
            "source_filename": path.name,
            "content_type": content_type,
            "width": width,
            "height": height,
            "scene_id": scene_id,
            **(extra_meta or {}),
        }
        image_id, dest = self.blob_store.save_raw(
            path, image_id=image_id or new_id("img"), suffix=suffix, meta=meta
        )
        return ImageRecord(
            image_id=image_id,
            original_path=str(dest),
            width=width,
            height=height,
            timestamp=utc_now(),
            content_type=content_type,
            meta=meta,
        )

    def ingest_bytes(
        self,
        data: bytes,
        filename: str = "upload.jpg",
        image_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> ImageRecord:
        from io import BytesIO

        suffix = Path(filename).suffix.lower() or ".jpg"
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            fmt = im.format or "JPEG"
            content_type = content_type or Image.MIME.get(fmt, "image/jpeg")

        meta = {
            "source": InputSource.IMAGE.value,
            "source_filename": filename,
            "content_type": content_type,
            "width": width,
            "height": height,
            "scene_id": scene_id,
        }
        image_id, dest = self.blob_store.save_raw(
            data, image_id=image_id or new_id("img"), suffix=suffix, meta=meta
        )
        return ImageRecord(
            image_id=image_id,
            original_path=str(dest),
            width=width,
            height=height,
            timestamp=utc_now(),
            content_type=content_type,
            meta=meta,
        )
