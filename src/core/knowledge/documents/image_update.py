"""Standalone document image-info update.

Applies one image record to a document chunk's caption / OCR children:
the parent chunk's ``image_info`` is refreshed, child chunks whose
caption / OCR text changed are updated, and missing caption / OCR
children are created when the record carries the corresponding text.
The owning document's file hash is refreshed so downstream change
detection sees the edit. The retrieval-index refresh is out of scope
here (the re-embedding hook lands with the retrieval engine).

A no-op is returned when the payload carries anything other than
exactly one image record; child chunks whose stored image URL does not
match the record are skipped.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from hashlib import md5
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from src.common.exception import ValidationError
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
)
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk


class ImageInfo(BaseModel):
    """One image record carried in a chunk's ``image_info`` JSON list."""

    model_config = ConfigDict(frozen=True)

    url: str | None = None
    original_url: str | None = None
    start_pos: int = 0
    end_pos: int = 0
    caption: str = ""
    ocr_text: str = ""


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_id(value: str, *, code: str, message: str) -> None:
    """Reject a blank id at the service boundary."""
    if not value.strip():
        raise ValidationError(code=code, message=message)


def _parse_image_info(image_info: str) -> list[ImageInfo]:
    """Decode the ``image_info`` JSON list, raising on malformed input."""
    try:
        raw = json.loads(image_info)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            code="knowledge.invalid_image_info",
            message="image_info must be a JSON array of image records",
        ) from exc
    if not isinstance(raw, list):
        raise ValidationError(
            code="knowledge.invalid_image_info",
            message="image_info must be a JSON array of image records",
        )
    images: list[ImageInfo] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValidationError(
                code="knowledge.invalid_image_info",
                message="image_info entries must be JSON objects",
            )
        images.append(ImageInfo.model_validate(item))
    return images


def _child_images(image_info: str | None) -> list[ImageInfo]:
    """Parse a child chunk's stored image info, tolerating malformed rows."""
    if not image_info:
        return []
    try:
        return _parse_image_info(image_info)
    except ValidationError:
        return []


def _file_hash(*parts: str) -> str:
    """Return the MD5 hex digest of the concatenated parts."""
    return md5("".join(parts).encode("utf-8")).hexdigest()


def _new_child_chunk(
    parent: Chunk,
    image_info: str,
    content: str,
    chunk_type: str,
    now: datetime,
) -> Chunk:
    """Build a caption / OCR child chunk linked to the parent chunk."""
    return Chunk(
        id=str(uuid4()),
        tenant_id=parent.tenant_id,
        knowledge_base_id=parent.knowledge_base_id,
        knowledge_id=parent.knowledge_id,
        content=content,
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=0,
        chunk_type=chunk_type,
        parent_chunk_id=parent.id,
        image_info=image_info,
        created_at=now,
        updated_at=now,
    )


async def update_document_image(
    *,
    tenant_id: int,
    knowledge_id: str,
    chunk_id: str,
    image_info: str,
    knowledge_repo: KnowledgeRepository,
    chunk_service: ChunkService,
) -> None:
    """Apply one image record to a chunk's caption / OCR children.

    Raises ``ValidationError`` on malformed input and propagates the
    chunk repository's ``NotFoundError`` when the chunk is absent or out
    of the tenant scope. Returns without writing when the payload holds
    anything other than exactly one image record.
    """
    _require_tenant_id(tenant_id)
    _require_id(knowledge_id, code="knowledge.id_required", message="document ID is required")
    _require_id(chunk_id, code="knowledge.chunk_id_required", message="chunk ID is required")

    images = _parse_image_info(image_info)
    if len(images) != 1:
        return

    image = images[0]
    parent = await chunk_service.get_chunk_by_id(tenant_id=tenant_id, id=chunk_id)
    children = await chunk_service.list_chunk_by_parent_id(
        tenant_id=tenant_id,
        parent_id=chunk_id,
    )

    now = datetime.now(UTC)
    updates: list[Chunk] = [
        parent.model_copy(update={"image_info": image_info, "updated_at": now})
    ]
    creates: list[Chunk] = []
    has_caption = False
    has_ocr = False

    for child in children:
        child_images = _child_images(child.image_info)
        if not child_images:
            continue
        if child_images[0].original_url != image.original_url:
            continue
        if child.chunk_type == CHUNK_TYPE_IMAGE_CAPTION:
            has_caption = True
            if image.caption != child_images[0].caption:
                updates.append(
                    child.model_copy(
                        update={
                            "content": image.caption,
                            "image_info": image_info,
                            "updated_at": now,
                        }
                    )
                )
        elif child.chunk_type == CHUNK_TYPE_IMAGE_OCR:
            has_ocr = True
            if image.ocr_text != child_images[0].ocr_text:
                updates.append(
                    child.model_copy(
                        update={
                            "content": image.ocr_text,
                            "image_info": image_info,
                            "updated_at": now,
                        }
                    )
                )

    if not has_caption and image.caption:
        creates.append(
            _new_child_chunk(parent, image_info, image.caption, CHUNK_TYPE_IMAGE_CAPTION, now)
        )
    if not has_ocr and image.ocr_text:
        creates.append(
            _new_child_chunk(parent, image_info, image.ocr_text, CHUNK_TYPE_IMAGE_OCR, now)
        )

    if creates:
        await chunk_service.create_chunks(chunks=creates)
    if updates:
        await chunk_service.update_chunks(chunks=updates)

    row = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if row is not None:
        refreshed = row.model_copy(
            update={
                "file_hash": _file_hash(knowledge_id, row.file_hash or "", image_info),
                "updated_at": now,
            }
        )
        # Best-effort hash refresh; the chunk edits remain saved.
        with contextlib.suppress(Exception):
            await knowledge_repo.update(refreshed)
    return


__all__ = ["ImageInfo", "update_document_image"]
