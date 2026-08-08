"""Cross-KB knowledge clone — standalone module.

``clone_knowledge`` deep-copies one knowledge item (its ``documents``
row plus every chunk row) from a source knowledge base into a target
knowledge base, mirroring the upstream clone semantics:

- Only ``completed`` items are cloned; any other parse status is
  skipped and ``None`` is returned.
- The destination row is a fresh UUID stamped with the target knowledge
  base's tenant and embedding model; parse status starts
  ``processing`` and is settled to ``completed`` / ``failed`` once the
  chunk copy finishes.
- Every chunk is re-keyed with a new UUID; ``pre`` / ``next`` / ``parent``
  relationships are remapped onto the new ids, and per-chunk tags are
  re-created in the target knowledge base by name (the unclassified tag
  is pinned to the front of the list, mirroring the tag contract).
- Extracted images are deep-copied into destination-owned objects so
  deleting the source never breaks the clone; the object copy is an
  injected ``ObjectCopier`` hook.

Storage and retrieval seams
---------------------------

The storage object copy (document file + extracted chunk images) is
injected as ``ObjectCopier``; the retrieval-index copy is injected as
``VectorIndexReplicator``. Until those domains land:

- A source that carries a stored file or extracted images REQUIRES an
  object copier — cloning without one would leave the destination
  pointing at source-owned objects (shared references that deleting
  the source would destroy), so the missing hook is rejected with
  ``DataError``.
- The retrieval-index copy is optional: without it the cloned chunks
  land without retrieval indices (recoverable by re-indexing), and the
  hook is simply not invoked.
"""

from __future__ import annotations

import json
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import DataError, NotFoundError, ValidationError
from src.common.json import JsonValue
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PROCESSING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.tags.types import UNTAGGED_TAG_NAME
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.knowledge_tag import KnowledgeTag

_NOT_FOUND_CODE = "knowledge.not_found"

# Chunk create batch, matching the upstream paging/chunking of a clone.
_CLONE_CHUNK_BATCH = 100

# Disabled while a clone is in flight; enabled once it completes.
_ENABLE_STATUS_DISABLED = "disabled"
_ENABLE_STATUS_ENABLED = "enabled"


class ObjectCopier(Protocol):
    """Deep-copy a stored object into destination-owned storage.

    Mirrors the upstream ``copyOwnedObject``: read the source bytes and
    re-save them as a NEW object owned by ``(dst_tenant_id,
    dst_knowledge_id)``, returning the new object's path. Used for both
    the source document file and extracted chunk images.
    """

    async def copy_object(
        self,
        *,
        src_path: str,
        dst_tenant_id: int,
        dst_knowledge_id: str,
    ) -> str:
        """Return the destination path of the copied object."""


class VectorIndexReplicator(Protocol):
    """Copy retrieval-index entries from one knowledge base to another."""

    async def copy_indices(
        self,
        *,
        source_kb_id: str,
        target_kb_id: str,
        knowledge_id_map: dict[str, str],
        chunk_id_map: dict[str, str],
    ) -> None:
        """Copy index entries so the cloned chunks surface in the target."""


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_non_empty(value: str, *, code: str, message: str) -> None:
    """Reject a blank string at the service boundary."""
    if not value.strip():
        raise ValidationError(code=code, message=message)


def _require_knowledge_id(id: str) -> None:
    """Reject a blank knowledge id at the service boundary."""
    _require_non_empty(id, code="knowledge.id_required", message="knowledge ID is required")


def _to_knowledge(row: Document) -> Knowledge:
    """Project a persisted ``documents`` row onto the wire shape."""
    return Knowledge(
        id=row.id,
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        type=row.type,
        title=row.title,
        description=row.description,
        source=row.source,
        channel=row.channel,
        summary_status=row.summary_status,
        parse_status=row.parse_status,
        enable_status=row.enable_status,
        embedding_model_id=row.embedding_model_id,
        file_name=row.file_name,
        file_type=row.file_type,
        file_size=row.file_size,
        file_hash=row.file_hash,
        file_path=row.file_path,
        storage_size=row.storage_size,
        metadata=row.metadata,
        created_at=row.created_at,
        updated_at=row.updated_at,
        processed_at=row.processed_at,
        error_message=row.error_message,
        deleted_at=row.deleted_at,
    )


async def _resolve_target_tag(
    *,
    tag_repo: TagRepository,
    src_tenant_id: int,
    dst_tenant_id: int,
    dst_kb_id: str,
    src_tag_id: str,
    cache: dict[str, str],
) -> str:
    """Map a source tag into the target knowledge base by name.

    Mirrors the upstream tag re-homing: look up the source tag, reuse an
    existing tag with the same name in the target, else create a new one
    with the same properties (the unclassified tag is pinned to the
    front of the list). Results are cached so repeated chunks share one
    tag row; an absent source tag resolves to no tag (``""``).
    """
    cached = cache.get(src_tag_id)
    if cached is not None:
        return cached
    src_tag = await tag_repo.get_by_id(src_tenant_id, src_tag_id)
    if src_tag is None:
        cache[src_tag_id] = ""
        return ""
    dst_tag = await tag_repo.get_by_name(dst_tenant_id, dst_kb_id, src_tag.name)
    if dst_tag is not None:
        cache[src_tag_id] = dst_tag.id
        return dst_tag.id
    now = datetime.now(UTC)
    new_tag = KnowledgeTag(
        id=str(uuid.uuid4()),
        tenant_id=dst_tenant_id,
        knowledge_base_id=dst_kb_id,
        name=src_tag.name,
        color=src_tag.color,
        sort_order=-1 if src_tag.name == UNTAGGED_TAG_NAME else src_tag.sort_order,
        created_at=now,
        updated_at=now,
    )
    await tag_repo.create(new_tag)
    cache[src_tag_id] = new_tag.id
    return new_tag.id


async def _clone_image_info(
    *,
    image_info: str,
    object_copier: ObjectCopier,
    dst_tenant_id: int,
    dst_knowledge_id: str,
    url_cache: dict[str, str],
) -> tuple[str, list[str]]:
    """Deep-copy every object referenced by a chunk's ``image_info`` JSON.

    Returns the re-serialized ``image_info`` and the list of newly
    created object paths. ``url_cache`` dedups identical source objects
    across chunks and accumulates the old -> new mapping so content
    image URLs can be rewritten in a final pass. A JSON parse failure
    fails the clone rather than inheriting a shared reference.
    """
    try:
        parsed: JsonValue = json.loads(image_info)
    except json.JSONDecodeError as exc:
        raise DataError(
            code="knowledge.clone_image_info_invalid",
            message="failed to parse chunk image_info JSON",
        ) from exc
    if not isinstance(parsed, list):
        raise DataError(
            code="knowledge.clone_image_info_invalid",
            message="chunk image_info must be a JSON list",
        )
    images = parsed
    copied_urls: list[str] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        original_matched = item.get("original_url") == url
        new_url = url_cache.get(url)
        if new_url is None:
            new_url = await object_copier.copy_object(
                src_path=url,
                dst_tenant_id=dst_tenant_id,
                dst_knowledge_id=dst_knowledge_id,
            )
            url_cache[url] = new_url
            copied_urls.append(new_url)
        if original_matched:
            item["original_url"] = new_url
        item["url"] = new_url
    return json.dumps(images), copied_urls


def _rewrite_content_image_urls(content: str, url_cache: dict[str, str]) -> str:
    """Rewrite in-content Markdown image URLs to their copied objects.

    Replacements run longest-old-URL first so a URL that is a prefix of
    another is not partially rewritten. Entries whose old == new are
    skipped.
    """
    if not content or not url_cache:
        return content
    old_urls = sorted(
        (old for old, new in url_cache.items() if old and old != new),
        key=len,
        reverse=True,
    )
    for old in old_urls:
        content = content.replace(old, url_cache[old])
    return content


def _remap_relation(chunk_id: str | None, mapping: dict[str, str]) -> str:
    """Map a chunk relationship to the destination id, else ``""``."""
    if chunk_id is None:
        return ""
    return mapping.get(chunk_id, "")


async def clone_knowledge(
    *,
    tenant_id: int,
    knowledge_id: str,
    target_kb_id: str,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    tag_repo: TagRepository,
    kb_service: KBService,
    object_copier: ObjectCopier | None = None,
    index_replicator: VectorIndexReplicator | None = None,
) -> Knowledge | None:
    """Clone one knowledge item into ``target_kb_id``, returning its shape.

    Returns ``None`` when the source item is not ``completed`` (the
    upstream clone gate skips items still being processed). A source
    that carries a stored file or extracted images requires
    ``object_copier``; without it the clone raises ``DataError`` rather
    than persisting shared storage references.
    """
    _require_tenant_id(tenant_id)
    _require_knowledge_id(knowledge_id)
    _require_non_empty(
        target_kb_id,
        code="knowledge.kb_required",
        message="target knowledge base ID is required",
    )

    source = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if source is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")

    if source.parse_status != PARSE_STATUS_COMPLETED:
        return None

    target_kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=target_kb_id)

    dst_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    # Deep-copy the source document file into destination-owned storage so
    # deleting the source never destroys the clone's file. This runs before
    # the row is created, so a failure leaves no partial destination row.
    dst_file_path = source.file_path
    if source.file_path:
        if object_copier is None:
            raise DataError(
                code="knowledge.clone_storage_hook_required",
                message="an object copier is required to clone a stored file",
            )
        dst_file_path = await object_copier.copy_object(
            src_path=source.file_path,
            dst_tenant_id=target_kb.tenant_id,
            dst_knowledge_id=dst_id,
        )

    dst = Document(
        id=dst_id,
        tenant_id=target_kb.tenant_id,
        knowledge_base_id=target_kb.id,
        type=source.type,
        title=source.title,
        description=source.description,
        source=source.source,
        channel=source.channel,
        parse_status=PARSE_STATUS_PROCESSING,
        summary_status=source.summary_status,
        enable_status=_ENABLE_STATUS_DISABLED,
        embedding_model_id=target_kb.embedding_model_id,
        file_name=source.file_name,
        file_type=source.file_type,
        file_size=source.file_size,
        file_hash=source.file_hash,
        file_path=dst_file_path,
        storage_size=source.storage_size,
        metadata=source.metadata,
        custom_metadata={},
        created_at=now,
        updated_at=now,
    )

    created = False
    try:
        await knowledge_repo.create(dst)
        created = True

        source_chunks = await chunk_repo.find_all_by_column_values(
            {"tenant_id": tenant_id, "knowledge_id": knowledge_id},
        )
        url_cache: dict[str, str] = {}
        tag_cache: dict[str, str] = {}
        src_to_dst: dict[str, str] = {}
        dst_chunks: list[Chunk] = []
        for src_chunk in source_chunks:
            target_tag_id = ""
            if src_chunk.tag_id:
                target_tag_id = await _resolve_target_tag(
                    tag_repo=tag_repo,
                    src_tenant_id=tenant_id,
                    dst_tenant_id=target_kb.tenant_id,
                    dst_kb_id=target_kb.id,
                    src_tag_id=src_chunk.tag_id,
                    cache=tag_cache,
                )

            new_image_info = src_chunk.image_info
            if src_chunk.image_info:
                if object_copier is None:
                    raise DataError(
                        code="knowledge.clone_storage_hook_required",
                        message="an object copier is required to clone chunk images",
                    )
                new_image_info, _copied = await _clone_image_info(
                    image_info=src_chunk.image_info,
                    object_copier=object_copier,
                    dst_tenant_id=target_kb.tenant_id,
                    dst_knowledge_id=dst_id,
                    url_cache=url_cache,
                )

            dst_chunk_id = str(uuid.uuid4())
            src_to_dst[src_chunk.id] = dst_chunk_id
            dst_chunks.append(
                Chunk(
                    id=dst_chunk_id,
                    tenant_id=target_kb.tenant_id,
                    knowledge_base_id=target_kb.id,
                    knowledge_id=dst_id,
                    tag_id=target_tag_id or None,
                    content=src_chunk.content,
                    chunk_index=src_chunk.chunk_index,
                    is_enabled=src_chunk.is_enabled,
                    start_at=src_chunk.start_at,
                    end_at=src_chunk.end_at,
                    chunk_type=src_chunk.chunk_type,
                    metadata=src_chunk.metadata,
                    content_hash=src_chunk.content_hash,
                    image_info=new_image_info,
                    status=src_chunk.status,
                    flags=src_chunk.flags,
                    created_at=now,
                    updated_at=now,
                )
            )

        # Content image-URL rewriting and relationship remapping need the
        # complete old -> new mapping, so they run as a final pass over all
        # cloned chunks once every source chunk has been processed.
        final_chunks: list[Chunk] = []
        for src_chunk, dst_chunk in zip(source_chunks, dst_chunks, strict=True):
            final_chunks.append(
                dst_chunk.model_copy(
                    update={
                        "content": _rewrite_content_image_urls(dst_chunk.content, url_cache),
                        "pre_chunk_id": _remap_relation(src_chunk.pre_chunk_id, src_to_dst),
                        "next_chunk_id": _remap_relation(src_chunk.next_chunk_id, src_to_dst),
                        "parent_chunk_id": _remap_relation(src_chunk.parent_chunk_id, src_to_dst),
                    }
                )
            )
        for offset in range(0, len(final_chunks), _CLONE_CHUNK_BATCH):
            await chunk_repo.create_many(final_chunks[offset : offset + _CLONE_CHUNK_BATCH])

        replicator = index_replicator
        if replicator is not None:
            await replicator.copy_indices(
                source_kb_id=source.knowledge_base_id,
                target_kb_id=target_kb.id,
                knowledge_id_map={source.id: dst_id},
                chunk_id_map=src_to_dst,
            )

        settled = await knowledge_repo.update(
            dst.model_copy(
                update={
                    "parse_status": PARSE_STATUS_COMPLETED,
                    "enable_status": _ENABLE_STATUS_ENABLED,
                    "updated_at": now,
                }
            )
        )
        return _to_knowledge(settled)
    except Exception as exc:
        # A failed clone is recorded on the destination row instead of being
        # silently dropped, mirroring the upstream failure settle. If that
        # settle also fails, the original error wins.
        if created:
            with suppress(Exception):
                await knowledge_repo.update(
                    dst.model_copy(
                        update={
                            "parse_status": PARSE_STATUS_FAILED,
                            "error_message": str(exc),
                            "updated_at": now,
                        }
                    )
                )
        raise


__all__ = [
    "ObjectCopier",
    "VectorIndexReplicator",
    "clone_knowledge",
]
