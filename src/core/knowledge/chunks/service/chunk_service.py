"""Chunk domain service — CRUD and the document chunk edit.

Request-scoped service over the ``chunks`` repository: batch create,
tenant-scoped reads, full-row update, batch and knowledge-scoped soft
delete, and the optimistic, revision-guarded document edit.

The edit path adds the one guard the repository does not own — a manual
edit must not introduce image URLs absent from the immutable source —
then delegates the atomic write to ``ChunkRepository.update_document_chunk``
(which enforces text-only chunks, non-empty trimmed content, the byte
limit, and the optimistic revision check) and settles the retrieval-index
lifecycle the repository leaves open: ``processing`` -> ``ready`` on a
successful re-embedding, ``processing`` -> ``failed`` when re-embedding
fails so the accepted edit stays saved and visible instead of presenting
a false success state.

The actual re-embedding is an injected ``ChunkIndexSyncer`` hook rather
than a hard dependency: the retrieval engine lands in a later wave, and
until it is wired the service settles ``index_status`` without it.

The paged/filtered listing (``ListPagedChunksByKnowledgeID``) and the
revision-history / generated-question operations are not part of this
service: the former needs a paged repository query that lands with the
chunk-list wiring, and the latter live with the revision and question
domains.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonValue
from src.core.contracts.knowledge import Chunk as ChunkContract
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk

# Retrieval-index lifecycle states the service settles after an edit.
_INDEX_STATUS_PROCESSING = "processing"
_INDEX_STATUS_READY = "ready"
_INDEX_STATUS_FAILED = "failed"

# Matches Markdown image links: ![alt](url).
_MARKDOWN_IMAGE_PATTERN: re.Pattern[str] = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Matches an HTML <img> tag with a quoted src attribute. ``src`` must be
# preceded by whitespace so hyphenated attribute names like ``data-src``
# are not mistaken for it. The src value is submatch group 2.
_HTML_IMG_SRC_PATTERN: re.Pattern[str] = re.compile(
    r"(?i)<img\b([^>]*?)\ssrc\s*=\s*['\"]([^'\"]+)['\"]([^>]*)>",
)


class ChunkIndexSyncer(Protocol):
    """Re-embedding hook that keeps the retrieval index in sync after an edit."""

    async def sync_chunk(self, *, tenant_id: int, chunk: Chunk) -> None:
        """Re-index ``chunk``'s current content in the retrieval store."""


def image_urls_in_content(content: str) -> set[str]:
    """Return the image URLs referenced by ``content``.

    Covers Markdown image links and HTML ``<img src=...>`` tags, mirroring
    the upstream image-URL extraction the edit guard relies on.
    """
    urls: set[str] = set()
    for match in _MARKDOWN_IMAGE_PATTERN.finditer(content):
        if match.group(2):
            urls.add(match.group(2))
    for match in _HTML_IMG_SRC_PATTERN.finditer(content):
        src = match.group(2).strip()
        if src:
            urls.add(src)
    return urls


def _string_list(value: JsonValue | None) -> list[str] | None:
    """Narrow a JSONB relation list to the wire ``list[str]`` shape.

    The relation columns store raw JSON; the wire contract narrows them to
    string lists, so a scalar or absent value projects as ``None``.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return None


def chunk_to_contract(row: Chunk) -> ChunkContract:
    """Project a storage chunk row onto the frozen wire contract.

    The contract carries the documented wire fields; internal retrieval
    bookkeeping columns (``seq_id``, ``content_revision``,
    ``index_status``, ``last_editor_id``, ``flags``) stay off the wire,
    mirroring the documented API response.
    """
    return ChunkContract(
        id=row.id,
        tenant_id=row.tenant_id,
        knowledge_id=row.knowledge_id,
        knowledge_base_id=row.knowledge_base_id,
        tag_id=row.tag_id,
        content=row.content,
        chunk_index=row.chunk_index,
        is_enabled=row.is_enabled,
        status=row.status,
        start_at=row.start_at,
        end_at=row.end_at,
        pre_chunk_id=row.pre_chunk_id,
        next_chunk_id=row.next_chunk_id,
        chunk_type=row.chunk_type,
        parent_chunk_id=row.parent_chunk_id,
        relation_chunks=_string_list(row.relation_chunks),
        indirect_relation_chunks=_string_list(row.indirect_relation_chunks),
        metadata=row.metadata,
        content_hash=row.content_hash,
        image_info=row.image_info,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )


def validate_edited_chunk_images(source_content: str, edited_content: str) -> None:
    """Reject an edit that adds an image URL absent from the immutable source.

    ``source_content`` is the parser output the chunk was first created
    from; a manual edit may keep or reorder existing images but cannot
    introduce new ones.
    """
    allowed = image_urls_in_content(source_content)
    for url in sorted(image_urls_in_content(edited_content)):
        if url not in allowed:
            raise ValidationError(
                code="chunk.image_add_unsupported",
                message=f"adding images to an existing chunk is not supported: {url}",
            )


class ChunkService:
    """Chunk operations over the ``chunks`` table, constructed per request."""

    def __init__(
        self,
        *,
        chunk_repo: ChunkRepository,
        index_syncer: ChunkIndexSyncer | None = None,
    ) -> None:
        self._chunk_repo = chunk_repo
        self._index_syncer = index_syncer

    # ── Create ──────────────────────────────────────────────────────

    async def create_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        """Persist a batch of chunks, returning the stored rows."""
        return await self._chunk_repo.create_many(chunks)

    # ── Read ────────────────────────────────────────────────────────

    async def get_chunk_by_id(self, *, tenant_id: int, id: str) -> Chunk:
        """Return one live chunk, tenant-scoped; raise when absent."""
        return await self._chunk_repo.get_by_id(tenant_id, id)

    async def get_chunk_by_id_only(self, *, id: str) -> Chunk:
        """Return one live chunk by id without a tenant filter.

        Used for cross-tenant permission resolution before the caller
        narrows to the owning tenant; raises when absent.
        """
        row = await self._chunk_repo.get_by_id_only(id)
        if row is None:
            raise NotFoundError(
                code="chunk.not_found",
                message=f"chunk {id} not found",
            )
        return row

    async def list_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> list[Chunk]:
        """Return the knowledge item's text chunks in document order."""
        return await self._chunk_repo.list_by_knowledge_id(tenant_id, knowledge_id)

    async def list_chunk_by_parent_id(
        self,
        *,
        tenant_id: int,
        parent_id: str,
    ) -> list[Chunk]:
        """Return the live chunks whose ``parent_chunk_id`` matches."""
        return await self._chunk_repo.list_by_parent_id(tenant_id, parent_id)

    # ── Update ──────────────────────────────────────────────────────

    async def update_chunk(self, *, chunk: Chunk) -> Chunk:
        """Overwrite every mutable column of the chunk, returning the result."""
        return await self._chunk_repo.update(chunk)

    async def update_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        """Update chunks in batch; an empty list is a no-op."""
        if not chunks:
            return []
        return [await self._chunk_repo.update(chunk) for chunk in chunks]

    async def update_document_chunk(
        self,
        *,
        tenant_id: int,
        chunk_id: str,
        content: str | None = None,
        is_enabled: bool | None = None,
        expected_revision: int | None = None,
        last_editor_id: str,
    ) -> Chunk:
        """Apply an optimistic, versioned document edit and settle its index.

        The repository owns the atomic write and its validation (text-only
        chunks, non-empty trimmed content, the byte limit, and the
        optimistic revision guard). This method adds the edit-image guard,
        resolves a missing ``expected_revision`` to the current revision
        (the upstream ``*int`` semantics: ``None`` skips the client-side
        staleness check while the write's WHERE guard still rejects a
        concurrent edit), and transitions ``index_status`` through the
        re-embedding hook. A failed re-embedding leaves the row saved with
        ``index_status=failed`` so the caller never sees a false success.
        """
        current = await self._chunk_repo.get_by_id(tenant_id, chunk_id)
        if content is not None:
            trimmed = content.strip()
            if trimmed != current.content:
                source_content = current.source_content or current.content
                validate_edited_chunk_images(source_content, trimmed)
        now = datetime.now(UTC)
        updated = await self._chunk_repo.update_document_chunk(
            tenant_id=tenant_id,
            chunk_id=chunk_id,
            content=content,
            is_enabled=is_enabled,
            expected_revision=(
                current.content_revision if expected_revision is None else expected_revision
            ),
            last_editor_id=last_editor_id,
            now=now,
        )
        if updated.index_status == _INDEX_STATUS_PROCESSING:
            # A change was persisted; re-embed, then settle the status.
            return await self._settle_index(tenant_id=tenant_id, chunk=updated, now=now)
        if updated.index_status == _INDEX_STATUS_FAILED:
            # No-op edit on a row that still reports a failed re-index:
            # retry the re-embedding so the row returns to a consistent state.
            return await self._settle_index(tenant_id=tenant_id, chunk=updated, now=now)
        return updated

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_chunk(self, *, tenant_id: int, id: str) -> bool:
        """Soft-delete one chunk; return whether a live row was removed."""
        return await self._chunk_repo.soft_delete(
            tenant_id=tenant_id,
            id=id,
            now=datetime.now(UTC),
        )

    async def delete_chunks(self, *, tenant_id: int, ids: list[str]) -> int:
        """Soft-delete chunks in batch; return the number of rows removed."""
        if not ids:
            return 0
        now = datetime.now(UTC)
        affected = 0
        for chunk_id in ids:
            if await self._chunk_repo.soft_delete(tenant_id=tenant_id, id=chunk_id, now=now):
                affected += 1
        return affected

    async def delete_chunks_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> int:
        """Soft-delete every live chunk of a knowledge item; return the count."""
        return await self._chunk_repo.delete_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            now=datetime.now(UTC),
        )

    async def delete_by_knowledge_list(self, *, tenant_id: int, ids: list[str]) -> int:
        """Soft-delete chunks under several knowledge items; return the count."""
        if not ids:
            return 0
        now = datetime.now(UTC)
        total = 0
        for knowledge_id in ids:
            total += await self._chunk_repo.delete_by_knowledge_id(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                now=now,
            )
        return total

    # ── Index lifecycle ─────────────────────────────────────────────

    async def _settle_index(self, *, tenant_id: int, chunk: Chunk, now: datetime) -> Chunk:
        """Run the re-embedding hook, then settle ``index_status``.

        The row is already committed with ``index_status=processing``; the
        hook re-indexes the current content. On failure the row is marked
        ``failed`` and returned — the edit is preserved and the caller sees
        the truth instead of an exception.
        """
        syncer = self._index_syncer
        if syncer is not None:
            try:
                await syncer.sync_chunk(tenant_id=tenant_id, chunk=chunk)
            except Exception:
                return await self._mark_index_status(chunk, _INDEX_STATUS_FAILED, now)
        return await self._mark_index_status(chunk, _INDEX_STATUS_READY, now)

    async def _mark_index_status(self, chunk: Chunk, status: str, now: datetime) -> Chunk:
        """Persist ``index_status`` on the row, returning the stored row."""
        if chunk.index_status == status:
            return chunk
        return await self._chunk_repo.update(
            chunk.model_copy(update={"index_status": status, "updated_at": now}),
        )


__all__ = [
    "ChunkIndexSyncer",
    "ChunkService",
    "chunk_to_contract",
    "image_urls_in_content",
    "validate_edited_chunk_images",
]
