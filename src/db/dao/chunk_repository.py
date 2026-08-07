"""Chunk persistence — raw SQL only, no ORM.

Covers the ``chunks`` table: batch create, tenant-scoped reads, full-row
update, an optimistic revision-checked document edit, soft delete, and
the small aggregate queries the knowledge base relies on. Reads filter
soft-deleted rows (``deleted_at is null``) unless the caller opts out.

``update_document_chunk`` mirrors the write side of the chunk-edit path:
the row is loaded tenant-scoped, the edit is validated, and the UPDATE is
guarded by ``content_revision`` so a stale edit fails with a revision
conflict instead of clobbering a newer revision. The caller owns the
retrieval-index side effects (transitioning ``index_status`` through
``processing``/``ready``/``failed`` after the write).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult
from sqlalchemy.sql.elements import BindParameter

from src.common.exception import ConflictError, DataError, NotFoundError, ValidationError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.chunk import Chunk

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")

_TABLE_NAME = "chunks"

# The only editable chunk type; parent / image / FAQ rows are managed by
# their owning pipelines and must not be edited directly.
_CHUNK_TYPE_TEXT = "text"

# Longest editable document body in bytes (mirrors the chunk-edit
# service guard); edits beyond this are rejected up front.
_MAX_EDITABLE_CHUNK_LENGTH = 200_000

# Optimistic-edit marker written before retrieval re-indexing runs.
_INDEX_STATUS_PROCESSING = "processing"

# Columns a full-row update must never touch: identity, the DB-assigned
# auto-increment, and the insert timestamp.
_IMMUTABLE_UPDATE_COLUMNS: frozenset[str] = frozenset({"id", "tenant_id", "seq_id", "created_at"})


class ChunkRepository(GenericRepository[Chunk]):
    """`chunks`-table SQL — CRUD, bulk create, and the guarded edit."""

    model_class = Chunk

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Chunk) -> Chunk:
        """Insert one chunk; the application supplies the UUID ``id``."""
        return await self.insert(row)

    async def create_many(self, rows: list[Chunk]) -> list[Chunk]:
        """Insert many chunks in a single statement, returning the rows.

        ``seq_id`` is DB-assigned (excluded from the INSERT column list),
        so concurrent writers never contend on the sequence default. The
        JSONB columns are bound through the dialect-aware bind type.
        """
        if not rows:
            return []
        columns = self.model_class.insert_sql_column_list()
        json_cols = self._json_columns
        value_groups: list[str] = []
        params: BindParams = {}
        json_bps: list[BindParameter[SqlValue]] = []
        for i, row in enumerate(rows):
            group: list[str] = []
            for col in columns:
                p = f"{col}_{i}"
                group.append(f":{p}")
                params[p] = getattr(row, col)
                if col in json_cols:
                    json_bps.append(bindparam(p, type_=_JSON))
            value_groups.append(f"({', '.join(group)})")
        col_list = ", ".join(f'"{c}"' for c in columns)
        stmt_text = (
            f"insert into {_TABLE_NAME} ({col_list}) values {', '.join(value_groups)} returning *"
        )
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def update(self, row: Chunk) -> Chunk:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``seq_id`` / ``created_at`` are immutable
        by contract and stay out of the SET clause.
        """
        updates = {k: v for k, v in row.model_dump().items() if k not in _IMMUTABLE_UPDATE_COLUMNS}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="chunk.update_no_row",
                message=f"chunk {row.id} not found for update",
            )
        return persisted

    async def update_document_chunk(
        self,
        *,
        tenant_id: int,
        chunk_id: str,
        content: str | None,
        is_enabled: bool | None,
        expected_revision: int,
        last_editor_id: str,
        now: datetime,
    ) -> Chunk:
        """Apply an optimistic, revision-guarded document edit.

        Loads the row tenant-scoped, validates the edit (text chunks only;
        content non-empty after trim and within the byte limit), then writes
        atomically with ``content_revision = expected_revision`` in the
        WHERE clause. A concurrent edit that already advanced the revision
        raises ``ConflictError`` instead of silently overwriting it.

        ``content`` / ``is_enabled`` are optional: ``None`` keeps the
        current value. The write bumps ``content_revision``, records
        ``last_editor_id``, and marks ``index_status`` as ``processing``
        (the caller transitions it to ``ready`` / ``failed`` after the
        retrieval index is synced).
        """
        current = await self.get_by_id_or_none(tenant_id, chunk_id)
        if current is None:
            raise NotFoundError(
                code="chunk.not_found",
                message=f"chunk {chunk_id} not found",
            )
        if current.chunk_type != _CHUNK_TYPE_TEXT:
            raise ValidationError(
                code="chunk.not_editable",
                message=f"only text chunks can be edited (chunk {chunk_id} is {current.chunk_type})",
            )
        new_content = current.content
        if content is not None:
            new_content = content.strip()
            if not new_content:
                raise ValidationError(
                    code="chunk.content_empty",
                    message="chunk content cannot be empty",
                )
            if len(new_content.encode("utf-8")) > _MAX_EDITABLE_CHUNK_LENGTH:
                raise ValidationError(
                    code="chunk.content_too_long",
                    message=f"chunk content exceeds {_MAX_EDITABLE_CHUNK_LENGTH} bytes",
                )
        new_enabled = current.is_enabled
        if is_enabled is not None:
            new_enabled = is_enabled

        if new_content == current.content and new_enabled == current.is_enabled:
            return current

        if expected_revision != current.content_revision:
            raise ConflictError(
                code="chunk.revision_conflict",
                message=(
                    f"chunk {chunk_id} has changed since revision {expected_revision} "
                    f"(current: {current.content_revision})"
                ),
            )

        source_content = current.source_content
        if source_content == "":
            source_content = current.content
        new_revision = current.content_revision + 1
        params: BindParams = {
            "content": new_content,
            "source_content": source_content,
            "content_revision": new_revision,
            "is_enabled": new_enabled,
            "index_status": _INDEX_STATUS_PROCESSING,
            "last_editor_id": last_editor_id,
            "now": now,
            "id": chunk_id,
            "tenant_id": tenant_id,
            "expected_revision": expected_revision,
        }
        stmt_text = (
            f"update {_TABLE_NAME} set "
            "content = :content, "
            "source_content = :source_content, "
            "content_revision = :content_revision, "
            "is_enabled = :is_enabled, "
            "index_status = :index_status, "
            "last_editor_id = :last_editor_id, "
            "updated_at = :now "
            "where id = :id and tenant_id = :tenant_id "
            "and content_revision = :expected_revision returning *"
        )
        stmt = text(stmt_text).bindparams(**params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        # Defense in depth: the WHERE guard should already have rejected a
        # stale revision above, but a concurrent write between the load and
        # this UPDATE would surface here as a missing row.
        if mapping is None:
            raise ConflictError(
                code="chunk.revision_conflict",
                message=f"chunk {chunk_id} changed while the edit was being saved",
            )
        return self._hydrate(mapping)

    async def soft_delete(self, *, tenant_id: int, id: str, now: datetime) -> bool:
        """Mark a live chunk deleted. Returns whether a row was affected."""
        stmt = text(
            f"update {_TABLE_NAME} set deleted_at = :now, updated_at = :now "
            "where id = :id and tenant_id = :tenant_id and deleted_at is null"
        ).bindparams(id=id, tenant_id=tenant_id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def delete_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        now: datetime,
    ) -> int:
        """Soft-delete every live chunk of a knowledge item. Returns count."""
        stmt = text(
            f"update {_TABLE_NAME} set deleted_at = :now, updated_at = :now "
            "where tenant_id = :tenant_id and knowledge_id = :knowledge_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, knowledge_id=knowledge_id, now=now)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    async def move_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        target_kb_id: str,
    ) -> int:
        """Re-point every chunk of a knowledge item to another KB. Returns count."""
        stmt = text(
            f"update {_TABLE_NAME} set knowledge_base_id = :target_kb_id "
            "where tenant_id = :tenant_id and knowledge_id = :knowledge_id"
        ).bindparams(tenant_id=tenant_id, knowledge_id=knowledge_id, target_kb_id=target_kb_id)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_or_none(self, tenant_id: int, id: str) -> Chunk | None:
        """Return one live chunk by ``(tenant_id, id)``, or ``None``."""
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_id(self, tenant_id: int, id: str) -> Chunk:
        """Return one live chunk by ``(tenant_id, id)``, raising on absence."""
        row = await self.get_by_id_or_none(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code="chunk.not_found",
                message=f"chunk {id} not found",
            )
        return row

    async def get_by_id_only(self, id: str) -> Chunk | None:
        """Return one live chunk by ``id`` without a tenant filter.

        Used for cross-tenant permission resolution before the caller
        narrows to the owning tenant.
        """
        stmt = text(
            f"select * from {_TABLE_NAME} where id = :id and deleted_at is null"
        ).bindparams(id=id)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def get_by_seq_id(self, tenant_id: int, seq_id: int) -> Chunk | None:
        """Return the live chunk whose auto-increment ``seq_id`` matches."""
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "seq_id": seq_id},
        )

    async def list_by_ids(self, tenant_id: int, ids: list[str]) -> list[Chunk]:
        """Return the live chunks whose ids are in ``ids`` (tenant-scoped)."""
        if not ids:
            return []
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        params: BindParams = {f"id_{i}": v for i, v in enumerate(ids)}
        params["tenant_id"] = tenant_id
        stmt = text(
            f"select * from {_TABLE_NAME} "
            f"where tenant_id = :tenant_id and id in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_knowledge_id(
        self,
        tenant_id: int,
        knowledge_id: str,
    ) -> list[Chunk]:
        """Return the text chunks of a knowledge item in document order.

        Mirrors the upstream knowledge listing: only ``text`` chunks,
        ordered by ``chunk_index`` ascending.
        """
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and knowledge_id = :knowledge_id "
            "and chunk_type = :chunk_type and deleted_at is null "
            "order by chunk_index asc"
        ).bindparams(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            chunk_type=_CHUNK_TYPE_TEXT,
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_parent_id(self, tenant_id: int, parent_id: str) -> list[Chunk]:
        """Return the live chunks whose ``parent_chunk_id`` matches."""
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and parent_chunk_id = :parent_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, parent_id=parent_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_by_knowledge_base_id(
        self,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> int:
        """Count the live chunks of a knowledge base."""
        stmt = text(
            f"select count(*) from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and knowledge_base_id = :knowledge_base_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        total = result.scalar_one()
        return int(total) if total is not None else 0


__all__ = ["ChunkRepository"]
