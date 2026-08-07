"""Temporary-document persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream temporary-document repository
interface. Every write is scoped by ``tenant_id`` (and ``session_id``
where the contract scopes by it) so a caller can never read or mutate
another workspace's rows.

Lifecycle mapping:

- ``create`` — upload leg (insert the ``uploaded`` row).
- ``mark_processing`` / ``mark_ready`` / ``mark_failed`` — pre-parse
  and promote legs (transition the row and clear the stale error).
- ``delete_scoped`` — soft-delete leg (``deleted_at`` tombstone).
- ``list_expired`` — the sweep read that backs expiry cleanup.

Reads filter ``deleted_at IS NULL`` so a soft-deleted row behaves as if
it no longer exists (the base ``GenericRepository`` helpers apply the
same filter on the ``find_*`` paths).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import JSON, CursorResult, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from src.common.json import JsonObject, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.temporary_document import (
    TEMPORARY_DOCUMENT_STATUS_FAILED,
    TEMPORARY_DOCUMENT_STATUS_PROCESSING,
    TEMPORARY_DOCUMENT_STATUS_READY,
    TemporaryDocument,
)

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

_LIVE = "deleted_at is null"


class TemporaryDocumentRepository(GenericRepository[TemporaryDocument]):
    """`temporary_documents`-table SQL — session-scoped CRUD + sweep."""

    model_class = TemporaryDocument

    # ── Create ───────────────────────────────────────────────────────

    async def create(self, row: TemporaryDocument) -> TemporaryDocument:
        """Insert a temporary-document row and return the persisted row."""
        return await self.insert(row)

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(
        self,
        *,
        tenant_id: int,
        document_id: str,
    ) -> TemporaryDocument | None:
        """Return the live row by primary key + tenant scope, or ``None``."""
        return await self.find_unique_by_column_values(
            {"id": document_id, "tenant_id": tenant_id},
        )

    async def get_scoped(
        self,
        *,
        tenant_id: int,
        session_id: str,
        document_id: str,
    ) -> TemporaryDocument | None:
        """Return the live row by (tenant, session, id), or ``None``."""
        return await self.find_unique_by_column_values(
            {"id": document_id, "tenant_id": tenant_id, "session_id": session_id},
        )

    async def list_scoped(
        self,
        *,
        tenant_id: int,
        session_id: str,
    ) -> list[TemporaryDocument]:
        """Return every live row of the session, oldest first."""
        stmt = text(
            "select * from temporary_documents "
            "where tenant_id = :tenant_id and session_id = :session_id "
            f"and {_LIVE} order by created_at asc"
        ).bindparams(tenant_id=tenant_id, session_id=session_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_expired(
        self,
        *,
        before: datetime,
        limit: int,
    ) -> list[TemporaryDocument]:
        """Return live rows whose ``expires_at`` is at or before ``before``.

        Ordered by expiry, oldest first, capped at ``limit`` — the sweep
        processes the most-stale rows first.
        """
        stmt = text(
            "select * from temporary_documents "
            f"where expires_at <= :before and {_LIVE} "
            "order by expires_at asc limit :limit"
        ).bindparams(before=before, limit=limit)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    # ── Lifecycle transitions ───────────────────────────────────────

    async def mark_processing(
        self,
        *,
        tenant_id: int,
        document_id: str,
        started_at: datetime,
        now: datetime,
    ) -> TemporaryDocument | None:
        """Transition a live row to ``processing`` with a start timestamp.

        Clears any stale error message. Returns the updated row, or
        ``None`` when no live row matched the scope.
        """
        stmt = text(
            "update temporary_documents set status = :status, "
            "started_at = :started_at, error_message = null, updated_at = :now "
            "where tenant_id = :tenant_id and id = :document_id "
            f"and {_LIVE} returning *"
        )
        result = await self._session.execute(
            stmt,
            {
                "status": TEMPORARY_DOCUMENT_STATUS_PROCESSING,
                "started_at": started_at,
                "now": now,
                "tenant_id": tenant_id,
                "document_id": document_id,
            },
        )
        return self._hydrate_opt(result.mappings().first())

    async def mark_ready(
        self,
        *,
        tenant_id: int,
        document_id: str,
        content: str,
        chunks: list[JsonObject],
        image_refs: list[JsonObject],
        metadata: JsonObject,
        token_count: int,
        chunk_count: int,
        ready_at: datetime,
        now: datetime,
    ) -> TemporaryDocument | None:
        """Persist the parsed artifacts and transition the row to ``ready``.

        Stores the extracted text plus the parsed chunk / image / metadata
        payloads. Clears any stale error message. Returns the updated row,
        or ``None`` when no live row matched the scope.
        """
        stmt = text(
            "update temporary_documents set status = :status, content = :content, "
            "chunks = :chunks, image_refs = :image_refs, metadata = :metadata, "
            "token_count = :token_count, chunk_count = :chunk_count, "
            "ready_at = :ready_at, error_message = null, updated_at = :now "
            "where tenant_id = :tenant_id and id = :document_id "
            f"and {_LIVE} returning *"
        ).bindparams(
            bindparam("chunks", type_=_JSON_BIND_TYPE),
            bindparam("image_refs", type_=_JSON_BIND_TYPE),
            bindparam("metadata", type_=_JSON_BIND_TYPE),
        )
        result = await self._session.execute(
            stmt,
            {
                "status": TEMPORARY_DOCUMENT_STATUS_READY,
                "content": content,
                "chunks": chunks,
                "image_refs": image_refs,
                "metadata": metadata,
                "token_count": token_count,
                "chunk_count": chunk_count,
                "ready_at": ready_at,
                "now": now,
                "tenant_id": tenant_id,
                "document_id": document_id,
            },
        )
        return self._hydrate_opt(result.mappings().first())

    async def mark_failed(
        self,
        *,
        tenant_id: int,
        document_id: str,
        message: str,
        now: datetime,
    ) -> TemporaryDocument | None:
        """Transition a live row to ``failed`` with the error message.

        Returns the updated row, or ``None`` when no live row matched.
        """
        stmt = text(
            "update temporary_documents set status = :status, "
            "error_message = :message, updated_at = :now "
            "where tenant_id = :tenant_id and id = :document_id "
            f"and {_LIVE} returning *"
        )
        result = await self._session.execute(
            stmt,
            {
                "status": TEMPORARY_DOCUMENT_STATUS_FAILED,
                "message": message,
                "now": now,
                "tenant_id": tenant_id,
                "document_id": document_id,
            },
        )
        return self._hydrate_opt(result.mappings().first())

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_scoped(
        self,
        *,
        tenant_id: int,
        session_id: str,
        document_id: str,
        now: datetime,
    ) -> bool:
        """Soft-delete the live (tenant, session, id) row.

        Returns whether a live row was affected.
        """
        stmt = text(
            "update temporary_documents set deleted_at = :now, updated_at = :now "
            "where tenant_id = :tenant_id and session_id = :session_id "
            "and id = :document_id and deleted_at is null"
        )
        result = await self._session.execute(
            stmt,
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "document_id": document_id,
                "now": now,
            },
        )
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0


__all__ = ["TemporaryDocumentRepository"]
