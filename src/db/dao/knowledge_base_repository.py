"""Knowledge-base persistence — raw SQL only, no ORM.

Implements CRUD for the ``knowledge_bases`` table plus the aggregate
counts the domain service computes per query (document count, chunk
count, share count). The three count queries read the sibling
``knowledges`` / ``chunks`` / ``kb_shares`` tables — they are the
single place that backs the non-persisted ``knowledge_count`` /
``chunk_count`` / ``share_count`` response fields.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; JSON
columns are bound through the ``GenericRepository`` JSONB helper.
Reads filter soft-deleted rows (``deleted_at is null``).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.knowledge_base import KnowledgeBase


class KnowledgeBaseRepository(GenericRepository[KnowledgeBase]):
    """`knowledge_bases`-table SQL — CRUD + aggregate counts."""

    model_class = KnowledgeBase

    # ── CRUD ─────────────────────────────────────────────────────────

    async def create(self, row: KnowledgeBase) -> KnowledgeBase:
        """Insert a knowledge base and return the persisted row."""
        return await self.insert(row)

    async def get_by_id_or_none(self, id: str) -> KnowledgeBase | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_id_and_tenant(self, id: str, tenant_id: int) -> KnowledgeBase | None:
        """Return the live row for ``id`` scoped to ``tenant_id``, or ``None``.

        Enforces tenant isolation: a row owned by another tenant reads as
        absent.
        """
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_ids(self, ids: list[str]) -> list[KnowledgeBase]:
        """Return every live row whose id is in ``ids`` (order not guaranteed)."""
        if not ids:
            return []
        placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
        params: dict[str, str] = {f"id{i}": value for i, value in enumerate(ids)}
        stmt = text(
            f"select * from knowledge_bases where id in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_tenant(self, tenant_id: int) -> list[KnowledgeBase]:
        """Return every live, non-temporary knowledge base of the tenant.

        Temporary rows (``is_temporary = true``) are hidden from the
        default listing and rows are ordered newest-first.
        """
        stmt = text(
            "select * from knowledge_bases "
            "where tenant_id = :tenant_id and is_temporary = false "
            "and deleted_at is null order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def update(self, row: KnowledgeBase) -> KnowledgeBase:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``vector_store_id`` / ``created_at`` are
        immutable by contract — ``vector_store_id`` is bound once at
        creation time and never changes — so they stay out of the SET
        clause.
        """
        immutable = {"id", "tenant_id", "vector_store_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="knowledge_base.update_no_row",
                message=f"knowledge base {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            "update knowledge_bases set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Aggregate counts ─────────────────────────────────────────────

    async def count_by_vector_store_id(self, *, tenant_id: int, store_id: str) -> int:
        """Count live knowledge bases bound to a vector store within a tenant."""
        stmt = text(
            "select count(*) from knowledge_bases "
            "where tenant_id = :tenant_id and vector_store_id = :store_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, store_id=store_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

    async def count_documents(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count knowledge rows of a knowledge base (backs ``knowledge_count``)."""
        stmt = text(
            "select count(*) from knowledges "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

    async def count_chunks(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count chunk rows of a knowledge base (backs ``chunk_count``)."""
        stmt = text(
            "select count(*) from chunks "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

    async def count_members(self, *, tenant_id: int, knowledge_base_id: str) -> int:
        """Count share rows of a knowledge base (backs ``share_count``)."""
        stmt = text(
            "select count(*) from kb_shares "
            "where source_tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)


__all__ = ["KnowledgeBaseRepository"]
