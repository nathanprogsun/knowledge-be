"""Knowledge-base share persistence — raw SQL only, no ORM.

One row per (knowledge base, organization) share. The (kb, org) pair
is unique among live rows (partial unique index on
``deleted_at is null``), so ``create_or_none`` suppresses a duplicate
share instead of raising — the service layer treats that as an
idempotent no-op.

Every read filters soft-deleted rows (``deleted_at is null``).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.kb_share import KnowledgeBaseShare

_LIVE = "deleted_at is null"

# Newest first: the share list reads as a reverse-chronological feed.
_SHARE_ORDER = "created_at desc, id desc"

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates this constant; user input never reaches the SQL
# string.
_KB_SHARE_TABLE = "kb_shares"


class KBShareRepository(GenericRepository[KnowledgeBaseShare]):
    """`kb_shares`-table SQL — CRUD + per-scope lists and counts."""

    model_class = KnowledgeBaseShare

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: KnowledgeBaseShare) -> KnowledgeBaseShare:
        """Insert a share row and return the persisted row."""
        return await self.insert(row)

    async def create_or_none(self, row: KnowledgeBaseShare) -> KnowledgeBaseShare | None:
        """Insert a share, returning ``None`` on a live duplicate.

        The conflict target matches the partial unique index on
        ``(knowledge_base_id, organization_id) WHERE deleted_at IS
        NULL``, so a second live share for the same pair is suppressed
        while soft-deleted rows insert freely.
        """
        columns = self.model_class.insert_sql_column_list()
        column_list = ", ".join(f'"{c}"' for c in columns)
        value_list = ", ".join(f":{c}" for c in columns)
        stmt = text(
            f"insert into {_KB_SHARE_TABLE} ({column_list}) values ({value_list}) "
            "on conflict (knowledge_base_id, organization_id) "
            f"where {_LIVE} do nothing returning *"
        ).bindparams(**row.insert_bind_params())
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def update(self, row: KnowledgeBaseShare) -> KnowledgeBaseShare:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``knowledge_base_id`` / ``organization_id`` /
        ``source_tenant_id`` / ``created_at`` are immutable by contract,
        so they stay out of the SET clause.
        """
        immutable = {"id", "knowledge_base_id", "organization_id", "source_tenant_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="kb_share.update_no_row",
                message=f"kb share {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            f"update {_KB_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where id = :id and {_LIVE}"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def delete_by_knowledge_base(self, *, knowledge_base_id: str, now: datetime) -> int:
        """Soft-delete every share of a knowledge base. Returns rows affected."""
        stmt = text(
            f"update {_KB_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where knowledge_base_id = :knowledge_base_id and {_LIVE}"
        ).bindparams(knowledge_base_id=knowledge_base_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return result.rowcount or 0

    async def delete_by_organization(self, *, organization_id: str, now: datetime) -> int:
        """Soft-delete every share into an organization. Returns rows affected."""
        stmt = text(
            f"update {_KB_SHARE_TABLE} set deleted_at = :now, updated_at = :now "
            f"where organization_id = :organization_id and {_LIVE}"
        ).bindparams(organization_id=organization_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return result.rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_or_none(self, id: str) -> KnowledgeBaseShare | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_kb_and_org_or_none(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShare | None:
        """Return the live share for the (kb, org) pair, or ``None``."""
        return await self.find_unique_by_column_values(
            {"knowledge_base_id": knowledge_base_id, "organization_id": organization_id},
        )

    async def list_by_knowledge_base(self, knowledge_base_id: str) -> list[KnowledgeBaseShare]:
        """Every live share of one knowledge base, newest first."""
        stmt = text(
            f"select * from {_KB_SHARE_TABLE} "
            f"where knowledge_base_id = :knowledge_base_id and {_LIVE} "
            f"order by {_SHARE_ORDER}"
        ).bindparams(knowledge_base_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_organization(self, organization_id: str) -> list[KnowledgeBaseShare]:
        """Every live share into one organization, newest first."""
        stmt = text(
            f"select * from {_KB_SHARE_TABLE} "
            f"where organization_id = :organization_id and {_LIVE} "
            f"order by {_SHARE_ORDER}"
        ).bindparams(organization_id=organization_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_by_knowledge_base(self, knowledge_base_id: str) -> int:
        """Count live shares of one knowledge base."""
        stmt = text(
            f"select count(*) from {_KB_SHARE_TABLE} "
            f"where knowledge_base_id = :knowledge_base_id and {_LIVE}"
        ).bindparams(knowledge_base_id=knowledge_base_id)
        return int((await self._session.execute(stmt)).scalar_one())


__all__ = ["KBShareRepository"]
