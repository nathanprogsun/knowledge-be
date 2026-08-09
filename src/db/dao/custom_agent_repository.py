"""Custom-agent persistence — raw SQL only, no ORM.

Implements CRUD for the ``custom_agents`` table plus the model-usage
count the domain service computes for tenant model managers. The count
reads the ``config`` JSONB blob's model-binding fields and is the single
place that backs the non-persisted "agents using this model" response
field.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; the
``config`` JSONB column is bound through the ``GenericRepository``
JSONB helper. Reads filter soft-deleted rows (``deleted_at is null``).
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.custom_agent import CustomAgent


class CustomAgentRepository(GenericRepository[CustomAgent]):
    """`custom_agents`-table SQL — CRUD + model-usage count."""

    model_class = CustomAgent

    # ── CRUD ─────────────────────────────────────────────────────────

    async def create(self, row: CustomAgent) -> CustomAgent:
        """Insert a custom agent and return the persisted row."""
        return await self.insert(row)

    async def get_by_id_and_tenant(self, *, id: str, tenant_id: int) -> CustomAgent | None:
        """Return the live row for ``(id, tenant_id)``, or ``None``.

        Enforces tenant isolation: a row owned by another tenant reads
        as absent.
        """
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def list_by_tenant(self, tenant_id: int) -> list[CustomAgent]:
        """Return every live agent row of the tenant, newest-first."""
        stmt = text(
            "select * from custom_agents "
            "where tenant_id = :tenant_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def update(self, row: CustomAgent) -> CustomAgent:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``created_at`` / ``created_by`` /
        ``is_builtin`` are immutable by contract — built-in status is
        fixed at creation time and the service rejects edits to built-in
        rows — so they stay out of the SET clause.
        """
        immutable = {"id", "tenant_id", "created_at", "created_by", "is_builtin"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key(
            {"id": row.id, "tenant_id": row.tenant_id},
            updates,
        )
        if persisted is None:
            raise DataError(
                code="custom_agent.update_no_row",
                message=f"custom agent {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, tenant_id: int, now: datetime) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            "update custom_agents set deleted_at = :now, updated_at = :now "
            "where id = :id and tenant_id = :tenant_id and deleted_at is null"
        ).bindparams(id=id, tenant_id=tenant_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Model usage count ─────────────────────────────────────────────

    async def count_by_model_id(self, *, tenant_id: int, model_id: str) -> int:
        """Count live agents whose config references ``model_id``.

        Mirrors the upstream JSONB scope: any of the six model-binding
        fields (conversation, rerank, VLM, ASR, query-understand, and
        the follow-up suggestion model) may reference the id.
        """
        stmt = text(
            "select count(*) from custom_agents "
            "where tenant_id = :tenant_id and deleted_at is null "
            "and (config->>'model_id' = :model_id "
            "or config->>'rerank_model_id' = :model_id "
            "or config->>'vlm_model_id' = :model_id "
            "or config->>'asr_model_id' = :model_id "
            "or config->>'query_understand_model_id' = :model_id "
            "or config->'question_suggestions'->'follow_ups'->>'model_id' = :model_id)"
        ).bindparams(tenant_id=tenant_id, model_id=model_id)
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)


__all__ = ["CustomAgentRepository"]
