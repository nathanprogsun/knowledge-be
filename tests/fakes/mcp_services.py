"""In-memory fakes for the MCP service domain repositories.

Mirror the real ``MCPServiceRepository`` / ``MCPToolApprovalRepository``
method signatures so a drift between the two surfaces surfaces as a
type error rather than a passing test.

Generator-based UUIDs (uuid4 hex) are used for the primary key so the
service can stay caller-assigned, exactly like the Go ``BeforeCreate``
hook. Soft delete sets ``deleted_at``; reads filter the deleted rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime

from src.common.exception import NotFoundError
from src.db.models.infra.mcp_services import MCPService, MCPToolApproval


def _new_id() -> str:
    return uuid.uuid4().hex


class FakeMCPServiceRepository:
    """In-memory stand-in for ``MCPServiceRepository``."""

    def __init__(self) -> None:
        self.rows: dict[str, MCPService] = {}

    # ── Reads ───────────────────────────────────────────────────────

    async def find_for_tenant(
        self,
        tenant_id: int,
        id: str,
    ) -> MCPService | None:
        row = self.rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def get_by_id(self, tenant_id: int, id: str) -> MCPService:
        row = await self.find_for_tenant(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code="mcp_service.not_found",
                message=f"MCP service {id} not found",
            )
        return row

    async def list_for_tenant(self, tenant_id: int) -> list[MCPService]:
        live = [
            row
            for row in self.rows.values()
            if row.tenant_id == tenant_id and not row.is_builtin and row.deleted_at is None
        ]
        return sorted(live, key=lambda r: r.created_at, reverse=True)

    async def find_builtin(self, id: str) -> MCPService | None:
        row = self.rows.get(id)
        if row is None or not row.is_builtin or row.deleted_at is not None:
            return None
        return row

    async def exists_by_tenant_and_name(
        self,
        tenant_id: int,
        name: str,
    ) -> bool:
        """Return whether a live row with ``(tenant_id, name)`` exists.

        Mirrors the real ``MCPServiceRepository.exists_by_tenant_and_name``
        used by the 409 pre-check path.
        """
        return any(
            row.tenant_id == tenant_id
            and row.name == name
            and row.deleted_at is None
            for row in self.rows.values()
        )

    # ── Mutations ───────────────────────────────────────────────────

    async def insert(self, row: MCPService) -> MCPService:
        self.rows[row.id] = row
        return row

    async def update_by_primary_key(
        self,
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
    ) -> MCPService | None:
        # Not used by the service; retained so the fake's interface
        # stays aligned with ``GenericRepository``.
        pk_id = str(primary_key_to_value.get("id"))
        existing = self.rows.get(pk_id)
        if existing is None or existing.deleted_at is not None:
            return None
        updated = existing.model_copy(update=column_to_update)
        self.rows[pk_id] = updated
        return updated

    async def soft_delete(
        self,
        tenant_id: int,
        id: str,
        *,
        deleted_at: datetime,
    ) -> bool:
        row = await self.find_for_tenant(tenant_id, id)
        if row is None:
            return False
        self.rows[id] = row.model_copy(
            update={"deleted_at": deleted_at, "updated_at": deleted_at},
        )
        return True

    async def update(
        self,
        tenant_id: int,
        id: str,
        *,
        columns: dict[str, object],
    ) -> MCPService | None:
        row = await self.find_for_tenant(tenant_id, id)
        if row is None:
            return None
        updated = row.model_copy(update=columns)
        self.rows[id] = updated
        return updated


class FakeMCPToolApprovalRepository:
    """In-memory stand-in for ``MCPToolApprovalRepository``."""

    def __init__(self) -> None:
        self.rows: dict[str, MCPToolApproval] = {}

    async def list_by_service(
        self,
        tenant_id: int,
        service_id: str,
    ) -> list[MCPToolApproval]:
        live = [
            row
            for row in self.rows.values()
            if row.tenant_id == tenant_id and row.service_id == service_id
        ]
        return sorted(live, key=lambda r: r.tool_name)

    async def upsert(self, *, row: MCPToolApproval) -> MCPToolApproval:
        key = (row.tenant_id, row.service_id, row.tool_name)
        existing = next(
            (r for r in self.rows.values() if (r.tenant_id, r.service_id, r.tool_name) == key),
            None,
        )
        stored_id = existing.id if existing is not None else row.id
        merged = row.model_copy(
            update={"id": stored_id, "updated_at": row.created_at},
        )
        self.rows[stored_id] = merged
        return merged


def make_id() -> str:
    """Expose a deterministic id factory for tests that prefer fixed ids."""
    return _new_id()


def ids_of(rows: Iterable[MCPService]) -> list[str]:
    """Return the ids in the order they were inserted."""
    return [r.id for r in rows]


__all__ = [
    "FakeMCPServiceRepository",
    "FakeMCPToolApprovalRepository",
    "ids_of",
    "make_id",
]
