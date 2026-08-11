"""Storage row for the `tenant_disabled_shared_agents` table.

Per-tenant opt-out list for agents that have been shared into the
tenant through an organisation share. The composite primary key
``(tenant_id, agent_id, source_tenant_id)`` lets a tenant disable the
same agent that was shared by multiple source tenants independently.

Rows are append-only; disabling is a row insert, re-enabling is a
row delete. The table carries no ``updated_at`` / ``deleted_at`` to
match the upstream contract: there is no mutation path other than
insert and delete.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class TenantDisabledSharedAgent(TableModel):
    """One row of the `tenant_disabled_shared_agents` table."""

    table: ClassVar[str] = "tenant_disabled_shared_agents"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "agent_id",
        "source_tenant_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()

    tenant_id: int
    agent_id: str
    source_tenant_id: int
    created_at: datetime


__all__ = ["TenantDisabledSharedAgent"]
