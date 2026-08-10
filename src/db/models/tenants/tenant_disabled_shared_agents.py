"""Storage row for the ``tenant_disabled_shared_agents`` table.

Per-tenant "disabled by me" toggle for shared agents received from other
tenants via the organization share graph. Membership is a triple
``(tenant_id, agent_id, source_tenant_id)``: a tenant may opt out of any
agent it sees without affecting what other tenants see.

Column notes
------------

- The composite primary key makes ``ON CONFLICT DO NOTHING`` the natural
  upsert shape; the per-tenant lookup index supports the "list my
  disabled agents" view.
- ``created_at`` is stamped by the database; the application never reads
  it back as part of the upsert flow.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class TenantDisabledSharedAgent(TableModel):
    """One row of the ``tenant_disabled_shared_agents`` table."""

    table: ClassVar[str] = "tenant_disabled_shared_agents"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "tenant_id",
        "agent_id",
        "source_tenant_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``created_at`` carries a DB default; the application stamps it on
    # the upsert path and the row identity is the (tenant, agent, source)
    # triple, not the timestamp.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    tenant_id: int
    agent_id: str
    source_tenant_id: int
    created_at: datetime


__all__ = ["TenantDisabledSharedAgent"]