"""Storage row for the `agent_shares` table.

One row records that a custom agent was shared into an organization for
cross-tenant collaboration. ``source_tenant_id`` is the tenant that owns
the agent (the sharing side); ``permission`` is the org-level grant
(admin / editor / viewer) capped at runtime by the receiver's own role
inside the organization.

``id`` is application-assigned (a caller-minted UUID). The (agent,
source tenant, organization) tuple is unique among live rows, so an
agent can be shared into the same organization at most once.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel

# ── Share permission levels ──────────────────────────────────────────
# Same ladder as the org member roles: admin > editor > viewer. The
# grant is capped at runtime by the receiver's own role inside the
# organization.

SHARE_PERMISSION_ADMIN = "admin"
SHARE_PERMISSION_EDITOR = "editor"
SHARE_PERMISSION_VIEWER = "viewer"

SHARE_PERMISSIONS: frozenset[str] = frozenset(
    {SHARE_PERMISSION_ADMIN, SHARE_PERMISSION_EDITOR, SHARE_PERMISSION_VIEWER}
)


class AgentShare(TableModel):
    """One row of the `agent_shares` table."""

    table: ClassVar[str] = "agent_shares"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``id`` is application-assigned (a caller-minted UUID), so it takes
    # part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    agent_id: str
    organization_id: str
    shared_by_user_id: str
    source_tenant_id: int
    permission: str = SHARE_PERMISSION_VIEWER
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = [
    "SHARE_PERMISSIONS",
    "SHARE_PERMISSION_ADMIN",
    "SHARE_PERMISSION_EDITOR",
    "SHARE_PERMISSION_VIEWER",
    "AgentShare",
]
