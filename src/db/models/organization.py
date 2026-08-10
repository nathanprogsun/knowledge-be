"""Storage rows for the organization collaboration tables.

Three tables back the cross-tenant sharing domain:

- ``organizations`` — one row per collaboration space. ``id`` is
  application-assigned (a caller-minted UUID); ``owner_tenant_id`` is
  pinned at creation time and never changes, so the owning workspace
  can never be orphaned even if the owner user later moves tenants.
- ``organization_tenant_members`` — one row per (org, tenant)
  participation. Membership is tenant-scoped, not user-scoped: the
  ``representative_user_id`` is display/audit only, and permission
  checks use the (org, tenant, role) tuple exclusively.
- ``organization_join_requests`` — one row per join / role-upgrade
  request awaiting admin review. ``request_type`` distinguishes a new
  member join from a role upgrade; ``status`` is ``pending ->
  approved | rejected``.

All three tables use the same soft-delete convention as the rest of
the storage layer: reads filter ``deleted_at is null``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel

# ── Org member roles ──────────────────────────────────────────────────

ORG_ROLE_ADMIN = "admin"
ORG_ROLE_EDITOR = "editor"
ORG_ROLE_VIEWER = "viewer"

ORG_ROLES: frozenset[str] = frozenset({ORG_ROLE_ADMIN, ORG_ROLE_EDITOR, ORG_ROLE_VIEWER})

# Role ladder: admin > editor > viewer. A higher index means more
# permission; ``role_level`` is the single source of truth for the
# ``has_org_permission`` comparison.
_ROLE_LEVELS: dict[str, int] = {
    ORG_ROLE_ADMIN: 3,
    ORG_ROLE_EDITOR: 2,
    ORG_ROLE_VIEWER: 1,
}


def is_valid_org_role(role: str) -> bool:
    """Whether ``role`` is one of the three sanctioned org roles."""
    return role in ORG_ROLES


def has_org_permission(role: str, required: str) -> bool:
    """Whether ``role`` grants at least the ``required`` permission level."""
    return _ROLE_LEVELS.get(role, 0) >= _ROLE_LEVELS.get(required, 0)


# ── Join request statuses / types ─────────────────────────────────────

JOIN_REQUEST_STATUS_PENDING = "pending"
JOIN_REQUEST_STATUS_APPROVED = "approved"
JOIN_REQUEST_STATUS_REJECTED = "rejected"

JOIN_REQUEST_STATUSES: frozenset[str] = frozenset(
    {JOIN_REQUEST_STATUS_PENDING, JOIN_REQUEST_STATUS_APPROVED, JOIN_REQUEST_STATUS_REJECTED}
)

JOIN_REQUEST_TYPE_JOIN = "join"
JOIN_REQUEST_TYPE_UPGRADE = "upgrade"

JOIN_REQUEST_TYPES: frozenset[str] = frozenset(
    {JOIN_REQUEST_TYPE_JOIN, JOIN_REQUEST_TYPE_UPGRADE}
)


class Organization(TableModel):
    """One row of the `organizations` table."""

    table: ClassVar[str] = "organizations"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``id`` is application-assigned (a caller-minted UUID), so it takes
    # part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    name: str
    description: str | None = None
    owner_id: str
    owner_tenant_id: int
    invite_code: str | None = None
    invite_code_expires_at: datetime | None = None
    invite_code_validity_days: int = 7
    avatar: str = ""
    require_approval: bool = False
    searchable: bool = False
    member_limit: int = 50
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class OrganizationTenantMember(TableModel):
    """One row of the `organization_tenant_members` table."""

    table: ClassVar[str] = "organization_tenant_members"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    organization_id: str
    tenant_id: int
    role: str = ORG_ROLE_VIEWER
    representative_user_id: str = ""
    joined_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class OrganizationJoinRequest(TableModel):
    """One row of the `organization_join_requests` table."""

    table: ClassVar[str] = "organization_join_requests"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    organization_id: str
    user_id: str
    tenant_id: int
    status: str = JOIN_REQUEST_STATUS_PENDING
    requested_role: str = ORG_ROLE_VIEWER
    request_type: str = JOIN_REQUEST_TYPE_JOIN
    prev_role: str | None = None
    message: str | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_message: str | None = None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "JOIN_REQUEST_STATUSES",
    "JOIN_REQUEST_STATUS_APPROVED",
    "JOIN_REQUEST_STATUS_PENDING",
    "JOIN_REQUEST_STATUS_REJECTED",
    "JOIN_REQUEST_TYPES",
    "JOIN_REQUEST_TYPE_JOIN",
    "JOIN_REQUEST_TYPE_UPGRADE",
    "ORG_ROLES",
    "ORG_ROLE_ADMIN",
    "ORG_ROLE_EDITOR",
    "ORG_ROLE_VIEWER",
    "Organization",
    "OrganizationJoinRequest",
    "OrganizationTenantMember",
    "has_org_permission",
    "is_valid_org_role",
]
