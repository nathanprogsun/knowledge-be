"""Internal DTOs and constants for the `organizations` domain.

Service-output projections (not the HTTP wire shape). Every
``map_from_db`` performs the boundary translation: drops storage-only
columns (``deleted_at``) and credential-bearing columns (``invite_code``
is the join credential — the web layer renders an invite URL from it
and it is never a response field).

Role / status constants are re-exported from the storage model so the
domain layer and the storage layer share one vocabulary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.core.agents.types import CustomAgentInfo
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.models.organization import (
    JOIN_REQUEST_STATUS_APPROVED,
    JOIN_REQUEST_STATUS_PENDING,
    JOIN_REQUEST_STATUS_REJECTED,
    JOIN_REQUEST_TYPE_JOIN,
    JOIN_REQUEST_TYPE_UPGRADE,
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    Organization,
    OrganizationJoinRequest,
    OrganizationTenantMember,
)

# Columns of an `organizations` row that stay inside the service:
# ``invite_code`` is the join credential and ``deleted_at`` is
# storage-only.
_ORGANIZATION_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {"invite_code", "deleted_at"}
)

# Storage-only column of `organization_tenant_members` rows: the table
# has no soft-delete column, but the representative user id is
# display/audit only and stays on the wire projection.
_MEMBER_EXCLUDE_COLUMNS: frozenset[str] = frozenset()

# Storage-only columns of `organization_join_requests` rows: the review
# trail (reviewer id / message) is internal to the admin flow.
_JOIN_REQUEST_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {"reviewed_by", "review_message"}
)


class OrganizationInfo(BaseModel):
    """Service-side projection of an `organizations` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = Field(default=None)
    avatar: str = ""
    owner_id: str
    owner_tenant_id: int
    invite_code_expires_at: datetime | None = Field(default=None)
    invite_code_validity_days: int = 7
    require_approval: bool = False
    searchable: bool = False
    member_limit: int = 50
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: Organization) -> Self:
        """Project a storage row, dropping the invite code and soft-delete mark."""
        return cls.model_validate(db.model_dump(exclude=set(_ORGANIZATION_EXCLUDE_COLUMNS)))


class OrganizationMemberInfo(BaseModel):
    """Service-side projection of an `organization_tenant_members` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    organization_id: str
    tenant_id: int
    role: str
    representative_user_id: str = ""
    joined_at: datetime | None = Field(default=None)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: OrganizationTenantMember) -> Self:
        """Project a storage row unchanged (no secret-bearing columns)."""
        return cls.model_validate(db.model_dump(exclude=set(_MEMBER_EXCLUDE_COLUMNS)))


class OrganizationJoinRequestInfo(BaseModel):
    """Service-side projection of an `organization_join_requests` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    organization_id: str
    user_id: str
    tenant_id: int
    status: str
    requested_role: str
    request_type: str
    prev_role: str | None = Field(default=None)
    message: str | None = Field(default=None)
    reviewed_at: datetime | None = Field(default=None)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: OrganizationJoinRequest) -> Self:
        """Project a storage row, dropping the internal review trail."""
        return cls.model_validate(db.model_dump(exclude=set(_JOIN_REQUEST_EXCLUDE_COLUMNS)))


class SharedKnowledgeBaseInfo(BaseModel):
    """Service-side projection of one shared knowledge base row.

    ``permission`` is the effective permission after the org-role cap
    and the caller's tenant-role ceiling; ``knowledge_base`` carries the
    source-tenant row with its count enrichment. ``org_name`` is the
    organization the share landed in.
    """

    model_config = ConfigDict(frozen=True)

    knowledge_base: KnowledgeBaseInfo
    share_id: str
    organization_id: str
    org_name: str = ""
    permission: str
    source_tenant_id: int
    shared_at: datetime


class SharedAgentInfo(BaseModel):
    """Service-side projection of one shared agent row.

    ``permission`` is the effective permission after the same cap;
    ``web_search_ready`` is resolved against the source workspace and
    ``disabled_by_me`` records the caller's own hide preference.
    """

    model_config = ConfigDict(frozen=True)

    agent: CustomAgentInfo
    share_id: str
    organization_id: str
    org_name: str = ""
    permission: str
    source_tenant_id: int
    shared_at: datetime
    shared_by_user_id: str = ""
    shared_by_username: str = ""
    web_search_ready: bool = False
    disabled_by_me: bool = False


__all__ = [
    "JOIN_REQUEST_STATUS_APPROVED",
    "JOIN_REQUEST_STATUS_PENDING",
    "JOIN_REQUEST_STATUS_REJECTED",
    "JOIN_REQUEST_TYPE_JOIN",
    "JOIN_REQUEST_TYPE_UPGRADE",
    "ORG_ROLE_ADMIN",
    "ORG_ROLE_EDITOR",
    "ORG_ROLE_VIEWER",
    "OrganizationInfo",
    "OrganizationJoinRequestInfo",
    "OrganizationMemberInfo",
    "SharedAgentInfo",
    "SharedKnowledgeBaseInfo",
]
