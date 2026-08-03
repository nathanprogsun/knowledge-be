"""Internal DTOs for the `tenants` domain.

Service-output projections (not the HTTP wire shape). Every ``map_from_db``
performs the boundary translation: drops secret-bearing / storage-only
columns and hydrates typed objects from JSON-backed columns.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from src.db.models.tenants.tenant_invitations import TenantInvitation
from src.db.models.tenants.tenant_members import TenantMember
from src.db.models.tenants.tenants import Tenant

# Columns on the storage ``Tenant`` row that must NOT cross the service
# boundary: ``api_principal_config`` carries an HMAC secret;
# ``agent_config`` / ``conversation_config`` are legacy tenant-KV blobs
# owned by the settings endpoints; ``default_storage_backend_id``
# belongs to the storage-backend domain.
_TENANT_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "api_principal_config",
        "agent_config",
        "conversation_config",
        "default_storage_backend_id",
    }
)

# Credential columns of a `tenant_api_keys` row. ``key_hash`` is the
# authentication lookup value and ``api_key`` is the token itself;
# neither may leave the service.
_API_KEY_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"key_hash", "api_key"})

# Storage-only column of a `tenant_members` row: every read filters
# soft-deleted rows out, so the marker never carries information.
_MEMBERSHIP_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})

# Columns of a `tenant_invitations` row that stay inside the service:
# ``token`` is the share-link credential and ``deleted_at`` is
# storage-only.
_INVITATION_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"token", "deleted_at"})


class RetrieverEngineEntry(BaseModel):
    """One retriever/engine pair."""

    model_config = ConfigDict(frozen=True)

    retriever_type: str
    retriever_engine_type: str


class RetrieverEngines(BaseModel):
    """Typed view of the ``retriever_engines`` JSON column."""

    model_config = ConfigDict(frozen=True)

    engines: list[RetrieverEngineEntry] = Field(default_factory=list)

    @classmethod
    def from_json(
        cls,
        raw: dict[str, object] | list[dict[str, object]] | str | None,
    ) -> RetrieverEngines:
        """Build from the column value, accepting every persisted shape.

        Accepts the current wrapper form ``{"engines": [...]}``, the
        legacy bare-array form ``[...]``, and a raw JSON string (some
        drivers surface JSON columns as text).
        """
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            return cls.from_json(json.loads(raw))
        if isinstance(raw, list):
            return cls(engines=[RetrieverEngineEntry.model_validate(e) for e in raw])
        return cls.model_validate(raw)


class TenantInfo(BaseModel):
    """Service-side projection of a `tenants` row."""

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    description: str | None = Field(default=None)
    status: str
    retriever_engines: RetrieverEngines = Field(default_factory=RetrieverEngines)
    business: str | None = Field(default=None)
    storage_quota: int | None = Field(default=None)
    storage_used: int | None = Field(default=None)
    context_config: dict[str, object] | None = Field(default=None)
    web_search_config: dict[str, object] | None = Field(default=None)
    parser_engine_config: dict[str, object] | None = Field(default=None)
    credentials: dict[str, object] | None = Field(default=None)
    storage_engine_config: dict[str, object] | None = Field(default=None)
    chat_history_config: dict[str, object] | None = Field(default=None)
    retrieval_config: dict[str, object] | None = Field(default=None)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)

    @classmethod
    def map_from_db(cls, db: Tenant) -> Self:
        """Project a storage ``Tenant`` row to the service-side DTO."""
        record = db.model_dump(exclude=set(_TENANT_EXCLUDE_COLUMNS))
        record["retriever_engines"] = RetrieverEngines.from_json(record.get("retriever_engines"))
        return cls.model_validate(record)


class TenantAPIKeyInfo(BaseModel):
    """Service-side projection of a `tenant_api_keys` row.

    Drops both credential columns: ``key_hash`` (the authentication
    lookup value) and ``api_key`` (the token itself). The plaintext
    token is returned exactly once, by the create call, and never again.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int | None = Field(default=None)
    scope_type: str
    name: str
    full_access: bool
    knowledge_base_ids: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    last_used_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)
    revoked_at: datetime | None = Field(default=None)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: TenantAPIKey) -> Self:
        """Project a storage row, dropping both credential columns."""
        return cls.model_validate(db.model_dump(exclude=set(_API_KEY_EXCLUDE_COLUMNS)))


class MembershipInfo(BaseModel):
    """Service-side projection of a `tenant_members` row.

    Drops the soft-delete marker: callers only ever see live
    memberships, so ``deleted_at`` carries no information across the
    boundary.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    user_id: str
    tenant_id: int
    role: str
    status: str
    invited_by: str | None = Field(default=None)
    joined_at: datetime
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: TenantMember) -> Self:
        """Project a storage ``TenantMember`` row to the service DTO."""
        return cls.model_validate(db.model_dump(exclude=set(_MEMBERSHIP_EXCLUDE_COLUMNS)))


class TenantInvitationInfo(BaseModel):
    """Service-side projection of a `tenant_invitations` row.

    Drops the share-link ``token`` (the web layer renders an invite URL
    from it; it is never a response field) and the soft-delete marker.
    ``is_share_link`` is surfaced so callers do not have to special-case
    an empty invitee id.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int
    invitee_user_id: str
    invited_by: str | None = Field(default=None)
    role: str
    status: str
    message: str | None = Field(default=None)
    expires_at: datetime
    responded_at: datetime | None = Field(default=None)
    accepted_count: int = 0
    is_share_link: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: TenantInvitation) -> Self:
        """Project a storage row, dropping the token and soft-delete mark."""
        record = db.model_dump(exclude=set(_INVITATION_EXCLUDE_COLUMNS))
        record["is_share_link"] = db.is_share_link
        return cls.model_validate(record)


__all__ = [
    "MembershipInfo",
    "RetrieverEngineEntry",
    "RetrieverEngines",
    "TenantAPIKeyInfo",
    "TenantInfo",
    "TenantInvitationInfo",
]
