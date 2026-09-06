"""Storage rows for the MCP service domain.

Two tables backed by ``mcp_services`` (the MCP server configurations)
and ``mcp_tool_approvals`` (per-tool approval overrides for the same
service). The shape matches the upstream Go DDL in
``migrations/sqlite/000000_init.up.sql``.

Column notes
------------

- ``mcp_services.id`` is a UUID v4 assigned by the service (Go seeds
  via ``uuid.NewString()``); the column is the primary key and is
  populated by the caller.
- The structural JSON columns (``headers`` / ``auth_config`` /
  ``advanced_config`` / ``stdio_config`` / ``env_vars``) persist as
  JSON on SQLite and JSONB on Postgres via the dialect variant in the
  generic repository.
- ``mcp_tool_approvals`` has no ``deleted_at`` column on the Go side;
  rows are mutable and the unique constraint prevents duplicates per
  (tenant_id, service_id, tool_name).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonValue
from src.common.table_model import TableModel


class MCPService(TableModel):
    """One row of the ``mcp_services`` table."""

    table: ClassVar[str] = "mcp_services"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "headers",
        "auth_config",
        "advanced_config",
        "stdio_config",
        "env_vars",
    )
    db_generated_columns: ClassVar[tuple[str, ...]] = ()  # id is caller-assigned (UUID).

    id: str
    tenant_id: int
    name: str
    description: str | None = None
    enabled: bool = True
    transport_type: str
    url: str | None = None
    headers: dict[str, str] | None = None
    auth_config: JsonValue | None = None
    advanced_config: JsonValue | None = None
    stdio_config: JsonValue | None = None
    env_vars: dict[str, str] | None = None
    is_builtin: bool = False
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class MCPToolApproval(TableModel):
    """One row of the ``mcp_tool_approvals`` table.

    Append-only-ish: a row is upserted on each `SetRequireApproval` so
    the (tenant_id, service_id, tool_name) unique constraint is the
    source of truth. There is no ``deleted_at`` — clearing an approval
    is a write of ``require_approval = false``, not a delete.
    """

    table: ClassVar[str] = "mcp_tool_approvals"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    service_id: str
    tool_name: str
    require_approval: bool = False
    created_at: datetime
    updated_at: datetime = Field(default_factory=lambda: datetime.now())  # noqa: DTZ005


class MCPOAuthClient(TableModel):
    """One row of the ``mcp_oauth_clients`` table.

    Caches the dynamic-client-registration result for one
    ``(tenant_id, service_id)`` pair so OAuth flows against the same
    MCP server do not re-register on every authorization attempt.
    No ``deleted_at``: superseded registrations are overwritten in
    place (the unique constraint keeps one live row per pair).
    """

    table: ClassVar[str] = "mcp_oauth_clients"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    service_id: str
    client_id: str
    client_secret: str | None = None
    redirect_uri: str | None = None
    created_at: datetime
    updated_at: datetime


class MCPOAuthToken(TableModel):
    """One row of the ``mcp_oauth_tokens`` table.

    Holds the OAuth tokens a user granted for one MCP service,
    unique per ``(tenant_id, user_id, service_id)``. Refresh tokens
    rotate under a lease (``refresh_lease_id`` / ``refresh_lease_until``)
    so concurrent refreshes do not invalidate each other.
    """

    table: ClassVar[str] = "mcp_oauth_tokens"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    user_id: str
    service_id: str
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str | None = None
    expires_at: datetime | None = None
    refresh_lease_id: str | None = None
    refresh_lease_until: datetime | None = None
    created_at: datetime
    updated_at: datetime


__all__ = ["MCPOAuthClient", "MCPOAuthToken", "MCPService", "MCPToolApproval"]
