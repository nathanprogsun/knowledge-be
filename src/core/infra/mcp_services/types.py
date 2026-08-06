"""Internal DTOs for the MCP service domain.

Service-output projections. Every ``map_from_db`` performs the boundary
translation: drops internal-only columns and hydrates typed objects
from JSON-backed columns. The wire-side ``MCPService`` contract from
``src/core/contracts/infra.py`` is what the web layer renders.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.db.models.infra.mcp_services import MCPService, MCPToolApproval

# ``auth_config.api_key`` / ``auth_config.token`` are NOT a row column —
# they live inside the JSON blob — and the wire contract carries them
# verbatim on user-created services (mirrors Go's
# ``dto.NewMCPServiceResponse``). Built-in services are not exposed via
# the API, so credential stripping is intentionally not done at this
# layer.

# PR-30.6c H2 / H4: storage columns that must not cross into the
# service-output projection. ``deleted_at`` is a soft-delete tombstone;
# the service layer treats a missing row as the only delete signal.
# ``_MCP_SERVICE_EXCLUDE_COLUMNS`` is a frozenset (per AGENTS.md §9)
# consumed by ``map_from_db`` to drop these keys before
# ``model_validate``.
_MCP_SERVICE_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})

# ``headers`` / ``env_vars`` carry operator-supplied credentials (the
# Go ``MCPService.Headers`` / ``EnvVars`` columns). Strip them at the
# service boundary so the wire contract never sees a plaintext bearer
# token or secret env var.
_AUTH_HEADER_NAMES: frozenset[str] = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)
_REDACTED_CREDENTIAL_PLACEHOLDER = "***"


class MCPServiceInfo(BaseModel):
    """Service-side projection of an ``mcp_services`` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    description: str | None = Field(default=None)
    enabled: bool = Field(default=True)
    transport_type: str
    url: str | None = Field(default=None)
    headers: dict[str, str] | None = Field(default=None)
    auth_config: JsonObject | None = Field(default=None)
    advanced_config: JsonObject | None = Field(default=None)
    stdio_config: JsonObject | None = Field(default=None)
    env_vars: dict[str, str] | None = Field(default=None)
    is_builtin: bool = Field(default=False)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: MCPService) -> Self:
        """Project the storage row, parsing JSON columns; redact secrets.

        PR-30.6c H2 / H4:

        - ``_MCP_SERVICE_EXCLUDE_COLUMNS`` (frozen per AGENTS.md §9)
          drops the soft-delete tombstone (``deleted_at``) before the
          Pydantic model is built.
        - ``headers`` keys carrying a credential (``Authorization`` /
          ``Cookie`` / ``Proxy-Authorization``) are replaced with
          ``"***"`` so the wire contract never sees a plaintext bearer
          token, mirroring Go's ``redactAuthHeader`` behaviour.
        - ``env_vars`` are replaced with a presence map
          (``{name: "***"}``) — only the keys survive, values are
          stripped, so the UI can show "var set" without leaking the
          secret value (mirrors Go ``dto.MCPServiceResponse.EnvVars``).
        - ``auth_config`` keeps its verbatim shape (user-created
          services carry credentials in the wire contract per Go
          ``dto.NewMCPServiceResponse``).
        """
        record = db.model_dump(exclude=_MCP_SERVICE_EXCLUDE_COLUMNS)
        for column in ("headers", "auth_config", "advanced_config", "stdio_config", "env_vars"):
            record[column] = _parse_json_blob(record.get(column), column)
        record = _redact_mcp_secrets(record)
        return cls.model_validate(record)


class MCPToolApprovalInfo(BaseModel):
    """Service-side projection of an ``mcp_tool_approvals`` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    service_id: str
    tool_name: str
    require_approval: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: MCPToolApproval) -> Self:
        return cls.model_validate(db.model_dump())


def _parse_json_blob(raw: JsonValue, column: str) -> JsonValue:
    """Decode a JSON-backed column, accepting the persisted raw shape.

    SQLite stores JSON columns as text on some paths; Pydantic may
    receive a dict directly. Normalise both to a JSON object (or pass
    through ``None``).
    """
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return cast("JsonValue", raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"mcp_services.{column} is not valid JSON: {exc}") from exc
        return cast("JsonValue", decoded)
    return None


def _redact_mcp_secrets(record: JsonObject) -> JsonObject:
    """Apply the §9 projection: redact credential columns.

    PR-30.6c H4: redact known auth headers (``Authorization`` /
    ``Cookie`` / ``Proxy-Authorization``) to ``"***"`` and reduce
    ``env_vars`` to a presence map (``{name: "***"}``).

    Note: storage-only columns (``deleted_at``) are already dropped
    by the caller's ``model_dump(exclude=...)``; this helper only
    handles the in-place redaction.
    """
    out: JsonObject = dict(record)
    headers = out.get("headers")
    if isinstance(headers, dict):
        redacted_headers: dict[str, str] = {}
        for header_name, header_value in headers.items():
            if (
                isinstance(header_name, str)
                and header_name.lower() in _AUTH_HEADER_NAMES
                and isinstance(header_value, str)
                and header_value
            ):
                redacted_headers[header_name] = _REDACTED_CREDENTIAL_PLACEHOLDER
            elif isinstance(header_value, str):
                redacted_headers[header_name] = header_value
            # Non-string header values are dropped (HTTP headers are
            # always string-shaped).
        out["headers"] = cast("dict[str, str]", redacted_headers)
    env_vars = out.get("env_vars")
    if isinstance(env_vars, dict):
        presence_map: dict[str, str] = {}
        for env_name in env_vars:
            if isinstance(env_name, str) and env_name:
                presence_map[env_name] = _REDACTED_CREDENTIAL_PLACEHOLDER
        out["env_vars"] = cast("dict[str, str]", presence_map)
    return out


# PR-30.6c H2: §9 public name — consumers reference this frozenset by
# its module-level public name (per spec: ``_<NAME>_EXCLUDE_COLUMNS``
# becomes ``<NAME>_EXCLUDE_COLUMNS``).
MCP_SERVICE_EXCLUDE_COLUMNS: frozenset[str] = _MCP_SERVICE_EXCLUDE_COLUMNS


__all__ = [
    "MCP_SERVICE_EXCLUDE_COLUMNS",
    "MCPServiceInfo",
    "MCPToolApprovalInfo",
]
