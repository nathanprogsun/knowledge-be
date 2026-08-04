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

# Storage-only or secret-bearing columns of an ``mcp_services`` row.
# ``auth_config.api_key`` / ``auth_config.token`` are NOT a row column
# — they live inside the JSON blob — so the wire DTO strips them at
# the contract layer instead.
_AUTH_SECRET_KEYS: tuple[str, ...] = ("api_key", "token")


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
        """Project the storage row, parsing JSON columns and stripping secrets."""
        record = db.model_dump()
        for column in ("headers", "auth_config", "advanced_config", "stdio_config", "env_vars"):
            record[column] = _parse_json_blob(record.get(column), column)
        if isinstance(record.get("auth_config"), dict):
            auth = dict(record["auth_config"])
            for secret_key in _AUTH_SECRET_KEYS:
                auth.pop(secret_key, None)
            record["auth_config"] = auth
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


__all__ = ["MCPServiceInfo", "MCPToolApprovalInfo"]
