"""Internal DTOs for the auth domain.

Service-output projections, not the HTTP wire shape. The HTTP wire shape
lives in ``src/core/contracts/auth.py::AuthUser``. ``UserInfo`` is a
stripped view of the ``users`` row with the sensitive fields
(``password_hash``, ``deleted_at``) removed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.db.models.auth.users import User

# Columns present on the storage ``User`` row that must NOT cross the
# service boundary: ``password_hash`` is sensitive; ``deleted_at`` is a
# storage-only soft-delete flag. Centralised here so ``map_from_db`` is
# the single place that knows the projection rules.
_USER_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"password_hash", "deleted_at"})


class UserPreferences(BaseModel):
    """Per-user UI/feature preferences. Stored as JSONB on Postgres."""

    model_config = ConfigDict(frozen=True)

    last_active_tenant_id: int | None = None

    @classmethod
    def from_json(cls, raw: JsonObject | str | None) -> UserPreferences:
        """Build from the JSON-backed ``preferences`` column value.

        Accepts both a parsed ``dict`` and a raw JSON string.
        """
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            return cls.model_validate(json.loads(raw))
        return cls.model_validate(raw)


class UserInfo(BaseModel):
    """Wire-side projection of a ``users`` row (sensitive columns stripped)."""

    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    email: str
    avatar: str | None = Field(default=None)
    tenant_id: int | None = Field(default=None)
    is_active: bool
    can_access_all_tenants: bool = False
    is_system_admin: bool = False
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: User) -> Self:
        """Project a storage ``User`` row to the wire-side ``UserInfo``."""
        record = db.model_dump(exclude=set(_USER_EXCLUDE_COLUMNS))
        record["preferences"] = UserPreferences.from_json(record.get("preferences"))
        return cls.model_validate(record)


__all__ = ["UserInfo", "UserPreferences"]
