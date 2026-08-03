"""Internal DTOs for the `auth` domain.

These are service-output projections, not the HTTP wire shape. The
HTTP wire shape is in ``src/core/contracts/auth.py::AuthUser``.

`UserInfo` mirrors ``internal/types/user.go::UserInfo``: a stripped view
of the `users` row with the sensitive fields (``password_hash``,
``deleted_at``) removed. Services return ``UserInfo`` to the web layer
once authentication is complete; the password hash never leaves the
auth service.

`UserPreferences` mirrors ``types.UserPreferences``.

DB-row projection convention
-----------------------------

Each service-output DTO that mirrors a storage row exposes a
``map_from_db(cls, db) -> Self`` classmethod performing the boundary
translation: stripping storage-only / sensitive columns, hydrating
nested typed sub-models from JSON-backed dicts, and defensively
``json.loads``-ing raw-string JSON that some drivers surface. The
service layer calls ``UserInfo.map_from_db(user)``; the db layer never
references the wire DTO (AGENTS.md §1).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

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
    def from_json(cls, raw: dict[str, object] | str | None) -> UserPreferences:
        """Build from the JSON-backed ``preferences`` column value.

        Postgres (asyncpg) returns a parsed ``dict``; some drivers / a
        raw ``text`` column surface a JSON string. We accept both and
        ``json.loads`` the latter defensively.
        """
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            return cls.model_validate(json.loads(raw))
        return cls.model_validate(raw)


class UserInfo(BaseModel):
    """Wire-side projection of a `users` row.

    Mirrors ``internal/types/user.go::UserInfo``: the same field set as
    the storage ``User`` minus the sensitive columns
    (``password_hash``, ``deleted_at``). The service owns the
    ``User`` row during the login flow, then returns this
    ``UserInfo`` to the web layer.
    """

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
        """Project a storage ``User`` row to the wire-side ``UserInfo``.

        Strips ``password_hash`` (sensitive) and ``deleted_at``
        (storage-only soft-delete flag). Hydrates the typed
        :class:`UserPreferences` from the JSON-backed ``preferences``
        column value via :meth:`UserPreferences.from_json`.
        """
        record = db.model_dump(exclude=set(_USER_EXCLUDE_COLUMNS))
        record["preferences"] = UserPreferences.from_json(record.get("preferences"))
        return cls.model_validate(record)


__all__ = ["UserInfo", "UserPreferences"]
