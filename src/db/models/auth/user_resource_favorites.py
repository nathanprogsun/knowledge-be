"""Storage row for the ``user_resource_favorites`` table.

Per-``(user, tenant)`` starred resources. ``resource_type`` is a string
discriminator (``"kb"``, ``"agent"``, ``...``); the primary key includes
``resource_type`` so different kinds with colliding ids coexist.

Column notes
------------

- No foreign keys: favorites survive across share revocations /
  re-grants and across the eventual soft-delete window. Hydration
  happens at read time with a ``LEFT JOIN`` and silently drops entries
  for resources the user can no longer see.
- The ``(user_id, tenant_id, resource_type, created_at DESC)`` index
  backs the per-type listing without an in-memory sort step; the
  ``(tenant_id)`` index supports bulk cleanup on tenant deletion.
- ``created_at`` is stamped by the database.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class UserResourceFavorite(TableModel):
    """One row of the ``user_resource_favorites`` table."""

    table: ClassVar[str] = "user_resource_favorites"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "user_id",
        "tenant_id",
        "resource_type",
        "resource_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``created_at`` carries a DB default; the application never inserts
    # it explicitly so it is excluded from the INSERT bindparam set.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("created_at",)

    user_id: str
    tenant_id: int
    resource_type: str
    resource_id: str
    created_at: datetime


__all__ = ["UserResourceFavorite"]