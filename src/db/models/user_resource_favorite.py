"""Storage row for the `user_resource_favorites` table.

Per-(user, tenant, resource_type, resource_id) star. ``resource_type``
is a small enum string (``'kb'`` / ``'agent'``) so a new favourite
target only requires a new constant, not a new table. The composite
primary key covers all four columns so a duplicate star is
rejected by the database and the upsert path is idempotent.

There are no foreign keys: favourites survive the underlying
resource being soft-deleted and are dropped lazily by the read
path when the resource is no longer visible to the user.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class UserResourceFavorite(TableModel):
    """One row of the `user_resource_favorites` table."""

    table: ClassVar[str] = "user_resource_favorites"
    primary_keys: ClassVar[tuple[str, ...]] = (
        "user_id",
        "tenant_id",
        "resource_type",
        "resource_id",
    )
    json_columns: ClassVar[tuple[str, ...]] = ()
    # All columns are caller-supplied: PK + the timestamp the
    # application stamps before insert.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    user_id: str
    tenant_id: int
    resource_type: str
    resource_id: str
    created_at: datetime


__all__ = ["UserResourceFavorite"]
