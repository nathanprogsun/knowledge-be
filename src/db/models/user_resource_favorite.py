"""Storage row for the `user_resource_favorites` table.

One row records that a user starred a single resource of a specific
type (``kb`` or ``agent``) in the current workspace. The composite
primary key ``(user_id, tenant_id, resource_type, resource_id)``
guarantees idempotent inserts and matches the upstream contract: a
favoriting action collapses to one row even under concurrent
double-clicks, and tenant-scoped reads are naturally enforced at the
key level.

Favorites are intentionally personal — there is no share / cross-user
visibility. The handler layer rejects attempts to operate on another
user's favorites so the key never has to carry an authorization
filter beyond the natural ``user_id`` / ``tenant_id`` lookup.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel

# ── Resource type vocabulary ───────────────────────────────────────
# Mirrors the upstream ``IsValidFavoriteResourceType`` allowlist. Adding
# a new favoritable resource is just a new constant + a frontend hook;
# no schema change is required because ``resource_type`` is a free
# ``VARCHAR(16)`` column.

RESOURCE_TYPE_KB = "kb"
RESOURCE_TYPE_AGENT = "agent"

FAVORITE_RESOURCE_TYPES: frozenset[str] = frozenset({RESOURCE_TYPE_KB, RESOURCE_TYPE_AGENT})


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
    # ``created_at`` carries a server-side default; the database fills
    # it on insert, so the column stays out of the INSERT list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ("created_at",)

    user_id: str
    tenant_id: int
    resource_type: str
    resource_id: str
    created_at: datetime


__all__ = [
    "FAVORITE_RESOURCE_TYPES",
    "RESOURCE_TYPE_AGENT",
    "RESOURCE_TYPE_KB",
    "UserResourceFavorite",
]
