"""User-resource favorite persistence — raw SQL only, no ORM.

Backs the ``user_resource_favorites`` table. Every read is
tenant-scoped: a user switching workspaces sees only that workspace's
favorites. The composite primary key makes ``add`` an idempotent
upsert (concurrent double-clicks collapse into one row) and ``remove``
a no-op when the target is absent.

No soft-delete column is carried by this table — the upstream Go
schema is a pure insert/delete table, and the handler treats a
"removed" favorite as "no row". Callers that need an audit trail
should emit one at the service layer.
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import CursorResult, text

from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.user_resource_favorite import UserResourceFavorite

# Newest first: the favorites list reads as a reverse-chronological
# feed, so the most recently starred item lands at the top.
_LIST_ORDER = "created_at desc, user_id desc"

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates this constant; user input never reaches the SQL
# string.
_FAVORITE_TABLE = "user_resource_favorites"


class UserResourceFavoriteRepository(GenericRepository[UserResourceFavorite]):
    """`user_resource_favorites`-table SQL — add, remove, list, exists."""

    model_class = UserResourceFavorite

    # ── Writes ──────────────────────────────────────────────────────

    async def add(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> UserResourceFavorite | None:
        """Insert one favorite row, suppressing duplicates on the PK.

        Returns the persisted row, or ``None`` when the (user, tenant,
        type, id) combination already exists. The composite primary key
        is the conflict target, so concurrent double-clicks collapse
        into one row with no error path.
        """
        columns = self.model_class.insert_sql_column_list()
        column_list = ", ".join(f'"{c}"' for c in columns)
        value_list = ", ".join(f":{c}" for c in columns)
        stmt = text(
            f"insert into {_FAVORITE_TABLE} ({column_list}) values ({value_list}) "
            "on conflict (user_id, tenant_id, resource_type, resource_id) "
            "do nothing returning *"
        ).bindparams(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def remove(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Delete the row matching the composite key.

        Returns whether a row was actually removed. A missing target is
        not an error — the service layer treats the operation as a
        no-op idempotent unstar.
        """
        stmt = text(
            f"delete from {_FAVORITE_TABLE} "
            "where user_id = :user_id and tenant_id = :tenant_id "
            "and resource_type = :resource_type and resource_id = :resource_id"
        ).bindparams(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def list_by_user(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
    ) -> list[UserResourceFavorite]:
        """Return the live favorites of one user, newest first.

        ``resource_type`` is required (the handler validates against
        the allowlist) so the read always returns a single segment of
        the user's star list — the frontend renders KB and agent
        favorites as separate views.
        """
        stmt = text(
            f"select * from {_FAVORITE_TABLE} "
            "where user_id = :user_id and tenant_id = :tenant_id "
            "and resource_type = :resource_type "
            f"order by {_LIST_ORDER}"
        ).bindparams(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def is_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Return whether the (user, tenant, type, id) tuple is starred.

        Backs the rare read-only probe paths (e.g. a future
        GET-by-id endpoint that wants a single boolean rather than
        the full list).
        """
        stmt = text(
            f"select 1 from {_FAVORITE_TABLE} "
            "where user_id = :user_id and tenant_id = :tenant_id "
            "and resource_type = :resource_type and resource_id = :resource_id "
            "limit 1"
        ).bindparams(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return (await self._session.execute(stmt)).first() is not None


__all__ = ["UserResourceFavoriteRepository"]
