"""Tenant membership persistence — raw SQL only, no ORM.

Every read filters ``deleted_at IS NULL``; a removed membership is
invisible and the (user, workspace) pair becomes re-addable.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenants.tenant_members import TenantMember

ROLE_OWNER = "owner"
STATUS_ACTIVE = "active"

# Stable ordering for every membership listing.
_MEMBER_ORDER = "joined_at asc, id asc"

_LIVE = "deleted_at is null"

# Module-level alias for the table name — every ``text(f"...")`` in this
# file interpolates either this constant or ``self._table``; user input
# is always bound via ``:search`` / ``:user_id`` / etc.
_TABLE_NAME = "tenant_members"

# LIKE wildcards must be neutralised in user-supplied search terms.
_LIKE_ESCAPE_CHAR = "\\"
_LIKE_SPECIAL_CHARS = ("\\", "%", "_")

# The member search matches the joined user's email or username, so the
# filter needs the `users` table rather than `tenant_members` alone.
_USER_SEARCH_JOIN = (
    "inner join users on users.id = tenant_members.user_id and users.deleted_at is null"
)
_USER_SEARCH_PREDICATE = (
    f"(lower(users.email) like lower(:search) escape '{_LIKE_ESCAPE_CHAR}' "
    f"or lower(users.username) like lower(:search) escape '{_LIKE_ESCAPE_CHAR}')"
)


def escape_like_pattern(term: str) -> str:
    """Escape LIKE wildcards so the search term matches literally."""
    escaped = term
    for char in _LIKE_SPECIAL_CHARS:
        escaped = escaped.replace(char, _LIKE_ESCAPE_CHAR + char)
    return escaped


class TenantMemberRepository(GenericRepository[TenantMember]):
    """`tenant_members`-table SQL."""

    model_class = TenantMember

    # ── Reads ───────────────────────────────────────────────────────

    async def find_membership(self, *, user_id: str, tenant_id: int) -> TenantMember | None:
        """Return the live membership for the pair, or ``None``."""
        return await self.find_unique_by_column_values(
            {"user_id": user_id, "tenant_id": tenant_id},
        )

    async def list_by_user(self, user_id: str) -> list[TenantMember]:
        """Every live membership of one user, oldest join first."""
        return await self._select_members("user_id = :user_id", {"user_id": user_id})

    async def list_by_tenant(self, tenant_id: int) -> list[TenantMember]:
        """Every live membership inside one workspace, oldest join first."""
        return await self._select_members("tenant_id = :tenant_id", {"tenant_id": tenant_id})

    async def has_any_members(self, tenant_id: int) -> bool:
        """Whether the workspace has at least one active member."""
        stmt = text(
            f"select 1 from {_TABLE_NAME} "
            f"where tenant_id = :tenant_id and status = :status and {_LIVE} limit 1"
        ).bindparams(tenant_id=tenant_id, status=STATUS_ACTIVE)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def count_by_tenant(self, tenant_id: int, *, search: str | None = None) -> int:
        """Count live memberships, optionally filtered by user email/username."""
        join, where, params = self._search_fragments(tenant_id, search)
        stmt = text(f"select count(*) from {_TABLE_NAME} {join} where {where}").bindparams(**params)
        return int((await self._session.execute(stmt)).scalar_one())

    async def list_page_by_tenant(
        self,
        tenant_id: int,
        *,
        search: str | None = None,
        limit: int,
        offset: int,
    ) -> list[TenantMember]:
        """One page of live memberships, oldest join first."""
        join, where, params = self._search_fragments(tenant_id, search)
        stmt = text(
            f"select {_TABLE_NAME}.* from {_TABLE_NAME} {join} where {where} "
            f"order by {_TABLE_NAME}.joined_at asc, {_TABLE_NAME}.id asc "
            "limit :limit offset :offset"
        ).bindparams(**params, limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_active_owners(self, tenant_id: int) -> int:
        """Count live, active Owner rows in the workspace."""
        stmt = text(
            f"select count(*) from {_TABLE_NAME} "
            f"where tenant_id = :tenant_id and role = :role and status = :status and {_LIVE}"
        ).bindparams(tenant_id=tenant_id, role=ROLE_OWNER, status=STATUS_ACTIVE)
        return int((await self._session.execute(stmt)).scalar_one())

    async def count_other_active_owners_for_update(
        self,
        *,
        tenant_id: int,
        exclude_user_id: str,
    ) -> int:
        """Lock the workspace's *other* active Owner rows and count them.

        The ``FOR UPDATE`` makes "demote / remove an Owner" safe against
        a concurrent demotion of a different Owner: the second
        transaction blocks here instead of reading a stale count and
        leaving the workspace ownerless. Transaction boundaries stay
        with the caller.
        """
        stmt = text(
            f"select id from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and user_id <> :user_id "
            f"and role = :role and status = :status and {_LIVE} for update"
        ).bindparams(
            tenant_id=tenant_id,
            user_id=exclude_user_id,
            role=ROLE_OWNER,
            status=STATUS_ACTIVE,
        )
        return len((await self._session.execute(stmt)).all())

    # ── Mutations ───────────────────────────────────────────────────

    async def insert_live_or_none(self, row: TenantMember) -> TenantMember | None:
        """Insert a membership, returning ``None`` on a live duplicate.

        The conflict target matches the partial unique index
        (``(user_id, tenant_id) WHERE deleted_at IS NULL``), so a row
        that duplicates a *live* membership is suppressed while
        re-adding a previously removed member still inserts.
        """
        columns = self.model_class.insert_sql_column_list()
        column_list = ", ".join(f'"{c}"' for c in columns)
        value_list = ", ".join(f":{c}" for c in columns)
        stmt = text(
            f"insert into {_TABLE_NAME} ({column_list}) values ({value_list}) "
            f"on conflict (user_id, tenant_id) where {_LIVE} do nothing returning *"
        ).bindparams(**row.insert_bind_params())
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def update_role(
        self,
        *,
        user_id: str,
        tenant_id: int,
        role: str,
        updated_at: datetime,
    ) -> int:
        """Set the role of a live membership; return rows affected."""
        return await self._update_live(
            "role = :role, updated_at = :updated_at",
            {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role,
                "updated_at": updated_at,
            },
        )

    async def soft_delete(
        self,
        *,
        user_id: str,
        tenant_id: int,
        deleted_at: datetime,
    ) -> int:
        """Soft-delete a live membership; return rows affected."""
        return await self._update_live(
            "deleted_at = :deleted_at, updated_at = :deleted_at",
            {"user_id": user_id, "tenant_id": tenant_id, "deleted_at": deleted_at},
        )

    async def soft_delete_by_tenant(
        self,
        *,
        tenant_id: int,
        deleted_at: datetime,
    ) -> int:
        """Soft-delete every live membership of a workspace.

        Called when the workspace itself is deleted (Go deletes
        ``TenantMember`` rows in the same transaction) so the deleted
        workspace never surfaces in ``/auth/me``.
        """
        return await self._update_live(
            "deleted_at = :deleted_at, updated_at = :deleted_at",
            {"tenant_id": tenant_id, "deleted_at": deleted_at},
            where="tenant_id = :tenant_id",
        )

    # ── Query builders ──────────────────────────────────────────────

    async def _select_members(
        self,
        conditions: str,
        params: BindParams,
    ) -> list[TenantMember]:
        stmt = text(
            f"select * from {_TABLE_NAME} where {conditions} and {_LIVE} order by {_MEMBER_ORDER}"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def _update_live(
        self,
        set_clause: str,
        params: BindParams,
        *,
        where: str = "user_id = :user_id and tenant_id = :tenant_id",
    ) -> int:
        stmt = text(
            f"update {_TABLE_NAME} set {set_clause} where {where} and {_LIVE}"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount

    def _search_fragments(
        self,
        tenant_id: int,
        search: str | None,
    ) -> tuple[str, str, BindParams]:
        """Build (join, where, params) for the filtered listing queries."""
        where = f"{_TABLE_NAME}.tenant_id = :tenant_id and {_TABLE_NAME}.{_LIVE}"
        params: BindParams = {"tenant_id": tenant_id}
        term = (search or "").strip()
        if not term:
            return "", where, params
        params["search"] = f"%{escape_like_pattern(term)}%"
        return _USER_SEARCH_JOIN, f"{where} and {_USER_SEARCH_PREDICATE}", params


__all__ = ["ROLE_OWNER", "STATUS_ACTIVE", "TenantMemberRepository", "escape_like_pattern"]
