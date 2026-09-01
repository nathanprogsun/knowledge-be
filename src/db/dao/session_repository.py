"""Session persistence — raw SQL only, no ORM.

Implements the ``sessions``-table surface: create, tenant-scoped reads,
paginated keyword listing, title/description update, the pin toggle,
and soft delete. Every read filters ``deleted_at is null``.

User scoping mirrors the upstream visibility rule: when a ``user_id`` is
supplied, a row is visible only if it belongs to that user OR carries an
empty / NULL ``user_id`` (legacy tenant-level rows). An empty
``user_id`` means tenant-wide access (API-key callers, admin views).

Every query is ``sqlalchemy.text()`` with named ``bindparams``; user
input reaches only bindparam slots, never the SQL string.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.session import Session

_LIVE = "deleted_at is null"

# Newest activity first: the chat console reads as a reverse-chronological
# feed, and the pin toggle bumps ``updated_at`` so a freshly pinned row
# surfaces at the top of the unpinned ordering too.
_SESSION_ORDER = "updated_at desc, id desc"

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates either this constant or ``self._table`` (whose
# cached_property validates the identifier at first use); user input
# never reaches the SQL string.
_TABLE_NAME = "sessions"


class SessionRepository(GenericRepository[Session]):
    """`sessions`-table SQL — CRUD + pin toggle + soft delete."""

    model_class = Session

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Session) -> Session:
        """Insert a session and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: Session, *, user_id: str = "") -> Session:
        """Overwrite the mutable columns of the row, returning the result.

        ``id`` / ``tenant_id`` / ``user_id`` / ``is_pinned`` /
        ``pinned_at`` / ``created_at`` are immutable by contract, so they
        stay out of the SET clause. The update is scoped to the row's
        tenant and (when ``user_id`` is supplied) the owner scope, so a
        caller cannot edit another user's session.
        """
        updates: BindParams = {
            "title": row.title,
            "description": row.description,
            "updated_at": row.updated_at,
        }
        persisted = await self._update_scoped(
            tenant_id=row.tenant_id,
            id=row.id,
            user_id=user_id,
            updates=updates,
        )
        if persisted is None:
            raise DataError(
                code="session.update_no_row",
                message=f"session {row.id} not found for update",
            )
        return persisted

    async def soft_delete(
        self, *, tenant_id: int, id: str, now: datetime, user_id: str = ""
    ) -> bool:
        """Mark the row deleted. Returns whether a live row was affected."""
        stmt = text(
            f"update {_TABLE_NAME} set deleted_at = :now, updated_at = :now "
            f"where tenant_id = :tenant_id and id = :id and {_LIVE}"
            f"{self._user_scope_sql(user_id)}"
        ).bindparams(
            tenant_id=tenant_id,
            id=id,
            now=now,
            **self._user_scope_params(user_id),
        )
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def set_pinned(
        self,
        *,
        tenant_id: int,
        id: str,
        pinned: bool,
        now: datetime,
        user_id: str = "",
    ) -> bool:
        """Toggle ``is_pinned`` / ``pinned_at`` for one session.

        Returns whether a live, visible row was affected (0 = absent or
        not visible to the caller). Pinning stamps ``pinned_at``;
        unpinning clears it.
        """
        pinned_at = now if pinned else None
        stmt = text(
            f"update {_TABLE_NAME} set is_pinned = :pinned, pinned_at = :pinned_at, "
            f"updated_at = :now where tenant_id = :tenant_id and id = :id and {_LIVE}"
            f"{self._user_scope_sql(user_id)}"
        ).bindparams(
            tenant_id=tenant_id,
            id=id,
            pinned=pinned,
            pinned_at=pinned_at,
            now=now,
            **self._user_scope_params(user_id),
        )
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def set_owner_id(self, *, tenant_id: int, id: str, owner_id: str, now: datetime) -> bool:
        """Assign ``user_id`` to a tenant-scoped row. Returns rows affected."""
        stmt = text(
            f"update {_TABLE_NAME} set user_id = :owner_id, updated_at = :now "
            f"where tenant_id = :tenant_id and id = :id and {_LIVE}"
        ).bindparams(tenant_id=tenant_id, id=id, owner_id=owner_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, *, tenant_id: int, id: str) -> Session | None:
        """Return the live row for ``(tenant_id, id)``, or ``None``.

        Tenant-scoped only — no owner filter. Used by admin / API-key
        paths that may read any session of the workspace.
        """
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_id_for_user(self, *, tenant_id: int, user_id: str, id: str) -> Session | None:
        """Return the live row visible to ``user_id``, or ``None``.

        Applies the owner scope: the row must belong to the user or be a
        legacy tenant-level row (empty / NULL ``user_id``).
        """
        if not user_id:
            return await self.get_by_id(tenant_id=tenant_id, id=id)
        stmt = text(
            f"select * from {_TABLE_NAME} where tenant_id = :tenant_id and id = :id and {_LIVE}"
            f"{self._user_scope_sql(user_id)}"
        ).bindparams(tenant_id=tenant_id, id=id, **self._user_scope_params(user_id))
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def list_ids_by_tenant(self, *, tenant_id: int) -> list[str]:
        """Ids of the tenant's live sessions, newest first.

        Id-only scan for callers that narrow another tenant-less table
        (e.g. message search) to this tenant's session set — hydrating
        full rows would only discard them again.
        """
        stmt = text(
            f"select id from {_TABLE_NAME} "
            f"where tenant_id = :tenant_id and {_LIVE} "
            f"order by {_SESSION_ORDER}"
        ).bindparams(tenant_id=tenant_id)
        result = await self._session.execute(stmt)
        return [str(value) for value in result.scalars().all()]

    async def list_by_tenant(self, *, tenant_id: int, user_id: str = "") -> list[Session]:
        """Every live session of the workspace, newest activity first."""
        where = f"tenant_id = :tenant_id and {_LIVE}"
        params: BindParams = {"tenant_id": tenant_id}
        if user_id:
            where += self._user_scope_sql(user_id)
            params.update(self._user_scope_params(user_id))
        stmt = text(
            f"select * from {_TABLE_NAME} where {where} order by {_SESSION_ORDER}"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_pinned(self, *, tenant_id: int, user_id: str = "") -> list[Session]:
        """Every pinned live session of the workspace, most recently pinned first."""
        where = f"tenant_id = :tenant_id and is_pinned = true and {_LIVE}"
        params: BindParams = {"tenant_id": tenant_id}
        if user_id:
            where += self._user_scope_sql(user_id)
            params.update(self._user_scope_params(user_id))
        stmt = text(
            f"select * from {_TABLE_NAME} where {where} "
            "order by pinned_at desc, updated_at desc, id desc"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_paged(
        self,
        *,
        tenant_id: int,
        user_id: str = "",
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> tuple[list[Session], int]:
        """One page of the workspace's sessions plus the total.

        ``keyword`` filters ``title`` with a case-insensitive LIKE
        (wildcards neutralised). Ordering mirrors the upstream list
        query: pinned rows first, then ``pinned_at`` / ``updated_at``
        descending so OFFSET pagination stays stable.
        """
        keyword = keyword.strip()
        where = f"tenant_id = :tenant_id and {_LIVE}"
        params: BindParams = {"tenant_id": tenant_id}
        if user_id:
            where += self._user_scope_sql(user_id)
            params.update(self._user_scope_params(user_id))
        if keyword:
            where += " and lower(title) like :keyword escape '\\'"
            params["keyword"] = f"%{_escape_like(keyword.lower())}%"

        count_stmt = text(f"select count(*) from {_TABLE_NAME} where {where}").bindparams(**params)
        total = int((await self._session.execute(count_stmt)).scalar_one())

        offset = (page - 1) * page_size
        stmt = text(
            f"select * from {_TABLE_NAME} where {where} "
            "order by is_pinned desc, pinned_at desc, updated_at desc, id desc "
            "limit :limit offset :offset"
        ).bindparams(**params, limit=page_size, offset=offset)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        return rows, total

    # ── Query builders ──────────────────────────────────────────────

    async def _update_scoped(
        self,
        *,
        tenant_id: int,
        id: str,
        user_id: str,
        updates: BindParams,
    ) -> Session | None:
        set_clause = ", ".join(f'"{k}" = :u_{k}' for k in updates)
        update_params: BindParams = {f"u_{k}": v for k, v in updates.items()}
        stmt = text(
            f"update {_TABLE_NAME} set {set_clause} "
            f"where tenant_id = :tenant_id and id = :id and {_LIVE}"
            f"{self._user_scope_sql(user_id)} returning *"
        ).bindparams(
            tenant_id=tenant_id,
            id=id,
            **update_params,
            **self._user_scope_params(user_id),
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    @staticmethod
    def _user_scope_sql(user_id: str) -> str:
        """WHERE fragment enforcing the owner scope, or empty for tenant-wide."""
        if not user_id:
            return ""
        return " and (user_id = :scope_user_id or user_id is null or user_id = '')"

    @staticmethod
    def _user_scope_params(user_id: str) -> BindParams:
        """Bindparams for :meth:`_user_scope_sql` (empty when unscoped)."""
        if not user_id:
            return {}
        return {"scope_user_id": user_id}


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so the search term matches literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


__all__ = ["SessionRepository"]
