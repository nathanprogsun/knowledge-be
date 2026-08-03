"""Tenant invitation persistence — raw SQL only, no ORM.

Every read filters ``deleted_at IS NULL``.

Two write helpers carry the concurrency semantics:
``mark_status_if_pending`` gates the state transition on the row still
being pending so a concurrent responder loses cleanly instead of
overwriting a terminal state; ``increment_accepted_count`` bumps the
counter with a single UPDATE expression so parallel share-link accepts
cannot lose an increment.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenants.tenant_invitations import (
    STATUS_EXPIRED,
    STATUS_PENDING,
    TenantInvitation,
)

_LIVE = "deleted_at is null"

# Newest first: the management UI and the invitee inbox both read as a
# reverse-chronological feed.
_INVITATION_ORDER = "created_at desc, id desc"


class TenantInvitationRepository(GenericRepository[TenantInvitation]):
    """`tenant_invitations`-table SQL."""

    model_class = TenantInvitation

    # ── Writes ──────────────────────────────────────────────────────

    async def insert_pending_or_none(self, row: TenantInvitation) -> TenantInvitation | None:
        """Insert an invitation, returning ``None`` on a pending duplicate.

        The conflict target matches the partial unique index on
        ``(tenant_id, invitee_user_id) WHERE status = 'pending' AND
        deleted_at IS NULL AND invitee_user_id <> ''``, so a second
        pending invitation for the same invitee is suppressed while
        terminal rows and share links insert freely.
        """
        columns = self.model_class.insert_sql_column_list()
        column_list = ", ".join(f'"{c}"' for c in columns)
        value_list = ", ".join(f":{c}" for c in columns)
        stmt = text(
            f"insert into {self._table} ({column_list}) values ({value_list}) "
            "on conflict (tenant_id, invitee_user_id) "
            f"where status = '{STATUS_PENDING}' and {_LIVE} and invitee_user_id <> '' "
            "do nothing returning *"
        ).bindparams(**row.insert_bind_params())
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def mark_status_if_pending(
        self,
        invitation_id: int,
        *,
        status: str,
        responded_at: datetime,
    ) -> int:
        """Transition a pending row; return rows affected (0 = lost race)."""
        stmt = text(
            f"update {self._table} "
            "set status = :status, responded_at = :responded_at, updated_at = :responded_at "
            f"where id = :id and status = :pending and {_LIVE}"
        ).bindparams(
            id=invitation_id,
            status=status,
            responded_at=responded_at,
            pending=STATUS_PENDING,
        )
        result = await self._session.execute(stmt)
        return cast("CursorResult[object]", result).rowcount

    async def sweep_expired(self, now: datetime) -> int:
        """Flip every overdue pending row to expired; return rows affected."""
        stmt = text(
            f"update {self._table} "
            "set status = :expired, responded_at = :now, updated_at = :now "
            f"where status = :pending and expires_at < :now and {_LIVE}"
        ).bindparams(expired=STATUS_EXPIRED, pending=STATUS_PENDING, now=now)
        result = await self._session.execute(stmt)
        return cast("CursorResult[object]", result).rowcount

    async def increment_accepted_count(self, invitation_id: int) -> int:
        """Bump ``accepted_count`` by one; return rows affected."""
        stmt = text(
            f"update {self._table} set accepted_count = accepted_count + 1 "
            f"where id = :id and {_LIVE}"
        ).bindparams(id=invitation_id)
        result = await self._session.execute(stmt)
        return cast("CursorResult[object]", result).rowcount

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_id_or_none(self, invitation_id: int) -> TenantInvitation | None:
        """Return the row whatever its status, or ``None`` when missing."""
        return await self.find_by_primary_key({"id": invitation_id})

    async def find_pending_by_pair(
        self,
        *,
        tenant_id: int,
        invitee_user_id: str,
    ) -> TenantInvitation | None:
        """Return the invitee's pending invitation for the workspace."""
        return await self.find_unique_by_column_values(
            {
                "tenant_id": tenant_id,
                "invitee_user_id": invitee_user_id,
                "status": STATUS_PENDING,
            }
        )

    async def find_pending_by_token(self, token: str) -> TenantInvitation | None:
        """Resolve a share-link token to its still-pending row."""
        if not token:
            return None
        return await self.find_unique_by_column_values(
            {"token": token, "status": STATUS_PENDING},
        )

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        include_terminal: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TenantInvitation]:
        """Invitations of one workspace, newest first."""
        where, params = self._list_conditions(
            "tenant_id = :tenant_id",
            {"tenant_id": tenant_id},
            include_terminal=include_terminal,
        )
        return await self._select_page(where, params, limit=limit, offset=offset)

    async def count_by_tenant(self, tenant_id: int, *, include_terminal: bool = False) -> int:
        """Count invitations of one workspace under the same filter."""
        where, params = self._list_conditions(
            "tenant_id = :tenant_id",
            {"tenant_id": tenant_id},
            include_terminal=include_terminal,
        )
        return await self._count(where, params)

    async def list_by_invitee(
        self,
        invitee_user_id: str,
        *,
        include_terminal: bool = False,
    ) -> list[TenantInvitation]:
        """One user's invitation inbox, newest first."""
        where, params = self._list_conditions(
            "invitee_user_id = :invitee_user_id",
            {"invitee_user_id": invitee_user_id},
            include_terminal=include_terminal,
        )
        return await self._select_page(where, params, limit=None, offset=0)

    async def count_pending_by_invitee(self, invitee_user_id: str) -> int:
        """Count the user's pending invitations (the inbox badge)."""
        return await self._count(
            f"invitee_user_id = :invitee_user_id and status = :pending and {_LIVE}",
            {"invitee_user_id": invitee_user_id, "pending": STATUS_PENDING},
        )

    # ── Query builders ──────────────────────────────────────────────

    @staticmethod
    def _list_conditions(
        scope: str,
        params: dict[str, object],
        *,
        include_terminal: bool,
    ) -> tuple[str, dict[str, object]]:
        where = f"{scope} and {_LIVE}"
        if include_terminal:
            return where, params
        return f"{where} and status = :pending", {**params, "pending": STATUS_PENDING}

    async def _select_page(
        self,
        where: str,
        params: dict[str, object],
        *,
        limit: int | None,
        offset: int,
    ) -> list[TenantInvitation]:
        stmt_text = f"select * from {self._table} where {where} order by {_INVITATION_ORDER}"
        page_params: dict[str, object] = dict(params)
        if limit is not None:
            stmt_text += " limit :limit offset :offset"
            page_params["limit"] = limit
            page_params["offset"] = offset
        result = await self._session.execute(text(stmt_text).bindparams(**page_params))
        return [self._hydrate(m) for m in result.mappings().all()]

    async def _count(self, where: str, params: dict[str, object]) -> int:
        stmt = text(f"select count(*) from {self._table} where {where}").bindparams(**params)
        return int((await self._session.execute(stmt)).scalar_one())


__all__ = ["TenantInvitationRepository"]
