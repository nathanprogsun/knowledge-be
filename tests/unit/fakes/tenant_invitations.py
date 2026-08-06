"""Shared in-memory fake for the tenant invitation repository.

Method signatures mirror ``TenantInvitationRepository``. The partial
pending-uniqueness rule is reproduced here: a second *pending* row for
the same (workspace, invitee) is refused, while terminal rows and
share-link rows accumulate freely.
"""

from __future__ import annotations

from datetime import datetime

from src.db.models.tenants.tenant_invitations import (
    STATUS_EXPIRED,
    STATUS_PENDING,
    TenantInvitation,
    is_expired,
)


class FakeTenantInvitationRepository:
    """In-memory replacement for `TenantInvitationRepository`."""

    def __init__(self) -> None:
        self.rows: dict[int, TenantInvitation] = {}
        self._next_id = 1

    # ── Writes ──────────────────────────────────────────────────────

    async def insert_pending_or_none(self, row: TenantInvitation) -> TenantInvitation | None:
        if row.invitee_user_id and await self.find_pending_by_pair(
            tenant_id=row.tenant_id,
            invitee_user_id=row.invitee_user_id,
        ):
            return None
        stored = row.model_copy(update={"id": self._next_id})
        self.rows[stored.id] = stored
        self._next_id += 1
        return stored

    async def mark_status_if_pending(
        self,
        invitation_id: int,
        *,
        status: str,
        responded_at: datetime,
    ) -> int:
        row = self.rows.get(invitation_id)
        if row is None or row.status != STATUS_PENDING or row.deleted_at is not None:
            return 0
        self.rows[invitation_id] = row.model_copy(
            update={
                "status": status,
                "responded_at": responded_at,
                "updated_at": responded_at,
            }
        )
        return 1

    async def sweep_expired(self, now: datetime) -> int:
        swept = 0
        for key, row in list(self.rows.items()):
            if row.status == STATUS_PENDING and row.deleted_at is None and is_expired(row, now):
                self.rows[key] = row.model_copy(
                    update={
                        "status": STATUS_EXPIRED,
                        "responded_at": now,
                        "updated_at": now,
                    }
                )
                swept += 1
        return swept

    async def increment_accepted_count(self, invitation_id: int) -> int:
        row = self.rows.get(invitation_id)
        if row is None or row.deleted_at is not None:
            return 0
        self.rows[invitation_id] = row.model_copy(update={"accepted_count": row.accepted_count + 1})
        return 1

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_id_or_none(self, invitation_id: int) -> TenantInvitation | None:
        row = self.rows.get(invitation_id)
        return row if row is not None and row.deleted_at is None else None

    async def find_pending_by_pair(
        self,
        *,
        tenant_id: int,
        invitee_user_id: str,
    ) -> TenantInvitation | None:
        for row in self._live():
            if (
                row.tenant_id == tenant_id
                and row.invitee_user_id == invitee_user_id
                and row.status == STATUS_PENDING
            ):
                return row
        return None

    async def find_pending_by_token(self, token: str) -> TenantInvitation | None:
        if not token:
            return None
        for row in self._live():
            if row.token == token and row.status == STATUS_PENDING:
                return row
        return None

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        include_terminal: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TenantInvitation]:
        rows = self._filtered(
            [r for r in self._live() if r.tenant_id == tenant_id],
            include_terminal=include_terminal,
        )
        return rows[offset : offset + limit] if limit is not None else rows

    async def count_by_tenant(self, tenant_id: int, *, include_terminal: bool = False) -> int:
        return len(
            self._filtered(
                [r for r in self._live() if r.tenant_id == tenant_id],
                include_terminal=include_terminal,
            )
        )

    async def list_by_invitee(
        self,
        invitee_user_id: str,
        *,
        include_terminal: bool = False,
    ) -> list[TenantInvitation]:
        return self._filtered(
            [r for r in self._live() if r.invitee_user_id == invitee_user_id],
            include_terminal=include_terminal,
        )

    async def count_pending_by_invitee(self, invitee_user_id: str) -> int:
        return len(
            [
                r
                for r in self._live()
                if r.invitee_user_id == invitee_user_id and r.status == STATUS_PENDING
            ]
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _live(self) -> list[TenantInvitation]:
        return [r for r in self.rows.values() if r.deleted_at is None]

    @staticmethod
    def _filtered(
        rows: list[TenantInvitation],
        *,
        include_terminal: bool,
    ) -> list[TenantInvitation]:
        selected = rows if include_terminal else [r for r in rows if r.status == STATUS_PENDING]
        return sorted(selected, key=lambda r: (r.created_at, r.id), reverse=True)


__all__ = ["FakeTenantInvitationRepository"]
