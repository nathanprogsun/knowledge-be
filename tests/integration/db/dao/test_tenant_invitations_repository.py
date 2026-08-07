"""Integration tests for ``TenantInvitationRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique invitee ids and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.tenant_invitations_repository import TenantInvitationRepository
from src.db.models.tenants.tenant_invitations import (
    STATUS_ACCEPTED,
    STATUS_DECLINED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    TenantInvitation,
)
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _uid() -> str:
    return f"usr-{uuid.uuid4().hex[:12]}"


def _invitation(
    *,
    invitee_user_id: str | None = None,
    tenant_id: int | None = None,
    token: str = "",
    role: str = "viewer",
    status: str = STATUS_PENDING,
    expires_at: datetime = _FUTURE,
    created_at: datetime = _NOW,
) -> TenantInvitation:
    return TenantInvitation(
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        invitee_user_id=invitee_user_id or _uid(),
        token=token,
        role=role,
        status=status,
        expires_at=expires_at,
        created_at=created_at,
        updated_at=created_at,
    )


# ── insert / pending uniqueness ─────────────────────────────────────


async def test_insert_assigns_id(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)

    stored = await repo.insert_pending_or_none(_invitation())
    await session.commit()

    assert stored is not None
    assert stored.id > 0


async def test_second_pending_invitation_is_suppressed(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    inv = _invitation()
    await repo.insert_pending_or_none(inv)
    await session.commit()

    duplicate = await repo.insert_pending_or_none(inv)
    await session.commit()

    assert duplicate is None


async def test_new_invitation_allowed_once_the_previous_is_terminal(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)
    inv = _invitation()
    first = await repo.insert_pending_or_none(inv)
    await session.commit()
    assert first is not None
    await repo.mark_status_if_pending(first.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    second = await repo.insert_pending_or_none(inv)
    await session.commit()

    assert second is not None


async def test_share_link_rows_are_exempt_from_pending_uniqueness(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)
    tid = make_test_tenant_id()

    first = await repo.insert_pending_or_none(
        _invitation(invitee_user_id="", tenant_id=tid, token=f"tok-{uuid.uuid4().hex[:8]}")
    )
    second = await repo.insert_pending_or_none(
        _invitation(invitee_user_id="", tenant_id=tid, token=f"tok-{uuid.uuid4().hex[:8]}")
    )
    await session.commit()

    assert first is not None
    assert second is not None


# ── lookups ─────────────────────────────────────────────────────────


async def test_find_pending_by_pair(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    uid = _uid()
    tid = make_test_tenant_id()
    stored = await repo.insert_pending_or_none(_invitation(invitee_user_id=uid, tenant_id=tid))
    await session.commit()
    assert stored is not None

    found = await repo.find_pending_by_pair(tenant_id=tid, invitee_user_id=uid)

    assert found is not None
    assert found.id == stored.id


async def test_find_pending_by_token_ignores_terminal_rows(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    token = f"tok-{uuid.uuid4().hex[:8]}"
    stored = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token=token))
    await session.commit()
    assert stored is not None
    assert await repo.find_pending_by_token(token) is not None

    await repo.mark_status_if_pending(stored.id, status=STATUS_ACCEPTED, responded_at=_NOW)
    await session.commit()

    assert await repo.find_pending_by_token(token) is None


async def test_find_pending_by_token_rejects_empty_token(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)

    assert await repo.find_pending_by_token("") is None


async def test_find_by_id_returns_terminal_rows_too(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation())
    await session.commit()
    assert stored is not None
    await repo.mark_status_if_pending(stored.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    found = await repo.find_by_id_or_none(stored.id)

    assert found is not None
    assert found.status == STATUS_DECLINED


# ── state transitions ───────────────────────────────────────────────


async def test_mark_status_if_pending_stamps_the_response(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation())
    await session.commit()
    assert stored is not None

    affected = await repo.mark_status_if_pending(
        stored.id,
        status=STATUS_ACCEPTED,
        responded_at=_NOW,
    )
    await session.commit()

    assert affected == 1
    row = await repo.find_by_id_or_none(stored.id)
    assert row is not None
    assert row.status == STATUS_ACCEPTED
    assert row.responded_at == _NOW


async def test_mark_status_if_pending_is_a_noop_on_terminal_rows(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation())
    await session.commit()
    assert stored is not None
    await repo.mark_status_if_pending(stored.id, status=STATUS_ACCEPTED, responded_at=_NOW)
    await session.commit()

    affected = await repo.mark_status_if_pending(
        stored.id,
        status=STATUS_DECLINED,
        responded_at=_NOW,
    )

    assert affected == 0


async def test_sweep_expires_only_overdue_pending_rows(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    uid_a = _uid()
    uid_b = _uid()
    stale = await repo.insert_pending_or_none(
        _invitation(invitee_user_id=uid_a, expires_at=_NOW - timedelta(days=1))
    )
    fresh = await repo.insert_pending_or_none(_invitation(invitee_user_id=uid_b))
    await session.commit()
    assert stale is not None
    assert fresh is not None

    swept = await repo.sweep_expired(_NOW)
    await session.commit()

    assert swept >= 1
    stale_row = await repo.find_by_id_or_none(stale.id)
    fresh_row = await repo.find_by_id_or_none(fresh.id)
    assert stale_row is not None
    assert fresh_row is not None
    assert stale_row.status == STATUS_EXPIRED
    assert fresh_row.status == STATUS_PENDING


async def test_increment_accepted_count_accumulates(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    token = f"tok-{uuid.uuid4().hex[:8]}"
    stored = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token=token))
    await session.commit()
    assert stored is not None

    await repo.increment_accepted_count(stored.id)
    await repo.increment_accepted_count(stored.id)
    await session.commit()

    row = await repo.find_by_id_or_none(stored.id)
    assert row is not None
    assert row.accepted_count == 2


# ── listings ────────────────────────────────────────────────────────


async def test_list_by_tenant_filters_terminal_rows_by_default(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)
    tid = make_test_tenant_id()
    pending = await repo.insert_pending_or_none(_invitation(invitee_user_id=_uid(), tenant_id=tid))
    terminal = await repo.insert_pending_or_none(_invitation(invitee_user_id=_uid(), tenant_id=tid))
    await session.commit()
    assert pending is not None
    assert terminal is not None
    await repo.mark_status_if_pending(terminal.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    visible = await repo.list_by_tenant(tid)
    everything = await repo.list_by_tenant(tid, include_terminal=True)

    assert [r.id for r in visible] == [pending.id]
    assert {r.id for r in everything} == {pending.id, terminal.id}


async def test_list_by_tenant_is_newest_first_and_pageable(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    tid = make_test_tenant_id()
    created = []
    for index in range(3):
        inv = await repo.insert_pending_or_none(
            _invitation(
                invitee_user_id=_uid(),
                tenant_id=tid,
                created_at=_NOW + timedelta(hours=index),
            )
        )
        assert inv is not None
        created.append(inv)
    await session.commit()

    page = await repo.list_by_tenant(tid, limit=2, offset=0)

    assert [r.invitee_user_id for r in page] == [
        created[2].invitee_user_id,
        created[1].invitee_user_id,
    ]


async def test_count_by_tenant_matches_the_filter(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    tid = make_test_tenant_id()
    pending = await repo.insert_pending_or_none(_invitation(invitee_user_id=_uid(), tenant_id=tid))
    terminal = await repo.insert_pending_or_none(_invitation(invitee_user_id=_uid(), tenant_id=tid))
    await session.commit()
    assert pending is not None
    assert terminal is not None
    await repo.mark_status_if_pending(terminal.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    assert await repo.count_by_tenant(tid) == 1
    assert await repo.count_by_tenant(tid, include_terminal=True) == 2


async def test_list_and_count_by_invitee(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    uid = _uid()
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    await repo.insert_pending_or_none(_invitation(invitee_user_id=uid, tenant_id=tid_a))
    await repo.insert_pending_or_none(_invitation(invitee_user_id=uid, tenant_id=tid_b))
    await repo.insert_pending_or_none(_invitation(invitee_user_id=_uid(), tenant_id=tid_a))
    await session.commit()

    inbox = await repo.list_by_invitee(uid)

    assert len(inbox) == 2
    assert await repo.count_pending_by_invitee(uid) == 2


# ── tenant isolation ────────────────────────────────────────────────


async def test_find_pending_by_pair_isolated_by_tenant(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    uid = _uid()
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    await repo.insert_pending_or_none(_invitation(invitee_user_id=uid, tenant_id=tid_a))
    await session.commit()

    assert await repo.find_pending_by_pair(tenant_id=tid_a, invitee_user_id=uid) is not None
    assert await repo.find_pending_by_pair(tenant_id=tid_b, invitee_user_id=uid) is None
