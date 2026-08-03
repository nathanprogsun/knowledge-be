"""Integration tests for `TenantInvitationRepository` against a real Postgres.

The DDL mirrors `alembic/versions/0006_tenant_invitations.py`, including
the three partial indexes — several tests exist specifically to pin their
predicates (pending-uniqueness that skips share links, token uniqueness
among rows that have one).

The session-scoped `pg_url` fixture skips the suite when Docker is
unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.db.dao.tenant_invitations_repository import TenantInvitationRepository
from src.db.models.tenants.tenant_invitations import (
    STATUS_ACCEPTED,
    STATUS_DECLINED,
    STATUS_EXPIRED,
    STATUS_PENDING,
    TenantInvitation,
)

_DROP_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenant_invitations CASCADE")

_CREATE_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenant_invitations (
        id BIGSERIAL PRIMARY KEY,
        tenant_id BIGINT NOT NULL,
        invitee_user_id VARCHAR(36) NOT NULL DEFAULT '',
        token VARCHAR(64) NOT NULL DEFAULT '',
        invited_by VARCHAR(36),
        role VARCHAR(20) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        message VARCHAR(500),
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        responded_at TIMESTAMP WITH TIME ZONE,
        accepted_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)

_CREATE_PENDING_INDEX_SQL = sqlalchemy.text(
    """
    CREATE UNIQUE INDEX idx_tenant_invitations_unique_pending
        ON tenant_invitations(tenant_id, invitee_user_id)
        WHERE status = 'pending' AND deleted_at IS NULL AND invitee_user_id <> ''
    """
)

_CREATE_TOKEN_INDEX_SQL = sqlalchemy.text(
    """
    CREATE UNIQUE INDEX idx_tenant_invitations_token
        ON tenant_invitations(token)
        WHERE token <> '' AND deleted_at IS NULL
    """
)

_TENANT_ID = 7
_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
        await conn.execute(_CREATE_SQL)
        await conn.execute(_CREATE_PENDING_INDEX_SQL)
        await conn.execute(_CREATE_TOKEN_INDEX_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
    await engine.dispose()


def _invitation(
    *,
    invitee_user_id: str = "usr-1",
    tenant_id: int = _TENANT_ID,
    token: str = "",
    role: str = "viewer",
    status: str = STATUS_PENDING,
    expires_at: datetime = _FUTURE,
    created_at: datetime = _NOW,
) -> TenantInvitation:
    return TenantInvitation(
        tenant_id=tenant_id,
        invitee_user_id=invitee_user_id,
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
    await repo.insert_pending_or_none(_invitation())
    await session.commit()

    duplicate = await repo.insert_pending_or_none(_invitation())
    await session.commit()

    assert duplicate is None


async def test_new_invitation_allowed_once_the_previous_is_terminal(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)
    first = await repo.insert_pending_or_none(_invitation())
    await session.commit()
    assert first is not None
    await repo.mark_status_if_pending(first.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    second = await repo.insert_pending_or_none(_invitation())
    await session.commit()

    assert second is not None


async def test_share_link_rows_are_exempt_from_pending_uniqueness(
    session: AsyncSession,
) -> None:
    repo = TenantInvitationRepository(session)

    first = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token="tok-1"))
    second = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token="tok-2"))
    await session.commit()

    assert first is not None
    assert second is not None


# ── lookups ─────────────────────────────────────────────────────────


async def test_find_pending_by_pair(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation())
    await session.commit()
    assert stored is not None

    found = await repo.find_pending_by_pair(tenant_id=_TENANT_ID, invitee_user_id="usr-1")

    assert found is not None
    assert found.id == stored.id


async def test_find_pending_by_token_ignores_terminal_rows(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token="tok-1"))
    await session.commit()
    assert stored is not None
    assert await repo.find_pending_by_token("tok-1") is not None

    await repo.mark_status_if_pending(stored.id, status=STATUS_ACCEPTED, responded_at=_NOW)
    await session.commit()

    assert await repo.find_pending_by_token("tok-1") is None


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
    stale = await repo.insert_pending_or_none(
        _invitation(invitee_user_id="usr-1", expires_at=_NOW - timedelta(days=1))
    )
    fresh = await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-2"))
    await session.commit()
    assert stale is not None
    assert fresh is not None

    swept = await repo.sweep_expired(_NOW)
    await session.commit()

    assert swept == 1
    stale_row = await repo.find_by_id_or_none(stale.id)
    fresh_row = await repo.find_by_id_or_none(fresh.id)
    assert stale_row is not None
    assert fresh_row is not None
    assert stale_row.status == STATUS_EXPIRED
    assert fresh_row.status == STATUS_PENDING


async def test_increment_accepted_count_accumulates(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    stored = await repo.insert_pending_or_none(_invitation(invitee_user_id="", token="tok-1"))
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
    pending = await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-1"))
    terminal = await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-2"))
    await session.commit()
    assert pending is not None
    assert terminal is not None
    await repo.mark_status_if_pending(terminal.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    visible = await repo.list_by_tenant(_TENANT_ID)
    everything = await repo.list_by_tenant(_TENANT_ID, include_terminal=True)

    assert [r.id for r in visible] == [pending.id]
    assert {r.id for r in everything} == {pending.id, terminal.id}


async def test_list_by_tenant_is_newest_first_and_pageable(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    for index in range(3):
        await repo.insert_pending_or_none(
            _invitation(
                invitee_user_id=f"usr-{index}",
                created_at=_NOW + timedelta(hours=index),
            )
        )
    await session.commit()

    page = await repo.list_by_tenant(_TENANT_ID, limit=2, offset=0)

    assert [r.invitee_user_id for r in page] == ["usr-2", "usr-1"]


async def test_count_by_tenant_matches_the_filter(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    pending = await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-1"))
    terminal = await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-2"))
    await session.commit()
    assert pending is not None
    assert terminal is not None
    await repo.mark_status_if_pending(terminal.id, status=STATUS_DECLINED, responded_at=_NOW)
    await session.commit()

    assert await repo.count_by_tenant(_TENANT_ID) == 1
    assert await repo.count_by_tenant(_TENANT_ID, include_terminal=True) == 2


async def test_list_and_count_by_invitee(session: AsyncSession) -> None:
    repo = TenantInvitationRepository(session)
    await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-1"))
    await repo.insert_pending_or_none(
        _invitation(invitee_user_id="usr-1", tenant_id=_TENANT_ID + 1)
    )
    await repo.insert_pending_or_none(_invitation(invitee_user_id="usr-2"))
    await session.commit()

    inbox = await repo.list_by_invitee("usr-1")

    assert len(inbox) == 2
    assert await repo.count_pending_by_invitee("usr-1") == 2
    assert await repo.count_pending_by_invitee("usr-2") == 1
