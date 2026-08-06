"""Integration tests for `TenantMemberRepository` against a real Postgres.

The DDL mirrors `alembic/versions/0005_tenant_members.py` plus the
`users` table the member search joins against. The partial unique index
is created too — several tests depend on its exact predicate.

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

from src.db.dao.tenant_members_repository import (
    TenantMemberRepository,
    escape_like_pattern,
)
from src.db.models.tenants.tenant_members import TenantMember

_DROP_SQL = sqlalchemy.text("DROP TABLE IF EXISTS tenant_members, users CASCADE")

_CREATE_USERS_SQL = sqlalchemy.text(
    """
    CREATE TABLE users (
        id VARCHAR(36) PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        email VARCHAR(255) NOT NULL,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)

_CREATE_MEMBERS_SQL = sqlalchemy.text(
    """
    CREATE TABLE tenant_members (
        id BIGSERIAL PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        tenant_id BIGINT NOT NULL,
        role VARCHAR(20) NOT NULL DEFAULT 'contributor',
        status VARCHAR(20) NOT NULL DEFAULT 'active',
        invited_by VARCHAR(36),
        joined_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)

_CREATE_UNIQUE_INDEX_SQL = sqlalchemy.text(
    """
    CREATE UNIQUE INDEX idx_tenant_members_user_tenant_unique
        ON tenant_members(user_id, tenant_id)
        WHERE deleted_at IS NULL
    """
)

_SEED_USER_SQL = sqlalchemy.text(
    "INSERT INTO users (id, username, email) VALUES (:id, :username, :email)"
)

_TENANT_ID = 7
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
        await conn.execute(_CREATE_USERS_SQL)
        await conn.execute(_CREATE_MEMBERS_SQL)
        await conn.execute(_CREATE_UNIQUE_INDEX_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_SQL)
    await engine.dispose()


def _member(
    *,
    user_id: str,
    role: str = "contributor",
    tenant_id: int = _TENANT_ID,
    joined_at: datetime = _NOW,
    status: str = "active",
) -> TenantMember:
    return TenantMember(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        status=status,
        joined_at=joined_at,
        created_at=joined_at,
        updated_at=joined_at,
    )


async def _seed_user(
    session: AsyncSession,
    *,
    user_id: str,
    username: str,
    email: str,
) -> None:
    await session.execute(_SEED_USER_SQL.bindparams(id=user_id, username=username, email=email))
    await session.commit()


# ── insert_or_none ──────────────────────────────────────────────────


async def test_insert_assigns_id(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)

    stored = await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()

    assert stored is not None
    assert stored.id > 0


async def test_insert_returns_none_on_live_duplicate(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()

    duplicate = await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()

    assert duplicate is None


async def test_insert_succeeds_after_soft_delete(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.soft_delete(user_id="usr-1", tenant_id=_TENANT_ID, deleted_at=_NOW)
    await session.commit()

    revived = await repo.insert_live_or_none(_member(user_id="usr-1", role="admin"))
    await session.commit()

    assert revived is not None
    assert revived.role == "admin"


async def test_same_user_can_join_two_workspaces(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)

    first = await repo.insert_live_or_none(_member(user_id="usr-1", tenant_id=_TENANT_ID))
    second = await repo.insert_live_or_none(_member(user_id="usr-1", tenant_id=_TENANT_ID + 1))
    await session.commit()

    assert first is not None
    assert second is not None


# ── reads ───────────────────────────────────────────────────────────


async def test_find_membership_ignores_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.soft_delete(user_id="usr-1", tenant_id=_TENANT_ID, deleted_at=_NOW)
    await session.commit()

    assert await repo.find_membership(user_id="usr-1", tenant_id=_TENANT_ID) is None


async def test_list_by_tenant_orders_by_join_time_then_id(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    later = await repo.insert_live_or_none(
        _member(user_id="usr-2", joined_at=_NOW + timedelta(days=1))
    )
    earlier = await repo.insert_live_or_none(_member(user_id="usr-1", joined_at=_NOW))
    await session.commit()

    rows = await repo.list_by_tenant(_TENANT_ID)

    assert earlier is not None
    assert later is not None
    assert [r.id for r in rows] == [earlier.id, later.id]


async def test_list_by_user_spans_workspaces(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1", tenant_id=_TENANT_ID))
    await repo.insert_live_or_none(_member(user_id="usr-1", tenant_id=_TENANT_ID + 1))
    await session.commit()

    rows = await repo.list_by_user("usr-1")

    assert {r.tenant_id for r in rows} == {_TENANT_ID, _TENANT_ID + 1}


async def test_has_any_members_requires_an_active_row(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    assert await repo.has_any_members(_TENANT_ID) is False

    await repo.insert_live_or_none(_member(user_id="usr-1", status="invited"))
    await session.commit()
    assert await repo.has_any_members(_TENANT_ID) is False

    await repo.insert_live_or_none(_member(user_id="usr-2", status="active"))
    await session.commit()
    assert await repo.has_any_members(_TENANT_ID) is True


# ── search / paging ─────────────────────────────────────────────────


async def test_count_and_page_without_search(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    for index in range(3):
        await repo.insert_live_or_none(
            _member(user_id=f"usr-{index}", joined_at=_NOW + timedelta(hours=index))
        )
    await session.commit()

    assert await repo.count_by_tenant(_TENANT_ID) == 3
    page = await repo.list_page_by_tenant(_TENANT_ID, limit=2, offset=1)
    assert [r.user_id for r in page] == ["usr-1", "usr-2"]


async def test_search_matches_email_case_insensitively(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await _seed_user(session, user_id="usr-1", username="alice", email="Alice@Example.com")
    await _seed_user(session, user_id="usr-2", username="bob", email="bob@example.com")
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.insert_live_or_none(_member(user_id="usr-2"))
    await session.commit()

    assert await repo.count_by_tenant(_TENANT_ID, search="ALICE@example") == 1
    rows = await repo.list_page_by_tenant(
        _TENANT_ID,
        search="ALICE@example",
        limit=10,
        offset=0,
    )
    assert [r.user_id for r in rows] == ["usr-1"]


async def test_search_matches_username(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await _seed_user(session, user_id="usr-1", username="alice", email="a@example.com")
    await _seed_user(session, user_id="usr-2", username="bob", email="b@example.com")
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.insert_live_or_none(_member(user_id="usr-2"))
    await session.commit()

    rows = await repo.list_page_by_tenant(_TENANT_ID, search="bob", limit=10, offset=0)

    assert [r.user_id for r in rows] == ["usr-2"]


async def test_search_treats_wildcards_literally(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await _seed_user(session, user_id="usr-1", username="a%b", email="a@example.com")
    await _seed_user(session, user_id="usr-2", username="axxb", email="b@example.com")
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.insert_live_or_none(_member(user_id="usr-2"))
    await session.commit()

    rows = await repo.list_page_by_tenant(_TENANT_ID, search="a%b", limit=10, offset=0)

    assert [r.user_id for r in rows] == ["usr-1"]


async def test_search_skips_members_whose_user_row_is_deleted(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await _seed_user(session, user_id="usr-1", username="alice", email="a@example.com")
    await session.execute(
        sqlalchemy.text("UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = 'usr-1'")
    )
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()

    assert await repo.count_by_tenant(_TENANT_ID, search="alice") == 0


def test_escape_like_pattern_neutralises_wildcards() -> None:
    assert escape_like_pattern(r"a%b_c\d") == r"a\%b\_c\\d"


# ── owner counting / locking ────────────────────────────────────────


async def test_count_active_owners(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1", role="owner"))
    await repo.insert_live_or_none(_member(user_id="usr-2", role="owner", status="invited"))
    await repo.insert_live_or_none(_member(user_id="usr-3", role="admin"))
    await session.commit()

    assert await repo.count_active_owners(_TENANT_ID) == 1


async def test_count_other_active_owners_excludes_the_subject(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1", role="owner"))
    await session.commit()

    others = await repo.count_other_active_owners_for_update(
        tenant_id=_TENANT_ID,
        exclude_user_id="usr-1",
    )

    assert others == 0


async def test_count_other_active_owners_sees_the_second_owner(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1", role="owner"))
    await repo.insert_live_or_none(_member(user_id="usr-2", role="owner"))
    await session.commit()

    others = await repo.count_other_active_owners_for_update(
        tenant_id=_TENANT_ID,
        exclude_user_id="usr-1",
    )

    assert others == 1


# ── mutations ───────────────────────────────────────────────────────


async def test_update_role_touches_only_live_rows(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()

    affected = await repo.update_role(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role="admin",
        updated_at=_NOW,
    )
    await session.commit()

    assert affected == 1
    row = await repo.find_membership(user_id="usr-1", tenant_id=_TENANT_ID)
    assert row is not None
    assert row.role == "admin"


async def test_update_role_reports_zero_for_removed_member(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    await repo.insert_live_or_none(_member(user_id="usr-1"))
    await repo.soft_delete(user_id="usr-1", tenant_id=_TENANT_ID, deleted_at=_NOW)
    await session.commit()

    affected = await repo.update_role(
        user_id="usr-1",
        tenant_id=_TENANT_ID,
        role="admin",
        updated_at=_NOW,
    )

    assert affected == 0


async def test_soft_delete_stamps_both_timestamps(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    stored = await repo.insert_live_or_none(_member(user_id="usr-1"))
    await session.commit()
    assert stored is not None

    await repo.soft_delete(user_id="usr-1", tenant_id=_TENANT_ID, deleted_at=_NOW)
    await session.commit()

    row = await repo.find_by_primary_key({"id": stored.id}, exclude_deleted_or_archived=False)
    assert row is not None
    assert row.deleted_at == _NOW
    assert row.updated_at == _NOW
