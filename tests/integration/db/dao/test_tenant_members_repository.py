"""Integration tests for ``TenantMemberRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique user ids and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.tenant_members_repository import (
    TenantMemberRepository,
    escape_like_pattern,
)
from src.db.models.tenants.tenant_members import TenantMember
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_SEED_USER_SQL = sqlalchemy.text(
    "INSERT INTO users (id, username, email, password_hash) "
    "VALUES (:id, :username, :email, 'placeholder-hash') "
    "ON CONFLICT (id) DO NOTHING"
)


def _uid() -> str:
    return f"usr-{uuid.uuid4().hex[:12]}"


def _member(
    *,
    user_id: str | None = None,
    role: str = "contributor",
    tenant_id: int | None = None,
    joined_at: datetime = _NOW,
    status: str = "active",
) -> TenantMember:
    return TenantMember(
        user_id=user_id or _uid(),
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
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

    stored = await repo.insert_live_or_none(_member())
    await session.commit()

    assert stored is not None
    assert stored.id > 0


async def test_insert_returns_none_on_live_duplicate(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    member = _member()
    await repo.insert_live_or_none(member)
    await session.commit()

    duplicate = await repo.insert_live_or_none(member)
    await session.commit()

    assert duplicate is None


async def test_insert_succeeds_after_soft_delete(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    member = _member()
    await repo.insert_live_or_none(member)
    await repo.soft_delete(user_id=member.user_id, tenant_id=member.tenant_id, deleted_at=_NOW)
    await session.commit()

    revived = await repo.insert_live_or_none(
        _member(user_id=member.user_id, tenant_id=member.tenant_id, role="admin")
    )
    await session.commit()

    assert revived is not None
    assert revived.role == "admin"


async def test_same_user_can_join_two_workspaces(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    uid = _uid()
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()

    first = await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid_a))
    second = await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid_b))
    await session.commit()

    assert first is not None
    assert second is not None


# ── reads ───────────────────────────────────────────────────────────


async def test_find_membership_ignores_soft_deleted_rows(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    member = _member()
    await repo.insert_live_or_none(member)
    await repo.soft_delete(user_id=member.user_id, tenant_id=member.tenant_id, deleted_at=_NOW)
    await session.commit()

    assert await repo.find_membership(user_id=member.user_id, tenant_id=member.tenant_id) is None


async def test_list_by_tenant_orders_by_join_time_then_id(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    later = await repo.insert_live_or_none(
        _member(user_id=_uid(), tenant_id=tid, joined_at=_NOW + timedelta(days=1))
    )
    earlier = await repo.insert_live_or_none(_member(user_id=_uid(), tenant_id=tid, joined_at=_NOW))
    await session.commit()

    rows = await repo.list_by_tenant(tid)

    assert earlier is not None
    assert later is not None
    assert [r.id for r in rows] == [earlier.id, later.id]


async def test_list_by_user_spans_workspaces(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    uid = _uid()
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid_a))
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid_b))
    await session.commit()

    rows = await repo.list_by_user(uid)

    assert {r.tenant_id for r in rows} == {tid_a, tid_b}


async def test_has_any_members_requires_an_active_row(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    assert await repo.has_any_members(tid) is False

    await repo.insert_live_or_none(_member(user_id=_uid(), tenant_id=tid, status="invited"))
    await session.commit()
    assert await repo.has_any_members(tid) is False

    await repo.insert_live_or_none(_member(user_id=_uid(), tenant_id=tid, status="active"))
    await session.commit()
    assert await repo.has_any_members(tid) is True


# ── search / paging ─────────────────────────────────────────────────


async def test_count_and_page_without_search(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    for index in range(3):
        await repo.insert_live_or_none(
            _member(user_id=_uid(), tenant_id=tid, joined_at=_NOW + timedelta(hours=index))
        )
    await session.commit()

    assert await repo.count_by_tenant(tid) == 3
    page = await repo.list_page_by_tenant(tid, limit=2, offset=1)
    assert len(page) == 2


async def test_search_matches_email_case_insensitively(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid_a = _uid()
    uid_b = _uid()
    await _seed_user(session, user_id=uid_a, username=uid_a, email=f"{uid_a}@Example.com")
    await _seed_user(session, user_id=uid_b, username=uid_b, email=f"{uid_b}@example.com")
    await repo.insert_live_or_none(_member(user_id=uid_a, tenant_id=tid))
    await repo.insert_live_or_none(_member(user_id=uid_b, tenant_id=tid))
    await session.commit()

    assert await repo.count_by_tenant(tid, search=f"{uid_a}@Example") == 1
    rows = await repo.list_page_by_tenant(tid, search=f"{uid_a}@Example", limit=10, offset=0)
    assert [r.user_id for r in rows] == [uid_a]


async def test_search_matches_username(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid_a = _uid()
    uid_b = _uid()
    await _seed_user(session, user_id=uid_a, username=uid_a, email=f"{uid_a}@example.com")
    await _seed_user(session, user_id=uid_b, username=uid_b, email=f"{uid_b}@example.com")
    await repo.insert_live_or_none(_member(user_id=uid_a, tenant_id=tid))
    await repo.insert_live_or_none(_member(user_id=uid_b, tenant_id=tid))
    await session.commit()

    rows = await repo.list_page_by_tenant(tid, search=uid_b, limit=10, offset=0)

    assert [r.user_id for r in rows] == [uid_b]


async def test_search_treats_wildcards_literally(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid_a = _uid()
    uid_b = _uid()
    wildcard_name = f"a%b-{uuid.uuid4().hex[:6]}"
    await _seed_user(session, user_id=uid_a, username=wildcard_name, email=f"{uid_a}@example.com")
    await _seed_user(
        session,
        user_id=uid_b,
        username=f"axxb-{uuid.uuid4().hex[:6]}",
        email=f"{uid_b}@example.com",
    )
    await repo.insert_live_or_none(_member(user_id=uid_a, tenant_id=tid))
    await repo.insert_live_or_none(_member(user_id=uid_b, tenant_id=tid))
    await session.commit()

    rows = await repo.list_page_by_tenant(tid, search=wildcard_name, limit=10, offset=0)

    assert [r.user_id for r in rows] == [uid_a]


async def test_search_skips_members_whose_user_row_is_deleted(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid = _uid()
    await _seed_user(session, user_id=uid, username=uid, email=f"{uid}@example.com")
    await session.execute(
        sqlalchemy.text(f"UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = '{uid}'")
    )
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid))
    await session.commit()

    assert await repo.count_by_tenant(tid, search=uid) == 0


def test_escape_like_pattern_neutralises_wildcards() -> None:
    assert escape_like_pattern(r"a%b_c\d") == r"a\%b\_c\\d"


# ── owner counting / locking ────────────────────────────────────────


async def test_count_active_owners(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    await repo.insert_live_or_none(_member(user_id=_uid(), tenant_id=tid, role="owner"))
    await repo.insert_live_or_none(
        _member(user_id=_uid(), tenant_id=tid, role="owner", status="invited")
    )
    await repo.insert_live_or_none(_member(user_id=_uid(), tenant_id=tid, role="admin"))
    await session.commit()

    assert await repo.count_active_owners(tid) == 1


async def test_count_other_active_owners_excludes_the_subject(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid = _uid()
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid, role="owner"))
    await session.commit()

    others = await repo.count_other_active_owners_for_update(
        tenant_id=tid,
        exclude_user_id=uid,
    )

    assert others == 0


async def test_count_other_active_owners_sees_the_second_owner(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid_a = _uid()
    uid_b = _uid()
    await repo.insert_live_or_none(_member(user_id=uid_a, tenant_id=tid, role="owner"))
    await repo.insert_live_or_none(_member(user_id=uid_b, tenant_id=tid, role="owner"))
    await session.commit()

    others = await repo.count_other_active_owners_for_update(
        tenant_id=tid,
        exclude_user_id=uid_a,
    )

    assert others == 1


# ── mutations ───────────────────────────────────────────────────────


async def test_update_role_touches_only_live_rows(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid = _uid()
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid))
    await session.commit()

    affected = await repo.update_role(
        user_id=uid,
        tenant_id=tid,
        role="admin",
        updated_at=_NOW,
    )
    await session.commit()

    assert affected == 1
    row = await repo.find_membership(user_id=uid, tenant_id=tid)
    assert row is not None
    assert row.role == "admin"


async def test_update_role_reports_zero_for_removed_member(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid = _uid()
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid))
    await repo.soft_delete(user_id=uid, tenant_id=tid, deleted_at=_NOW)
    await session.commit()

    affected = await repo.update_role(
        user_id=uid,
        tenant_id=tid,
        role="admin",
        updated_at=_NOW,
    )

    assert affected == 0


async def test_soft_delete_stamps_both_timestamps(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    tid = make_test_tenant_id()
    uid = _uid()
    stored = await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid))
    await session.commit()
    assert stored is not None

    await repo.soft_delete(user_id=uid, tenant_id=tid, deleted_at=_NOW)
    await session.commit()

    row = await repo.find_by_primary_key({"id": stored.id}, exclude_deleted_or_archived=False)
    assert row is not None
    assert row.deleted_at == _NOW
    assert row.updated_at == _NOW


# ── tenant isolation ────────────────────────────────────────────────


async def test_find_membership_isolated_by_tenant(session: AsyncSession) -> None:
    repo = TenantMemberRepository(session)
    uid = _uid()
    tid_a = make_test_tenant_id()
    tid_b = make_test_tenant_id()
    await repo.insert_live_or_none(_member(user_id=uid, tenant_id=tid_a))
    await session.commit()

    assert await repo.find_membership(user_id=uid, tenant_id=tid_a) is not None
    assert await repo.find_membership(user_id=uid, tenant_id=tid_b) is None
