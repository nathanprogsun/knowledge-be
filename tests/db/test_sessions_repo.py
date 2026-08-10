"""Integration tests for ``SessionRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique session ids
and tenant ids. Tests commit explicitly. The ``sessions`` table carries
no foreign keys, so no parent rows need seeding.
"""

from __future__ import annotations

import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import DataError
from src.db.dao.session_repository import SessionRepository
from src.db.models.session import Session
from src.settings import get_settings

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# ``sessions.tenant_id`` is INTEGER (32-bit); tests mint ids from this
# registry so they stay inside the range (mirrors the custom-agent DAO
# tests, whose table carries the same INTEGER tenant_id column).
_used_tenant_ids: set[int] = set()


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the test session, safe for INTEGER."""
    while True:
        candidate = secrets.randbelow(2**31 - 1) + 1
        if candidate not in _used_tenant_ids:
            _used_tenant_ids.add(candidate)
            return candidate


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup).

    Uses ``NullPool`` so each test gets a fresh connection bound to its
    own function-scoped event loop, mirroring the DAO conftest fixture.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _sid() -> str:
    return f"ses-{uuid.uuid4().hex[:12]}"


def _session(
    *,
    tenant_id: int | None = None,
    user_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    is_pinned: bool = False,
    pinned_at: datetime | None = None,
    created_at: datetime = _NOW,
) -> Session:
    return Session(
        id=_sid(),
        tenant_id=tenant_id if tenant_id is not None else _int32_tenant_id(),
        user_id=user_id,
        title=title,
        description=description,
        is_pinned=is_pinned,
        pinned_at=pinned_at,
        created_at=created_at,
        updated_at=created_at,
    )


async def _store(session: AsyncSession, row: Session) -> Session:
    repo = SessionRepository(session)
    stored = await repo.create(row)
    await session.commit()
    return stored


# ── create / get ──────────────────────────────────────────────────────


async def test_create_persists_row(session: AsyncSession) -> None:
    stored = await _store(session, _session(title="alpha", description="first chat"))

    assert stored.id != ""
    assert stored.title == "alpha"
    assert stored.description == "first chat"
    assert stored.is_pinned is False
    assert stored.pinned_at is None


async def test_get_by_id(session: AsyncSession) -> None:
    stored = await _store(session, _session(title="alpha"))

    found = await SessionRepository(session).get_by_id(tenant_id=stored.tenant_id, id=stored.id)

    assert found is not None
    assert found.id == stored.id
    assert found.title == "alpha"


async def test_get_by_id_isolated_by_tenant(session: AsyncSession) -> None:
    stored = await _store(session, _session())
    other_tenant = _int32_tenant_id()

    assert await SessionRepository(session).get_by_id(tenant_id=other_tenant, id=stored.id) is None


async def test_get_by_id_returns_none_for_absent(session: AsyncSession) -> None:
    found = await SessionRepository(session).get_by_id(tenant_id=_int32_tenant_id(), id=_sid())

    assert found is None


async def test_get_by_id_for_user_scopes_to_owner(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    owner = f"usr-{uuid.uuid4().hex[:8]}"
    other = f"usr-{uuid.uuid4().hex[:8]}"
    stored = await _store(session, _session(user_id=owner))

    assert (
        await repo.get_by_id_for_user(tenant_id=stored.tenant_id, user_id=owner, id=stored.id)
    ) is not None
    assert (
        await repo.get_by_id_for_user(tenant_id=stored.tenant_id, user_id=other, id=stored.id)
    ) is None


async def test_get_by_id_for_user_sees_legacy_tenant_row(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    stored = await _store(session, _session(user_id=None))
    user = f"usr-{uuid.uuid4().hex[:8]}"

    found = await repo.get_by_id_for_user(tenant_id=stored.tenant_id, user_id=user, id=stored.id)

    assert found is not None


# ── list by tenant ─────────────────────────────────────────────────────


async def test_list_by_tenant_orders_by_recency(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    old = await _store(session, _session(tenant_id=tid, created_at=_NOW))
    new = await _store(
        session,
        _session(tenant_id=tid, created_at=_NOW + timedelta(days=1)),
    )

    rows = await repo.list_by_tenant(tenant_id=tid)

    assert [r.id for r in rows] == [new.id, old.id]


async def test_list_by_tenant_scopes_to_tenant(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    await _store(session, _session(tenant_id=tid))
    await _store(session, _session(tenant_id=_int32_tenant_id()))

    rows = await repo.list_by_tenant(tenant_id=tid)

    assert len(rows) == 1


async def test_list_by_tenant_filters_by_user(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    owner = f"usr-{uuid.uuid4().hex[:8]}"
    await _store(session, _session(tenant_id=tid, user_id=owner))
    await _store(session, _session(tenant_id=tid, user_id=f"usr-{uuid.uuid4().hex[:8]}"))

    rows = await repo.list_by_tenant(tenant_id=tid, user_id=owner)

    assert [r.user_id for r in rows] == [owner]


async def test_list_by_tenant_includes_legacy_rows_for_user(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    owner = f"usr-{uuid.uuid4().hex[:8]}"
    await _store(session, _session(tenant_id=tid, user_id=None))
    await _store(session, _session(tenant_id=tid, user_id=owner))

    rows = await repo.list_by_tenant(tenant_id=tid, user_id=owner)

    assert {r.user_id for r in rows} == {None, owner}


# ── list pinned ───────────────────────────────────────────────────────


async def test_list_pinned_returns_only_pinned(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    pinned = await _store(
        session,
        _session(tenant_id=tid, is_pinned=True, pinned_at=_NOW + timedelta(days=1)),
    )
    await _store(session, _session(tenant_id=tid))

    rows = await repo.list_pinned(tenant_id=tid)

    assert [r.id for r in rows] == [pinned.id]


async def test_list_pinned_orders_by_pinned_at(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    older = await _store(
        session,
        _session(tenant_id=tid, is_pinned=True, pinned_at=_NOW),
    )
    newer = await _store(
        session,
        _session(tenant_id=tid, is_pinned=True, pinned_at=_NOW + timedelta(days=1)),
    )

    rows = await repo.list_pinned(tenant_id=tid)

    assert [r.id for r in rows] == [newer.id, older.id]


# ── list paged ────────────────────────────────────────────────────────


async def test_list_paged_paginates_and_counts(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    for index in range(3):
        await _store(
            session,
            _session(tenant_id=tid, created_at=_NOW + timedelta(days=index)),
        )

    page, total = await repo.list_paged(tenant_id=tid, page=1, page_size=2)
    _, total_all = await repo.list_paged(tenant_id=tid, page=2, page_size=2)

    assert total == 3
    assert len(page) == 2
    assert total_all == 3


async def test_list_paged_filters_keyword(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    await _store(session, _session(tenant_id=tid, title="infrastructure design"))
    await _store(session, _session(tenant_id=tid, title="api notes"))

    page, total = await repo.list_paged(tenant_id=tid, keyword="infra")

    assert total == 1
    assert [r.title for r in page] == ["infrastructure design"]


async def test_list_paged_treats_wildcards_literally(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()
    literal = f"a%b-{uuid.uuid4().hex[:6]}"
    await _store(session, _session(tenant_id=tid, title=literal))
    await _store(session, _session(tenant_id=tid, title="axxb"))

    page, total = await repo.list_paged(tenant_id=tid, keyword=literal)

    assert total == 1
    assert [r.title for r in page] == [literal]


# ── update / pin / soft delete ───────────────────────────────────────


async def test_update_overwrites_mutable_columns(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    stored = await _store(session, _session(title="old", description="before"))

    renamed = stored.model_copy(
        update={
            "title": "new",
            "description": "after",
            "updated_at": _NOW + timedelta(days=1),
        }
    )
    persisted = await repo.update(renamed)
    await session.commit()

    assert persisted.title == "new"
    assert persisted.description == "after"
    # Immutable columns survive untouched.
    assert persisted.id == stored.id
    assert persisted.tenant_id == stored.tenant_id
    assert persisted.user_id == stored.user_id
    assert persisted.is_pinned is False


async def test_update_scoped_to_user(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    owner = f"usr-{uuid.uuid4().hex[:8]}"
    other = f"usr-{uuid.uuid4().hex[:8]}"
    stored = await _store(session, _session(user_id=owner))

    renamed = stored.model_copy(update={"title": "hijacked", "updated_at": _NOW})
    with pytest.raises(DataError):
        await repo.update(renamed, user_id=other)


async def test_set_pinned_then_unpin(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    stored = await _store(session, _session())
    now = _NOW + timedelta(days=1)

    pinned = await repo.set_pinned(tenant_id=stored.tenant_id, id=stored.id, pinned=True, now=now)
    await session.commit()
    pinned_row = await repo.get_by_id(tenant_id=stored.tenant_id, id=stored.id)
    assert pinned is True
    assert pinned_row is not None
    assert pinned_row.is_pinned is True
    assert pinned_row.pinned_at == now

    unpinned = await repo.set_pinned(
        tenant_id=stored.tenant_id, id=stored.id, pinned=False, now=now
    )
    await session.commit()
    unpinned_row = await repo.get_by_id(tenant_id=stored.tenant_id, id=stored.id)
    assert unpinned is True
    assert unpinned_row is not None
    assert unpinned_row.is_pinned is False
    assert unpinned_row.pinned_at is None


async def test_set_pinned_scoped_to_user(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    owner = f"usr-{uuid.uuid4().hex[:8]}"
    other = f"usr-{uuid.uuid4().hex[:8]}"
    stored = await _store(session, _session(user_id=owner))

    affected = await repo.set_pinned(
        tenant_id=stored.tenant_id, id=stored.id, pinned=True, now=_NOW, user_id=other
    )

    assert affected is False


async def test_soft_delete_hides_row(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    stored = await _store(session, _session())

    removed = await repo.soft_delete(
        tenant_id=stored.tenant_id, id=stored.id, now=_NOW + timedelta(days=1)
    )
    await session.commit()

    assert removed is True
    assert await repo.get_by_id(tenant_id=stored.tenant_id, id=stored.id) is None


async def test_soft_delete_reports_false_for_absent_row(session: AsyncSession) -> None:
    repo = SessionRepository(session)
    tid = _int32_tenant_id()

    removed = await repo.soft_delete(tenant_id=tid, id=_sid(), now=_NOW)

    assert removed is False
