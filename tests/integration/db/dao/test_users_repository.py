"""Integration tests for ``UserRepository`` against the real applied schema.

Tests insert unique rows per run; isolation relies on unique ids and
emails, not per-test DDL or cleanup. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError
from src.common.json import JsonObject
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _uid() -> str:
    return f"usr-{uuid.uuid4().hex[:12]}"


def _sample_row(
    *,
    id: str | None = None,
    username: str | None = None,
    email: str | None = None,
    password_hash: str = "bcrypt-digest",
    tenant_id: int | None = None,
    is_active: bool = True,
    is_system_admin: bool = False,
    preferences: JsonObject | None = None,
    created_at: datetime = _NOW,
) -> User:
    uid = id or _uid()
    uname = username or f"user-{uid}"
    addr = email or f"{uid}@example.com"
    return User(
        id=uid,
        username=uname,
        email=addr,
        password_hash=password_hash,
        avatar=None,
        tenant_id=tenant_id,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=is_system_admin,
        preferences=preferences if preferences is not None else {},
        created_at=created_at,
        updated_at=created_at,
    )


async def test_insert_then_find_by_id(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.id == row.id
    assert found.username == row.username
    assert found.email == row.email


async def test_find_by_email(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_email(row.email)
    assert found is not None
    assert found.id == row.id


async def test_find_by_username(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_username(row.username)
    assert found is not None
    assert found.id == row.id


async def test_find_missing_raises_not_found(session: AsyncSession) -> None:
    repo = UserRepository(session)
    with pytest.raises(NotFoundError):
        await repo.find_by_id(_uid())
    with pytest.raises(NotFoundError):
        await repo.find_by_email(f"{_uid()}@example.com")
    with pytest.raises(NotFoundError):
        await repo.find_by_username(_uid())


async def test_duplicate_insert_raises_conflict(session: AsyncSession) -> None:
    repo = UserRepository(session)
    shared_email = f"{_uid()}@example.com"
    row = _sample_row(email=shared_email)
    await repo.insert(row)
    await session.commit()

    duplicate = _sample_row(email=shared_email)
    with pytest.raises(ConflictError):
        await repo.insert(duplicate)


async def test_update_by_primary_key_replaces_fields(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    now = datetime.now(UTC)
    updated = await repo.update_by_primary_key_or_fail(
        {"id": row.id},
        {
            "username": f"updated-{row.id}",
            "email": f"updated-{row.id}@example.com",
            "avatar": "https://example.com/a.png",
            "can_access_all_tenants": True,
            "is_system_admin": True,
            "preferences": {"last_active_tenant_id": 42},
            "updated_at": now,
        },
    )
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.username == f"updated-{row.id}"
    assert found.email == f"updated-{row.id}@example.com"
    assert found.avatar == "https://example.com/a.png"
    assert found.can_access_all_tenants is True
    assert found.is_system_admin is True
    assert found.preferences == {"last_active_tenant_id": 42}
    assert updated.id == row.id


async def test_update_by_primary_key_missing_returns_none(session: AsyncSession) -> None:
    repo = UserRepository(session)
    now = datetime.now(UTC)
    result = await repo.update_by_primary_key(
        {"id": _uid()},
        {"username": "x", "updated_at": now},
    )
    assert result is None


async def test_update_by_primary_key_or_fail_missing_raises(
    session: AsyncSession,
) -> None:
    repo = UserRepository(session)
    now = datetime.now(UTC)
    with pytest.raises(NotFoundError):
        await repo.update_by_primary_key_or_fail(
            {"id": _uid()},
            {"username": "x", "updated_at": now},
        )


async def test_update_password(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row(password_hash="old-digest")
    await repo.insert(row)
    await session.commit()

    await repo.update_by_primary_key_or_fail(
        {"id": row.id},
        {"password_hash": "new-digest", "updated_at": datetime.now(UTC)},
    )
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.password_hash == "new-digest"


async def test_soft_delete_excludes_from_reads(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    now = datetime.now(UTC)
    await repo.update_by_primary_key_or_fail(
        {"id": row.id},
        {"deleted_at": now, "updated_at": now},
    )
    await session.commit()

    with pytest.raises(NotFoundError):
        await repo.find_by_id(row.id)
    with pytest.raises(NotFoundError):
        await repo.find_by_email(row.email)


async def test_soft_delete_missing_returns_none(session: AsyncSession) -> None:
    repo = UserRepository(session)
    now = datetime.now(UTC)
    result = await repo.update_by_primary_key(
        {"id": _uid()},
        {"deleted_at": now, "updated_at": now},
    )
    assert result is None


async def test_list_paginates(session: AsyncSession) -> None:
    repo = UserRepository(session)
    ids = [_uid() for _ in range(5)]
    base = datetime.now(UTC)
    for i, uid in enumerate(ids):
        await repo.insert(
            _sample_row(
                id=uid,
                username=f"u-{uid}",
                email=f"{uid}@example.com",
                created_at=base + timedelta(milliseconds=i),
            ),
        )
    await session.commit()

    # Fetch all users and verify our 5 are present and ordered newest-first.
    all_users = await repo.list(limit=1_000_000, offset=0)
    mine = [u for u in all_users if u.id in set(ids)]
    assert len(mine) == 5
    assert [u.id for u in mine] == list(reversed(ids))

    # Verify limit/offset: a small limit returns a non-overlapping page.
    page1 = await repo.list(limit=2, offset=0)
    page2 = await repo.list(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert {u.id for u in page1}.isdisjoint({u.id for u in page2})


async def test_list_excludes_soft_deleted(session: AsyncSession) -> None:
    repo = UserRepository(session)
    keep_id = _uid()
    drop_id = _uid()
    base = datetime.now(UTC)
    await repo.insert(
        _sample_row(
            id=keep_id,
            username=f"keep-{keep_id}",
            email=f"{keep_id}@example.com",
            created_at=base,
        ),
    )
    await repo.insert(
        _sample_row(
            id=drop_id,
            username=f"drop-{drop_id}",
            email=f"{drop_id}@example.com",
            created_at=base + timedelta(milliseconds=1),
        ),
    )
    await session.commit()

    now = datetime.now(UTC)
    await repo.update_by_primary_key_or_fail(
        {"id": drop_id},
        {"deleted_at": now, "updated_at": now},
    )
    await session.commit()

    listed = await repo.list(limit=1_000_000, offset=0)
    ids = {u.id for u in listed}
    assert keep_id in ids
    assert drop_id not in ids


async def test_is_system_admin_round_trips(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row(is_system_admin=True)
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.is_system_admin is True


async def test_preferences_round_trips(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row(preferences={"last_active_tenant_id": 7})
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.preferences == {"last_active_tenant_id": 7}


async def test_empty_preferences_default(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.preferences == {}
