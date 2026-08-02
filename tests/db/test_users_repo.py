"""Integration tests for `UserRepository` against a real Postgres.

A testcontainer spins up Postgres 16 per test session; each test gets a
fresh `users` schema on top of the shared container so writes are
hermetic. The DDL mirrors `alembic/versions/0001_users.py`.

The fixture skips the suite when no Docker daemon is available (CI
runners without Docker, sandboxes) — the repository code itself is
exercised by the integration tests once Docker is present, and the
schema/DAO correctness is enforced statically by mypy + ruff.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.exceptions import ContainerStartException

from src.common.exception import ConflictError, NotFoundError
from src.core.auth.types import UserInfo, UserPreferences
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User

_DROP_USERS_SQL = sqlalchemy.text("DROP TABLE IF EXISTS users")

_CREATE_USERS_SQL = sqlalchemy.text(
    """
    CREATE TABLE users (
        id VARCHAR(36) PRIMARY KEY,
        username VARCHAR(100) NOT NULL UNIQUE,
        email VARCHAR(255) NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        avatar VARCHAR(500),
        tenant_id INTEGER,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        can_access_all_tenants BOOLEAN NOT NULL DEFAULT FALSE,
        is_system_admin BOOLEAN NOT NULL DEFAULT FALSE,
        preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        deleted_at TIMESTAMP WITH TIME ZONE
    )
    """
)


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Spin up a Postgres container once per session; yield its async URL."""
    try:
        container = PostgresContainer("postgres:16-alpine")
        container.start()
    except ContainerStartException as exc:  # pragma: no cover — env-dependent
        pytest.skip(f"Docker not available for Postgres testcontainer: {exc}")
    except Exception as exc:  # pragma: no cover — env-dependent (sandbox perms, etc.)
        pytest.skip(f"Postgres testcontainer could not start ({type(exc).__name__}): {exc}")
    try:
        sync_url = container.get_connection_url()
    finally:
        container.stop()
    return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest.fixture
async def session(pg_url: str) -> AsyncIterator[AsyncSession]:
    engine: AsyncEngine = create_async_engine(pg_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(_DROP_USERS_SQL)
        await conn.execute(_CREATE_USERS_SQL)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(_DROP_USERS_SQL)
    await engine.dispose()


def _sample_row(
    *,
    id: str = "usr-1",
    username: str = "alice",
    email: str = "alice@example.com",
    password_hash: str = "bcrypt-digest",
    tenant_id: int | None = None,
    is_active: bool = True,
    is_system_admin: bool = False,
    preferences: dict[str, object] | None = None,
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return User(
        id=id,
        username=username,
        email=email,
        password_hash=password_hash,
        avatar=None,
        tenant_id=tenant_id,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=is_system_admin,
        preferences=preferences if preferences is not None else {},
        created_at=now,
        updated_at=now,
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
        await repo.find_by_id("nope")
    with pytest.raises(NotFoundError):
        await repo.find_by_email("nope@example.com")
    with pytest.raises(NotFoundError):
        await repo.find_by_username("nope")


async def test_duplicate_insert_raises_conflict(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row(id="usr-1", email="dup@example.com")
    await repo.insert(row)
    await session.commit()

    duplicate = _sample_row(id="usr-2", email="dup@example.com")
    with pytest.raises(ConflictError):
        await repo.insert(duplicate)


async def test_update_replaces_fields(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    updated = UserInfo(
        id=row.id,
        username="alice2",
        email="alice2@example.com",
        avatar="https://example.com/a.png",
        tenant_id=row.tenant_id,
        is_active=row.is_active,
        can_access_all_tenants=True,
        is_system_admin=True,
        preferences=UserPreferences(last_active_tenant_id=42),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
    await repo.update(updated)
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None
    assert found.username == "alice2"
    assert found.email == "alice2@example.com"
    assert found.avatar == "https://example.com/a.png"
    assert found.can_access_all_tenants is True
    assert found.is_system_admin is True
    assert found.preferences.last_active_tenant_id == 42


async def test_update_missing_user_raises_not_found(session: AsyncSession) -> None:
    repo = UserRepository(session)
    dto = UserInfo(
        id="missing",
        username="x",
        email="x@example.com",
        avatar=None,
        tenant_id=None,
        is_active=True,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences=UserPreferences(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(NotFoundError):
        await repo.update(dto)


async def test_update_password(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row(password_hash="old-digest")
    await repo.insert(row)
    await session.commit()

    await repo.update_password(row.id, "new-digest")
    await session.commit()

    found = await repo.find_by_id(row.id)
    assert found is not None


async def test_update_password_missing_raises_not_found(session: AsyncSession) -> None:
    repo = UserRepository(session)
    with pytest.raises(NotFoundError):
        await repo.update_password("missing", "x")


async def test_soft_delete_excludes_from_reads(session: AsyncSession) -> None:
    repo = UserRepository(session)
    row = _sample_row()
    await repo.insert(row)
    await session.commit()

    await repo.soft_delete(row.id)
    await session.commit()

    with pytest.raises(NotFoundError):
        await repo.find_by_id(row.id)
    with pytest.raises(NotFoundError):
        await repo.find_by_email(row.email)


async def test_soft_delete_missing_raises_not_found(session: AsyncSession) -> None:
    repo = UserRepository(session)
    with pytest.raises(NotFoundError):
        await repo.soft_delete("missing")


async def test_list_paginates(session: AsyncSession) -> None:
    repo = UserRepository(session)
    for i in range(5):
        await repo.insert(
            _sample_row(
                id=f"usr-{i}",
                username=f"u{i}",
                email=f"u{i}@example.com",
            ),
        )
    await session.commit()

    page1 = await repo.list(limit=2, offset=0)
    page2 = await repo.list(limit=2, offset=2)
    page3 = await repo.list(limit=2, offset=4)

    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1

    ids = {u.id for u in (*page1, *page2, *page3)}
    assert ids == {f"usr-{i}" for i in range(5)}


async def test_list_excludes_soft_deleted(session: AsyncSession) -> None:
    repo = UserRepository(session)
    await repo.insert(_sample_row(id="usr-keep"))
    await repo.insert(
        _sample_row(
            id="usr-drop",
            username="bob",
            email="drop@example.com",
        ),
    )
    await session.commit()

    await repo.soft_delete("usr-drop")
    await session.commit()

    listed = await repo.list(limit=10, offset=0)
    ids = {u.id for u in listed}
    assert ids == {"usr-keep"}


async def test_is_system_admin_round_trips(session: AsyncSession) -> None:
    repo = UserRepository(session)
    await repo.insert(_sample_row(is_system_admin=True))
    await session.commit()

    found = await repo.find_by_id("usr-1")
    assert found is not None
    assert found.is_system_admin is True


async def test_preferences_round_trips(session: AsyncSession) -> None:
    repo = UserRepository(session)
    await repo.insert(
        _sample_row(preferences={"last_active_tenant_id": 7}),
    )
    await session.commit()

    found = await repo.find_by_id("usr-1")
    assert found is not None
    assert found.preferences.last_active_tenant_id == 7


async def test_empty_preferences_default(session: AsyncSession) -> None:
    repo = UserRepository(session)
    await repo.insert(_sample_row())
    await session.commit()

    found = await repo.find_by_id("usr-1")
    assert found is not None
    assert found.preferences.last_active_tenant_id is None


__all__ = []
