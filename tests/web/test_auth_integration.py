"""API integration tests for the auth endpoints against a real Postgres.

Unlike ``test_auth_views.py`` (which overrides ``get_auth_service`` with
in-memory fakes), this module lets the real ``UserRepository`` +
``AuthTokenRepository`` execute against a testcontainer Postgres. Only
``get_async_session`` is overridden - to point at the container's engine
- so the full web -> service -> repo -> DB path is exercised.

The suite skips when Docker is unavailable (CI runners without Docker,
sandboxes).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import sqlalchemy
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.exceptions import ContainerStartException

from src.app_context.lifespan import create_app
from src.util.security import hash_password
from src.web.deps import get_async_session

_DROP_SQL = sqlalchemy.text("DROP TABLE IF EXISTS users, auth_tokens CASCADE")

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

_CREATE_AUTH_TOKENS_SQL = sqlalchemy.text(
    """
    CREATE TABLE auth_tokens (
        id VARCHAR(36) PRIMARY KEY,
        user_id VARCHAR(36) NOT NULL,
        token TEXT NOT NULL,
        token_type VARCHAR(50) NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        is_revoked BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_auth_tokens_user FOREIGN KEY (user_id)
            REFERENCES users(id) ON DELETE CASCADE
    )
    """
)

_SEED_USER_SQL = sqlalchemy.text(
    """
    INSERT INTO users (id, username, email, password_hash, tenant_id,
        is_active, preferences, created_at, updated_at)
    VALUES (:id, :username, :email, :password_hash, :tenant_id,
        :is_active, '{}'::jsonb, :created_at, :updated_at)
    """
)


@pytest.fixture(scope="module")
def pg_url() -> str:
    try:
        container = PostgresContainer("postgres:16-alpine")
        container.start()
    except (ContainerStartException, Exception) as exc:
        pytest.skip(f"Docker not available for Postgres testcontainer: {exc}")
    try:
        sync_url = container.get_connection_url()
    finally:
        container.stop()
    return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)


@pytest.fixture
async def engine(pg_url: str) -> AsyncIterator[AsyncEngine]:
    eng: AsyncEngine = create_async_engine(pg_url)
    async with eng.begin() as conn:
        await conn.execute(_DROP_SQL)
        await conn.execute(_CREATE_USERS_SQL)
        await conn.execute(_CREATE_AUTH_TOKENS_SQL)
    yield eng
    async with eng.begin() as conn:
        await conn.execute(_DROP_SQL)
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def app(session_factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    application = create_app()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_async_session] = _override_session
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed_user(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    email: str = "alice@example.com",
    password: str = "correct-horse",
    is_active: bool = True,
) -> str:
    now = datetime.now(UTC)
    async with session_factory() as s:
        await s.execute(
            _SEED_USER_SQL.bindparams(
                id="usr-1",
                username="alice",
                email=email,
                password_hash=hash_password(password),
                tenant_id=7,
                is_active=is_active,
                created_at=now,
                updated_at=now,
            )
        )
        await s.commit()
    return "usr-1"


# ── POST /auth/login ────────────────────────────────────────────────


async def test_login_writes_tokens_to_db(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_user(session_factory)
    resp = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["id"] == "usr-1"
    assert body["token"]
    assert body["refresh_token"]
    assert body["active_tenant"] is None

    # Two token rows (access + refresh) should be in the DB.
    async with session_factory() as s:
        rows = (
            (
                await s.execute(
                    sqlalchemy.text(
                        "SELECT token_type, is_revoked FROM auth_tokens WHERE user_id = :uid"
                    ).bindparams(uid="usr-1")
                )
            )
            .mappings()
            .all()
        )
    types = {r["token_type"] for r in rows}
    assert types == {"access_token", "refresh_token"}
    assert all(not r["is_revoked"] for r in rows)


async def test_login_wrong_password_db_unchanged(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_user(session_factory)
    resp = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wrong"},
    )
    assert resp.status_code == 401
    async with session_factory() as s:
        count = (await s.execute(sqlalchemy.text("SELECT COUNT(*) FROM auth_tokens"))).scalar()
    assert count == 0


# ── POST /auth/refresh ───────────────────────────────────────────────


async def test_refresh_revokes_old_token_in_db(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_user(session_factory)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    old_refresh = login.json()["refresh_token"]

    resp = await client.post(
        "/auth/refresh",
        json={"refreshToken": old_refresh},
    )
    assert resp.status_code == 200
    new_body = resp.json()
    assert new_body["access_token"] != login.json()["token"]

    # Old refresh row should be revoked; 4 rows total (2 from login + 2 from refresh).
    async with session_factory() as s:
        rows = (
            (
                await s.execute(
                    sqlalchemy.text(
                        "SELECT token, is_revoked FROM auth_tokens "
                        "WHERE user_id = :uid ORDER BY created_at"
                    ).bindparams(uid="usr-1")
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 4
    old_row = next(r for r in rows if r["token"] == old_refresh)
    assert old_row["is_revoked"] is True


# ── POST /auth/logout ────────────────────────────────────────────────


async def test_logout_revokes_all_tokens_in_db(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _seed_user(session_factory)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    token = login.json()["token"]

    resp = await client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    async with session_factory() as s:
        rows = (
            (
                await s.execute(
                    sqlalchemy.text(
                        "SELECT is_revoked FROM auth_tokens WHERE user_id = :uid"
                    ).bindparams(uid="usr-1")
                )
            )
            .scalars()
            .all()
        )
    assert all(rows)
    assert len(rows) == 2


async def test_logout_then_refresh_fails(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """After logout, the refresh token is revoked and can no longer be used."""
    await _seed_user(session_factory)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    refresh = login.json()["refresh_token"]
    access = login.json()["token"]

    await client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {access}"},
    )

    resp = await client.post(
        "/auth/refresh",
        json={"refreshToken": refresh},
    )
    assert resp.status_code == 401


__all__ = []
