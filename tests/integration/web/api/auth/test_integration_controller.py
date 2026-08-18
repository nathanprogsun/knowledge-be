"""API integration tests for the auth endpoints against a real Postgres.

Unlike ``test_auth_views.py`` (which overrides ``get_auth_service`` with
in-memory fakes), this module lets the real ``UserRepository`` +
``AuthTokenRepository`` execute against the real Postgres instance.
Only ``get_async_session`` is overridden - to point at a dedicated
engine - so the full web -> service -> repo -> DB path is exercised.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
import sqlalchemy
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.app_context.lifespan import create_app
from src.settings import get_settings, reset_settings_cache
from src.util.security import hash_password
from src.web.deps import get_async_session
from tests.integration.conftest import _noop_lifespan

_SEED_USER_SQL = sqlalchemy.text(
    """
    INSERT INTO users (id, username, email, password_hash, tenant_id,
        is_active, preferences, created_at, updated_at)
    VALUES (:id, :username, :email, :password_hash, :tenant_id,
        :is_active, '{}'::jsonb, :created_at, :updated_at)
    """
)


@pytest.fixture
def _reset_settings() -> None:
    reset_settings_cache()


@pytest.fixture
async def engine(_reset_settings: None) -> AsyncIterator[AsyncEngine]:
    settings = get_settings()
    eng: AsyncEngine = create_async_engine(settings.database_url, poolclass=NullPool)
    yield eng
    await eng.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def app(session_factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    application = create_app()
    application.router.lifespan_context = _noop_lifespan

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
async def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app=app, base_url="http://test") as c:
        yield c


async def _seed_user(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    password: str = "correct-horse",
    is_active: bool = True,
) -> tuple[str, str]:
    """Insert a unique user and return ``(user_id, email)``."""
    uid = f"usr-{uuid.uuid4().hex[:12]}"
    email = f"{uid}@test.example"
    now = datetime.now(UTC)
    async with session_factory() as s:
        await s.execute(
            _SEED_USER_SQL.bindparams(
                id=uid,
                username=uid,
                email=email,
                password_hash=hash_password(password),
                tenant_id=7,
                is_active=is_active,
                created_at=now,
                updated_at=now,
            )
        )
        await s.commit()
    return uid, email


# ── POST /auth/login ────────────────────────────────────────────────


async def test_login_writes_tokens_to_db(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    uid, email = await _seed_user(session_factory)
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "correct-horse"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["id"] == uid
    assert body["token"]
    assert body["refresh_token"]
    assert body["active_tenant"] is None

    async with session_factory() as s:
        rows = (
            (
                await s.execute(
                    sqlalchemy.text(
                        "SELECT token_type, is_revoked FROM auth_tokens WHERE user_id = :uid"
                    ).bindparams(uid=uid)
                )
            )
            .mappings()
            .all()
        )
    types = {r["token_type"] for r in rows}
    assert types == {"access_token", "refresh_token"}
    assert all(not r["is_revoked"] for r in rows)


async def test_login_wrong_password_db_unchanged(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    uid, email = await _seed_user(session_factory)
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "wrong"},
    )
    assert resp.status_code == 401
    async with session_factory() as s:
        count = (
            await s.execute(
                sqlalchemy.text("SELECT COUNT(*) FROM auth_tokens WHERE user_id = :uid").bindparams(
                    uid=uid
                )
            )
        ).scalar()
    assert count == 0


# ── POST /auth/refresh ───────────────────────────────────────────────


async def test_refresh_revokes_old_token_in_db(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    uid, email = await _seed_user(session_factory)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "correct-horse"},
    )
    old_refresh = login.json()["refresh_token"]

    resp = client.post(
        "/api/v1/auth/refresh",
        json={"refreshToken": old_refresh},
    )
    assert resp.status_code == 200
    new_body = resp.json()
    assert new_body["access_token"] != login.json()["token"]

    async with session_factory() as s:
        rows = (
            (
                await s.execute(
                    sqlalchemy.text(
                        "SELECT token, is_revoked FROM auth_tokens "
                        "WHERE user_id = :uid ORDER BY created_at"
                    ).bindparams(uid=uid)
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
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    uid, email = await _seed_user(session_factory)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "correct-horse"},
    )
    token = login.json()["token"]

    resp = client.post(
        "/api/v1/auth/logout",
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
                    ).bindparams(uid=uid)
                )
            )
            .scalars()
            .all()
        )
    assert all(rows)
    assert len(rows) == 2


async def test_logout_then_refresh_fails(
    client: TestClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """After logout, the refresh token is revoked and can no longer be used."""
    _, email = await _seed_user(session_factory)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "correct-horse"},
    )
    refresh = login.json()["refresh_token"]
    access = login.json()["token"]

    client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {access}"},
    )

    resp = client.post(
        "/api/v1/auth/refresh",
        json={"refreshToken": refresh},
    )
    assert resp.status_code == 401


__all__ = []
