"""Integration-test conftest foundation.

Provides a session-scoped real ``DatabaseEngine`` against the
testcontainer Postgres, a real FastAPI app with a manually-constructed
``LifeSpanService`` (so the real lifespan's OIDC / MCP startup is
skipped), and an ``authed_client`` that sets the ``x-knowledge-*``
header trio so the real ``require_auth`` dependency runs end-to-end.

The principal is a real ``User`` row seeded into the test DB; the
header trust channel resolves it via ``UserRepository.find_by_id``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

import pytest
import pytest_asyncio
import sqlalchemy
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context.lifespan import create_app
from src.app_context.registry import LifeSpanService
from src.db.base import DatabaseEngine
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User
from src.settings import get_settings, reset_settings_cache
from src.util.security import hash_password
from src.web.deps.session import get_async_session
from tests.conftest import pg_url  # noqa: F401  - re-export for sub-conftests

# The header channel resolves this user id; tests assert against it.
_TEST_USER_ID = "usr-int-1"
_TEST_USER_EMAIL = "integration@example.com"
_TEST_USER_USERNAME = "integration"
_TEST_TENANT_ID = 1

_CREATE_USERS_SQL = sqlalchemy.text(
    """
    CREATE TABLE IF NOT EXISTS users (
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
def _integration_settings(pg_url: str) -> None:
    """Point ``Settings.database_url`` at the testcontainer for the session."""
    import os

    reset_settings_cache()
    os.environ["DATABASE_URL_OVERRIDE"] = pg_url
    reset_settings_cache()


@pytest.fixture(scope="session")
def _engine(_integration_settings: None) -> DatabaseEngine:
    """Session-scoped engine against the testcontainer Postgres.

    The engine is created synchronously (the asyncpg pool is lazy); the
    first ``async with session_factory()`` binds it to the calling
    loop. Tests use function-scoped loops so each test's session binds
    a fresh connection from the shared pool.
    """
    settings = get_settings()
    engine = DatabaseEngine(url=settings.database_url)
    return engine


@pytest_asyncio.fixture
async def _seed_user(_engine: DatabaseEngine) -> str:
    """Seed the integration-test user (function-scoped, idempotent)."""
    # Ensure the ``users`` table exists (idempotent).
    async with _engine.engine.begin() as conn:
        await conn.execute(_CREATE_USERS_SQL)

    now = datetime.now(UTC)
    async with _engine.session_factory() as session:
        repo = UserRepository(session)
        try:
            existing = await repo.find_by_id(_TEST_USER_ID)
            if existing is not None:
                return _TEST_USER_ID
        except Exception:
            await session.rollback()
        user = User(
            id=_TEST_USER_ID,
            username=_TEST_USER_USERNAME,
            email=_TEST_USER_EMAIL,
            password_hash=hash_password("integration-pw"),
            tenant_id=_TEST_TENANT_ID,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        try:
            await repo.insert(user)
        except Exception:
            await session.rollback()
        else:
            await session.commit()
    return _TEST_USER_ID


@pytest_asyncio.fixture
async def app(_engine: DatabaseEngine, _seed_user: str) -> AsyncIterator[FastAPI]:
    """Build the real app with a minimal ``LifeSpanService``.

    The real lifespan startup is skipped (it would try to spin up
    ``OidcClient`` and ``MCPConnectionManager``); instead a
    ``LifeSpanService`` carrying only the test ``db_engine`` is
    attached so ``get_db_engine_from_lifespan`` resolves.
    """
    application = create_app()
    lifespan_service = LifeSpanService(db_engine=_engine)
    application.state.lifespan_service = lifespan_service
    yield application
    application.state.lifespan_service = None


@pytest_asyncio.fixture
async def authed_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """An ``AsyncClient`` that authenticates via the ``x-knowledge-*`` headers.

    The headers trigger the header-trust branch in ``require_auth``,
    which looks up ``_TEST_USER_ID`` in the DB and populates
    ``request.state`` with the principal + tenant id 1.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.headers.update(
            {
                "x-knowledge-user-id": _TEST_USER_ID,
                "x-knowledge-tenant-id": str(_TEST_TENANT_ID),
                "x-knowledge-roles": "owner",
            }
        )
        yield c


@pytest_asyncio.fixture
async def session(_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """A per-test ``AsyncSession`` against the integration DB.

    Tests use this to seed / assert on real DB rows directly.
    """
    async with _engine.session_factory() as s:
        yield s
        await s.rollback()


__all__ = ["app", "authed_client", "pg_url", "session"]
