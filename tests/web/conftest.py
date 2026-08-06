"""Web-test fixture re-exports.

Pytest only auto-discovers ``conftest.py`` files in the test file's
parent chain, so the ``web_app`` / ``web_authed_client`` fixtures
defined in ``tests/integration/web/conftest.py`` are not reachable
from tests that live under ``tests/web/``. This module re-exports
them so the migration pattern
(``from tests.integration.web.conftest import web_app, web_authed_client``)
works uniformly for tests in both directories.

The integration ``_engine`` / ``_seed_user`` fixtures are
session-scoped, which collides with the function-scoped event loop
pytest-asyncio uses by default (asyncpg binds connections to the
loop they were created in). We redefine them here as function-scoped
fixtures so each web test gets a fresh engine and a freshly seeded
user, sidestepping the cross-loop reuse error.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy

from src.db.base import DatabaseEngine
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User
from src.settings import get_settings, reset_settings_cache
from src.util.security import hash_password
from tests.conftest import pg_url
from tests.integration.web.conftest import web_app, web_authed_client

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


@pytest.fixture
def _integration_settings(pg_url: str) -> None:
    """Point ``Settings.database_url`` at the testcontainer for the test."""
    reset_settings_cache()
    os.environ["DATABASE_URL_OVERRIDE"] = pg_url
    reset_settings_cache()


@pytest_asyncio.fixture
async def _engine(_integration_settings: None) -> DatabaseEngine:
    """Function-scoped engine for web tests.

    Bypasses the session-scoped integration engine's loop binding
    issue when each test runs in its own event loop.
    """
    settings = get_settings()
    engine = DatabaseEngine(url=settings.database_url)
    try:
        yield engine
    finally:
        await engine.close()


@pytest_asyncio.fixture
async def _seed_user(_engine: DatabaseEngine) -> str:
    """Function-scoped seed for web tests."""
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


__all__ = [
    "web_app",
    "web_authed_client",
    "_engine",
    "_integration_settings",
    "_seed_user",
]