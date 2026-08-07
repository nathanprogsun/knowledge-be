"""Shared fixtures and helpers for DAO integration tests.

DAO tests run against the real applied schema (no per-test DDL).
Isolation is by per-test generated tenant ids and unique entity ids;
tests commit explicitly and no cleanup is performed.

``make_test_tenant_id`` is hoisted to ``tests/integration/conftest.py``
so the web layer's ``make_user_org`` can share the same id registry;
this module re-exports it for DAO tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.db.base import DatabaseEngine
from tests.integration.conftest import make_test_tenant_id

__all__ = ["make_test_tenant_id"]


@pytest_asyncio.fixture
async def session(_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup).

    Uses ``NullPool`` so each test gets a fresh connection bound to its
    own function-scoped event loop, avoiding cross-loop reuse of
    connections from the session-scoped ``_engine`` pool.
    """
    engine = create_async_engine(_engine.engine.url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()
