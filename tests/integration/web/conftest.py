"""Web-integration test fixtures.

Builds on the ``tests/integration/conftest.py`` foundation: each web
test file that still injects fake repositories via
``dependency_overrides`` uses the ``web_app`` fixture here, which:

1. Reuses the session-scoped ``_engine`` and ``_seed_user`` from the
   parent conftest (real DB + real user for header auth).
2. Calls ``create_app()`` and attaches a ``LifeSpanService`` so the
   real ``get_async_session`` dep resolves.
3. Yields the app **before** applying dep overrides so the test file's
   own ``app`` fixture can call
   ``application.dependency_overrides[get_X_service] = ...``.

The ``authed_client`` fixture wraps the app in an ``AsyncClient`` with
the ``x-knowledge-*`` header trio set. Tests that have migrated to
fully real-DB integration can use the parent conftest's
``authed_client`` directly (no dep overrides).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
from src.app_context.registry import LifeSpanService
from src.db.base import DatabaseEngine
from tests.integration.conftest import _TEST_TENANT_ID, _TEST_USER_ID


@pytest_asyncio.fixture
async def web_app(
    _engine: DatabaseEngine,
    _seed_user: str,
) -> AsyncIterator[FastAPI]:
    """A real app with a ``LifeSpanService`` carrying the test engine.

    Test files override the ``get_X_service`` dep on this app to
    inject fake-backed services; header auth handles the principal
    without ``override_auth_gates``.
    """
    application = create_app()
    application.state.lifespan_service = LifeSpanService(db_engine=_engine)
    yield application
    application.state.lifespan_service = None


@pytest_asyncio.fixture
async def web_authed_client(web_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """``AsyncClient`` with ``x-knowledge-*`` headers against ``web_app``."""
    transport = ASGITransport(app=web_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        c.headers.update(
            {
                "x-knowledge-user-id": _TEST_USER_ID,
                "x-knowledge-tenant-id": str(_TEST_TENANT_ID),
                "x-knowledge-roles": "owner",
            }
        )
        yield c


__all__ = ["web_app", "web_authed_client"]
