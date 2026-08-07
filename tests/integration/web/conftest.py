"""Web-integration test fixtures.

Builds on the ``tests/integration/conftest.py`` foundation: each web
test file that still injects fake repositories via
``dependency_overrides`` uses the ``web_app`` fixture here, which:

1. Reuses the session-scoped ``_engine`` from the parent conftest
   (real DB) plus the per-test ``admin_user`` so header auth resolves
   a freshly-minted principal.
2. Calls ``create_app()`` and attaches a ``LifeSpanService`` so the
   real ``get_async_session`` dep resolves.
3. Replaces the lifespan context with a no-op so TestClient does not
   run the real startup.
4. Yields the app **before** applying dep overrides so the test file's
   own ``app`` fixture can call
   ``application.dependency_overrides[get_X_service] = ...``.

The ``web_authed_client`` fixture wraps the app in a ``TestClient``
with the ``x-knowledge-*`` header trio set from the same
``admin_user`` used by the parent conftest's ``authed_client``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app_context.lifespan import create_app
from src.app_context.registry import LifeSpanService
from src.core.tenants.member_service import ROLE_OWNER
from src.db.base import DatabaseEngine
from tests.integration.conftest import _noop_lifespan


@pytest_asyncio.fixture
async def web_app(
    _engine: DatabaseEngine,
) -> AsyncIterator[FastAPI]:
    """A real app with a ``LifeSpanService`` carrying the test engine.

    Test files override the ``get_X_service`` dep on this app to
    inject fake-backed services; header auth handles the principal
    without ``override_auth_gates``. The lifespan context is replaced
    with a no-op so TestClient does not run the real startup.
    """
    application = create_app()
    application.state.lifespan_service = LifeSpanService(db_engine=_engine)
    application.router.lifespan_context = _noop_lifespan
    yield application
    application.state.lifespan_service = None


@pytest_asyncio.fixture
async def web_authed_client(
    web_app: FastAPI,
    admin_user: tuple[int, int],
) -> AsyncIterator[TestClient]:
    """``TestClient`` with ``x-knowledge-*`` headers against ``web_app``.

    The headers are sourced from the freshly-minted ``admin_user`` so
    every web test resolves a real ``User`` row + a real ``Tenant``
    row created by ``make_user_org``. The ``with`` block is required:
    without it asyncpg raises a cross-loop ``RuntimeError`` because
    TestClient runs requests in its own portal loop.
    """
    user_id, tenant_id = admin_user
    with TestClient(app=web_app) as c:
        c.headers.update(
            {
                "x-knowledge-user-id": user_id,
                "x-knowledge-tenant-id": str(tenant_id),
                "x-knowledge-roles": ROLE_OWNER,
            }
        )
        yield c


__all__ = ["web_app", "web_authed_client"]
