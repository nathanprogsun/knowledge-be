"""Unit tests for OIDC service DI wiring.

The deps factory must inject the APP-scope ``OidcClient`` singleton from
the lifespan registry — never construct a fresh one per request (the
pooled ``httpx.AsyncClient`` underneath is the expensive part, and the
singleton is closed once at lifespan shutdown).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context.registry import LifeSpanService, get_oidc_client_from_lifespan
from src.common.oidc_client import OidcClient
from src.web.deps.auth import get_oidc_service


def _request_with(app: FastAPI) -> Request:
    return cast(Request, SimpleNamespace(app=app))


@pytest.mark.anyio
async def test_get_oidc_service_injects_lifespan_singleton() -> None:
    app = FastAPI()
    shared = OidcClient()
    app.state.lifespan_service = LifeSpanService(oidc_client=shared)
    session = cast(AsyncSession, object())

    first = get_oidc_service(_request_with(app), session)
    second = get_oidc_service(_request_with(app), session)

    assert first._oidc_client is shared
    assert second._oidc_client is shared
    await shared.aclose()


def test_get_oidc_client_from_lifespan_requires_registration() -> None:
    app = FastAPI()
    app.state.lifespan_service = LifeSpanService()

    with pytest.raises(RuntimeError, match="OidcClient is not initialized"):
        get_oidc_client_from_lifespan(app)
