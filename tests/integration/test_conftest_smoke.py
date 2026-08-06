"""Smoke test for the integration conftest foundation.

Verifies that the session-scoped engine, the manually-constructed
``LifeSpanService``, the real ``get_async_session`` dep, and the
header-based ``authed_client`` all wire up: a request to an
authenticated endpoint returns 200.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_authed_client_reaches_an_authenticated_endpoint(
    authed_client: AsyncClient,
) -> None:
    """The header-auth channel resolves the seeded user and returns 2xx.

    Hits ``/storage-backends/types`` (AuthDep + RoleViewerDep gated)
    which only reads the principal from ``request.state`` populated by
    the header-trust branch - no Bearer token required.
    """
    resp = await authed_client.get("/storage-backends/types")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "local" in body["data"]
