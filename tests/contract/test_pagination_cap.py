"""Contract tests for the pagination ``page_size`` cap.

The cap mirrors the upstream handler's ``maxListPageSize = 100`` constant:
values above 100 must be rejected as a validation error at the request
boundary (the Pydantic ``Pagination`` field and each ``Query``-bound
router parameter). The test pins:

- ``page_size = 100`` accepted;
- ``page_size = 101`` rejected;
- ``page_size = 200`` rejected.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.common.pagination import Pagination

# ── Pydantic boundary ────────────────────────────────────────────────


def test_pagination_cap_accepts_one_hundred() -> None:
    """``page_size = 100`` is exactly the upstream cap and must be accepted."""
    assert Pagination(page_size=100).page_size == 100


def test_pagination_cap_rejects_one_hundred_one() -> None:
    """``page_size = 101`` is one above the upstream cap and must be rejected."""
    with pytest.raises(ValidationError):
        Pagination(page_size=101)


def test_pagination_cap_rejects_two_hundred() -> None:
    """``page_size = 200`` (a common over-the-cap value) must be rejected."""
    with pytest.raises(ValidationError):
        Pagination(page_size=200)


# ── FastAPI Query boundary ───────────────────────────────────────────
#
# The cap must also hold at the HTTP boundary for routers that bind
# ``page_size`` via ``Query(le=...)``. A minimal app with one such
# endpoint lets the contract travel through Pydantic + FastAPI together.


def _build_app() -> FastAPI:
    """A minimal app exposing ``GET /items`` with the same ``le=100`` cap."""
    from fastapi import Query

    app = FastAPI()

    @app.get("/items")
    async def list_items(
        page_size: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, int]:
        return {"page_size": page_size}

    return app


def test_query_cap_accepts_one_hundred() -> None:
    """``?page_size=100`` reaches the handler unchanged."""
    client = TestClient(_build_app())
    resp = client.get("/items", params={"page_size": 100})
    assert resp.status_code == 200
    assert resp.json() == {"page_size": 100}


def test_query_cap_rejects_one_hundred_one() -> None:
    """``?page_size=101`` is rejected by the request schema (HTTP 422)."""
    client = TestClient(_build_app())
    resp = client.get("/items", params={"page_size": 101})
    assert resp.status_code == 422


def test_query_cap_rejects_two_hundred() -> None:
    """``?page_size=200`` is rejected by the request schema (HTTP 422)."""
    client = TestClient(_build_app())
    resp = client.get("/items", params={"page_size": 200})
    assert resp.status_code == 422


__all__ = [
    "test_pagination_cap_accepts_one_hundred",
    "test_pagination_cap_rejects_one_hundred_one",
    "test_pagination_cap_rejects_two_hundred",
    "test_query_cap_accepts_one_hundred",
    "test_query_cap_rejects_one_hundred_one",
    "test_query_cap_rejects_two_hundred",
]
