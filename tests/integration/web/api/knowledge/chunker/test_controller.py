"""Web-layer tests for the chunker preview router.

The preview endpoint runs the adaptive chunker over supplied text with
no database involvement, so no service dependency is overridden; the
shared ``web_app`` + ``web_authed_client`` fixtures provide header auth.

Covers the route surface plus the success and guard paths the upstream
handler defines: valid text, blank text (400), oversized input (413),
and the framework's standard 422 for a malformed body.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web.api.knowledge.chunker.router import router
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.middleware.auth import require_auth

MAX_PREVIEW_CHARS = 64 * 1024


@pytest.fixture
def client(web_authed_client: TestClient) -> TestClient:
    """Alias the shared authed client; the preview needs no overrides."""
    return web_authed_client


# ── Route inventory + permission gates ───────────────────────────────


def test_router_declares_exactly_the_upstream_routes() -> None:
    found: set[tuple[str, str]] = set()
    for route in router.routes:
        methods: set[str] = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        for method in methods:
            found.add((method, path))
    assert found == {("POST", "/api/v1/chunker/preview")}


def test_endpoint_declares_auth_and_viewer_gates() -> None:
    route = next(r for r in router.routes if getattr(r, "path", "") == "/api/v1/chunker/preview")
    deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
    assert require_auth in deps  # type: ignore[attr-defined]

    roles: set[str] = set()
    for dep in getattr(route, "dependant", None).dependencies:  # type: ignore[union-attr]
        closure = getattr(dep.call, "__closure__", None)
        wrapped = getattr(dep.call, "__wrapped__", None)
        if closure is None and wrapped is None:
            continue
        for cell in closure or ():
            if isinstance(cell.cell_contents, str):
                roles.add(cell.cell_contents)
    assert "viewer" in roles


def test_role_gate_helper_is_the_shared_rbac_dependency() -> None:
    dep = make_role_dep("viewer")
    assert dep.__module__ == require_role_dep.__module__


# ── POST /chunker/preview ────────────────────────────────────────────


async def test_preview_returns_chunks_with_diagnostics(client: TestClient) -> None:
    resp = client.post("/api/v1/chunker/preview", json={"text": "Hello world. " * 20})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["selected_tier"] == "legacy"
    assert data["chunks"]
    assert data["stats"]["count"] == len(data["chunks"])
    assert data["profile"]["total_chars"] > 0
    chunk = data["chunks"][0]
    assert chunk["seq"] == 0
    assert chunk["size_chars"] == len(chunk["content"])
    assert chunk["size_tokens_approx"] >= 0


async def test_preview_honours_chunking_config(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/chunker/preview",
        json={
            "text": ("A short paragraph of text. " * 5 + "\n\n") * 8,
            "chunking_config": {"chunk_size": 40, "chunk_overlap": 10},
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["chunks"]) > 1


async def test_preview_blank_text_returns_400(client: TestClient) -> None:
    resp = client.post("/api/v1/chunker/preview", json={"text": "   "})

    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "text is empty" in body["error"]


async def test_preview_oversized_text_returns_413(client: TestClient) -> None:
    resp = client.post("/api/v1/chunker/preview", json={"text": "x" * (MAX_PREVIEW_CHARS + 1)})

    assert resp.status_code == 413
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "text exceeds preview limit"
    assert body["limit"] == MAX_PREVIEW_CHARS


async def test_preview_missing_text_returns_422(client: TestClient) -> None:
    resp = client.post("/api/v1/chunker/preview", json={"chunking_config": {}})

    assert resp.status_code == 422


async def test_preview_non_string_text_returns_422(client: TestClient) -> None:
    resp = client.post("/api/v1/chunker/preview", json={"text": 123})

    assert resp.status_code == 422
