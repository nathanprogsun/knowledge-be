"""Web-layer tests for the model router.

Exercises the router over HTTP via ``httpx.AsyncClient`` with
``get_model_service`` overridden to use a real ``ModelService`` backed
by the shared in-memory fake repository, so the full web -> service
path runs without a database.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``x-knowledge-*`` header trio.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.core.infra.models.service.model_service import ModelService
from src.web.deps.infra_models import get_model_service
from tests.integration.web.conftest import web_app, web_authed_client  # noqa: F401
from tests.unit.fakes.models import FakeModelRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> FakeModelRepository:
    return FakeModelRepository()


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    repo: FakeModelRepository,
) -> FastAPI:
    """Override ``get_model_service`` on the shared web app."""

    def _override_service() -> ModelService:
        return ModelService(models_repo=repo)  # type: ignore[arg-type]

    web_app.dependency_overrides[get_model_service] = _override_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: AsyncClient) -> AsyncClient:  # noqa: ARG001
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _create_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "gpt-4o",
        "type": "KnowledgeQA",
        "source": "openai",
        "description": "Default chat model",
        "parameters": {
            "base_url": "https://api.openai.com/v1",
            "provider": "openai",
            "api_key": "sk-secret",
            "interface_type": "openai",
        },
    }
    body.update(overrides)
    return body


# ── POST /models ────────────────────────────────────────────────────


async def test_create_model_returns_201_envelope(
    client: AsyncClient,
    repo: FakeModelRepository,
) -> None:
    resp = await client.post("/models", json=_create_body())

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"]["name"] == "gpt-4o"
    assert payload["data"]["status"] == "active"
    assert payload["data"]["id"] in repo.rows


async def test_create_model_strips_credential_fields(
    client: AsyncClient,
) -> None:
    resp = await client.post("/models", json=_create_body())

    assert resp.status_code == 201
    params = resp.json()["data"]["parameters"]
    # Sensitive fields are redacted with the wire placeholder
    # ``sk-***`` (mirrors Go's ``dto.NewModelResponse``); non-credential
    # fields survive the projection verbatim. ``app_secret`` was not
    # supplied in the fixture so it round-trips as ``""`` (no
    # placeholder — Go's same convention).
    assert params["api_key"] == "sk-***"
    assert params["app_secret"] == ""
    assert params["provider"] == "openai"
    assert params["base_url"] == "https://api.openai.com/v1"


async def test_create_model_rejects_blank_name(client: AsyncClient) -> None:
    resp = await client.post("/models", json=_create_body(name="   "))

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model.name_required"


# ── GET /models ─────────────────────────────────────────────────────


async def test_list_models_returns_tenant_rows(client: AsyncClient) -> None:
    await client.post("/models", json=_create_body(name="chat-1"))
    await client.post("/models", json=_create_body(name="chat-2"))

    resp = await client.get("/models")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert len(payload["data"]) == 2


async def test_list_models_filters_by_type(client: AsyncClient) -> None:
    await client.post("/models", json=_create_body(name="chat", type="KnowledgeQA"))
    await client.post("/models", json=_create_body(name="embed", type="Embedding"))

    resp = await client.get("/models?type=Embedding")

    assert resp.status_code == 200
    payload = resp.json()
    assert all(item["type"] == "Embedding" for item in payload["data"])


# ── GET /models/{id} ────────────────────────────────────────────────


async def test_get_model_returns_one_model(client: AsyncClient) -> None:
    created = await client.post("/models", json=_create_body())

    model_id = created.json()["data"]["id"]
    resp = await client.get(f"/models/{model_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["id"] == model_id
    assert payload["data"]["name"] == "gpt-4o"


async def test_get_model_returns_404_when_absent(client: AsyncClient) -> None:
    resp = await client.get("/models/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


# ── PUT /models/{id} ────────────────────────────────────────────────


async def test_update_model_patches_supplied_columns(
    client: AsyncClient,
) -> None:
    created = await client.post("/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = await client.put(
        f"/models/{model_id}",
        json={"name": "gpt-4-turbo", "description": "renamed"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["name"] == "gpt-4-turbo"
    assert payload["data"]["description"] == "renamed"


async def test_update_model_returns_404_when_absent(client: AsyncClient) -> None:
    resp = await client.put("/models/does-not-exist", json={"name": "x"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


# ── DELETE /models/{id} ─────────────────────────────────────────────


async def test_delete_model_removes_row(
    client: AsyncClient,
    repo: FakeModelRepository,
) -> None:
    created = await client.post("/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = await client.delete(f"/models/{model_id}")

    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["message"] == "Model deleted"
    assert model_id not in repo.rows


# ── GET /models/providers ───────────────────────────────────────────


async def test_list_providers_returns_catalog(client: AsyncClient) -> None:
    resp = await client.get("/models/providers")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    # The static catalog mirrors ``docs/api/model.md``'s supported
    # providers list; assert only that the wire shape (value / label /
    # defaultUrls / modelTypes) is intact, not exact provider counts.
    providers = payload["data"]
    assert isinstance(providers, list) and providers
    sample = providers[0]
    assert {"value", "label", "defaultUrls", "modelTypes"} <= set(sample.keys())
    # Ali­yun / DashScope is the canonical first example from the Go docs.
    values = {p["value"] for p in providers}
    assert "aliyun" in values


# ── POST /models/{id}/debug ─────────────────────────────────────────


async def test_debug_model_returns_envelope(client: AsyncClient) -> None:
    created = await client.post("/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = await client.post(
        f"/models/{model_id}/debug",
        data={"input": "hello"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["request"]["input"] == "hello"
    assert data["request"]["model_id"] == model_id


async def test_debug_model_returns_404_when_absent(client: AsyncClient) -> None:
    resp = await client.post("/models/does-not-exist/debug", data={"input": "hi"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


async def test_debug_model_rejects_oversized_input(
    client: AsyncClient,
) -> None:
    created = await client.post("/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    oversize = "x" * (65 * 1024)
    resp = await client.post(
        f"/models/{model_id}/debug",
        data={"input": oversize},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model.debug_input_too_long"
