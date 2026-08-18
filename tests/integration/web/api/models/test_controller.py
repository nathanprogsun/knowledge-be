"""Web-layer tests for the model router.

Exercises the router over HTTP via ``TestClient`` with
``get_model_service`` overridden to use a real ``ModelService`` backed
by an ``AsyncMock(spec=ModelRepository)`` configured with stateful
closures, so the full web -> service path runs without a database.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError
from src.core.infra.models.service.model_service import ModelService
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model
from src.web.deps.infra_models import get_model_service


@pytest.fixture
def repo() -> AsyncMock:
    """``AsyncMock(spec=ModelRepository)`` with stateful ``insert`` + lookups."""
    repo = AsyncMock(spec=ModelRepository)
    rows: dict[str, Model] = {}

    def _live() -> dict[str, Model]:
        return {mid: r for mid, r in rows.items() if r.deleted_at is None}

    async def _insert(row: Model) -> Model:
        rows[row.id] = row
        return row

    async def _update_row(row: Model) -> Model | None:
        existing = rows.get(row.id)
        if existing is None or existing.tenant_id != row.tenant_id:
            return None
        rows[row.id] = row
        return row

    async def _delete_by_tenant_and_id(*, tenant_id: int, id: str) -> int:
        existing = rows.get(id)
        if existing is None or existing.tenant_id != tenant_id:
            return 0
        del rows[id]
        return 1

    async def _clear_default_by_type(
        *,
        tenant_id: int,
        model_type: str,
        exclude_id: str | None = None,
    ) -> int:
        affected = 0
        for k, v in list(rows.items()):
            if v.deleted_at is not None:
                continue
            if v.tenant_id != tenant_id:
                continue
            if v.type != model_type:
                continue
            if not v.is_default:
                continue
            if exclude_id is not None and k == exclude_id:
                continue
            rows[k] = v.model_copy(update={"is_default": False})
            affected += 1
        return affected

    async def _find_by_tenant_and_id(
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model | None:
        existing = _live().get(id)
        if existing is None:
            return None
        if existing.tenant_id == tenant_id:
            return existing
        if include_builtin and existing.is_builtin:
            return existing
        return None

    async def _find_by_tenant_and_id_or_fail(
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model:
        row = await _find_by_tenant_and_id(
            tenant_id=tenant_id, id=id, include_builtin=include_builtin
        )
        if row is None:
            raise NotFoundError(code="model.not_found", message=f"Model {id} not found")
        return row

    async def _list_by_tenant(
        *,
        tenant_id: int,
        model_type: str | None = None,
        source: str | None = None,
        include_builtin: bool = True,
    ) -> list[Model]:
        results: list[Model] = []
        for row in _live().values():
            if row.tenant_id != tenant_id and not (include_builtin and row.is_builtin):
                continue
            if model_type is not None and row.type != model_type:
                continue
            if source is not None and row.source != source:
                continue
            results.append(row)
        return results

    repo.insert.side_effect = _insert
    repo.update_row.side_effect = _update_row
    repo.delete_by_tenant_and_id.side_effect = _delete_by_tenant_and_id
    repo.clear_default_by_type.side_effect = _clear_default_by_type
    repo.find_by_tenant_and_id.side_effect = _find_by_tenant_and_id
    repo.find_by_tenant_and_id_or_fail.side_effect = _find_by_tenant_and_id_or_fail
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    repo: AsyncMock,
) -> FastAPI:
    """Override ``get_model_service`` on the shared web app."""

    def _override_service() -> ModelService:
        return ModelService(models_repo=repo)

    web_app.dependency_overrides[get_model_service] = _override_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
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
    client: TestClient,
    repo: AsyncMock,
) -> None:
    resp = client.post("/api/v1/models", json=_create_body())

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"]["name"] == "gpt-4o"
    assert payload["data"]["status"] == "active"
    rows = repo._rows  # type: ignore[attr-defined]
    assert payload["data"]["id"] in rows


async def test_create_model_strips_credential_fields(
    client: TestClient,
) -> None:
    resp = client.post("/api/v1/models", json=_create_body())

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


async def test_create_model_rejects_blank_name(client: TestClient) -> None:
    resp = client.post("/api/v1/models", json=_create_body(name="   "))

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model.name_required"


# ── GET /models ─────────────────────────────────────────────────────


async def test_list_models_returns_tenant_rows(client: TestClient) -> None:
    client.post("/api/v1/models", json=_create_body(name="chat-1"))
    client.post("/api/v1/models", json=_create_body(name="chat-2"))

    resp = client.get("/api/v1/models")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert len(payload["data"]) == 2


async def test_list_models_filters_by_type(client: TestClient) -> None:
    client.post("/api/v1/models", json=_create_body(name="chat", type="KnowledgeQA"))
    client.post("/api/v1/models", json=_create_body(name="embed", type="Embedding"))

    resp = client.get("/api/v1/models?type=Embedding")

    assert resp.status_code == 200
    payload = resp.json()
    assert all(item["type"] == "Embedding" for item in payload["data"])


# ── GET /models/{id} ────────────────────────────────────────────────


async def test_get_model_returns_one_model(client: TestClient) -> None:
    created = client.post("/api/v1/models", json=_create_body())

    model_id = created.json()["data"]["id"]
    resp = client.get(f"/api/v1/models/{model_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["id"] == model_id
    assert payload["data"]["name"] == "gpt-4o"


async def test_get_model_returns_404_when_absent(client: TestClient) -> None:
    resp = client.get("/api/v1/models/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


# ── PUT /models/{id} ────────────────────────────────────────────────


async def test_update_model_patches_supplied_columns(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = client.put(
        f"/api/v1/models/{model_id}",
        json={"name": "gpt-4-turbo", "description": "renamed"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["name"] == "gpt-4-turbo"
    assert payload["data"]["description"] == "renamed"


async def test_update_model_returns_404_when_absent(client: TestClient) -> None:
    resp = client.put("/api/v1/models/does-not-exist", json={"name": "x"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


# ── DELETE /models/{id} ─────────────────────────────────────────────


async def test_delete_model_removes_row(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    created = client.post("/api/v1/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = client.delete(f"/api/v1/models/{model_id}")

    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["message"] == "Model deleted"
    rows = repo._rows  # type: ignore[attr-defined]
    assert model_id not in rows


# ── GET /models/providers ───────────────────────────────────────────


async def test_list_providers_returns_catalog(client: TestClient) -> None:
    resp = client.get("/api/v1/models/providers")

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


async def test_debug_model_returns_envelope(client: TestClient) -> None:
    created = client.post("/api/v1/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    resp = client.post(
        f"/api/v1/models/{model_id}/debug",
        data={"input": "hello"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["request"]["input"] == "hello"
    assert data["request"]["model_id"] == model_id


async def test_debug_model_returns_404_when_absent(client: TestClient) -> None:
    resp = client.post("/api/v1/models/does-not-exist/debug", data={"input": "hi"})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model.not_found"


async def test_debug_model_rejects_oversized_input(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/models", json=_create_body())
    model_id = created.json()["data"]["id"]

    oversize = "x" * (65 * 1024)
    resp = client.post(
        f"/api/v1/models/{model_id}/debug",
        data={"input": oversize},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "model.debug_input_too_long"
