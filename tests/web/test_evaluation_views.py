"""Unit tests for the evaluation web endpoints.

The router is mounted on a minimal FastAPI app; the auth gate and the
evaluation service dep are overridden so tests exercise routing, request
validation, and response shapes without a database. The fake auth stamps
an admin principal so both the Admin-gated run endpoint and the
Viewer-gated result endpoint pass the RBAC dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError
from src.core.contracts.evaluation import (
    EvalTask,
    EvalTaskParams,
    EvaluationGetResponseData,
)
from src.core.evaluation.service.evaluation_service import EvaluationService
from src.web.api.evaluation.router import router
from src.web.deps.evaluation import get_evaluation_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

# Principal constants the fake auth stamps on every request.
_TENANT: int = 42

# Module-level service holder so the dependency override can resolve the
# per-test mock without threading mutable state through the FastAPI app.
_holder: dict[str, AsyncMock] = {}


def _snapshot(**overrides: Any) -> EvaluationGetResponseData:
    """Build an ``EvaluationGetResponseData`` view with sensible defaults."""
    defaults: dict[str, Any] = {
        "task": EvalTask(
            id="evaluation_42_1754000000000_abcd1234",
            tenant_id=_TENANT,
            dataset_id="default",
            start_time=datetime(2026, 8, 1, tzinfo=UTC),
            status=0,
            total=0,
            finished=0,
        ),
        "params": EvalTaskParams(
            knowledge_base_id="kb-1",
            chat_model_id="chat-1",
            rerank_model_id="rerank-1",
        ),
        "metric": None,
    }
    defaults.update(overrides)
    return EvaluationGetResponseData(**defaults)


def _happy_path_service() -> AsyncMock:
    """Build an ``EvaluationService`` mock preconfigured for the happy path."""
    svc = AsyncMock(spec=EvaluationService)
    svc.create.return_value = _snapshot()
    svc.get.return_value = _snapshot()
    return svc


async def _fake_auth(request: Request) -> None:
    """Stand-in for ``require_auth`` that stamps an admin principal."""
    request.state.tenant_id = str(_TENANT)
    request.state.tenant_role = "admin"
    request.state.user_info = {
        "id": "user-1",
        "username": "alice",
        "email": "alice@example.com",
        "is_active": "1",
        "can_access_all_tenants": "0",
        "is_system_admin": "0",
    }
    request.state.is_system_admin = False
    request.state.api_key_scope = None


def _get_evaluation_service() -> AsyncMock:
    """DI override factory: return the per-test evaluation service."""
    return _holder["service"]


@pytest.fixture
def client() -> TestClient:
    """A ``TestClient`` bound to a minimal app with the evaluation router."""
    _holder["service"] = _happy_path_service()
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(router)
    application.dependency_overrides[require_auth] = _fake_auth
    application.dependency_overrides[get_evaluation_service] = _get_evaluation_service
    return TestClient(application)


def evaluation_service(client: TestClient) -> AsyncMock:
    """Return the per-test evaluation service mock."""
    return _holder["service"]


# ── POST /evaluation ─────────────────────────────────────────────────


def test_run_evaluation_returns_envelope(client: TestClient) -> None:
    service = evaluation_service(client)

    response = client.post(
        "/evaluation",
        json={
            "dataset_id": "default",
            "knowledge_base_id": "kb-1",
            "chat_id": "chat-1",
            "rerank_id": "rerank-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["task"]["dataset_id"] == "default"
    assert data["task"]["tenant_id"] == _TENANT
    assert data["params"]["knowledge_base_id"] == "kb-1"
    assert data["params"]["chat_model_id"] == "chat-1"
    assert data["params"]["rerank_model_id"] == "rerank-1"
    assert data["metric"] is None

    query = service.create.await_args.args[0]
    assert query.dataset_id == "default"
    assert query.knowledge_base_id == "kb-1"
    assert query.chat_model_id == "chat-1"
    assert query.rerank_model_id == "rerank-1"


def test_run_evaluation_maps_empty_chat_and_rerank(client: TestClient) -> None:
    """Empty ``chat_id`` / ``rerank_id`` flow through for default resolution."""
    service = evaluation_service(client)

    response = client.post(
        "/evaluation",
        json={
            "dataset_id": "default",
            "knowledge_base_id": "kb-1",
            "chat_id": "",
            "rerank_id": "",
        },
    )

    assert response.status_code == 200
    query = service.create.await_args.args[0]
    assert query.chat_model_id == ""
    assert query.rerank_model_id == ""


def test_run_evaluation_rejects_missing_field(client: TestClient) -> None:
    response = client.post(
        "/evaluation",
        json={"dataset_id": "default"},
    )

    assert response.status_code == 422


def test_run_evaluation_missing_tenant_is_401(client: TestClient) -> None:
    async def _no_tenant(request: Request) -> None:
        request.state.tenant_id = "0"
        request.state.tenant_role = "admin"
        request.state.user_info = None
        request.state.is_system_admin = False
        request.state.api_key_scope = None

    client.app.dependency_overrides[require_auth] = _no_tenant

    response = client.post(
        "/evaluation",
        json={
            "dataset_id": "default",
            "knowledge_base_id": "kb-1",
            "chat_id": "chat-1",
            "rerank_id": "rerank-1",
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "evaluation.tenant_context_missing"


# ── GET /evaluation ──────────────────────────────────────────────────


def test_get_evaluation_returns_envelope(client: TestClient) -> None:
    service = evaluation_service(client)

    response = client.get(
        "/evaluation",
        params={"task_id": "evaluation_42_1_abcd"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["task"]["id"] == "evaluation_42_1754000000000_abcd1234"
    assert payload["data"]["params"]["chat_model_id"] == "chat-1"
    service.get.assert_awaited_once_with("evaluation_42_1_abcd")


def test_get_evaluation_requires_task_id(client: TestClient) -> None:
    response = client.get("/evaluation")

    assert response.status_code == 422


def test_get_evaluation_missing_tenant_is_401(client: TestClient) -> None:
    async def _no_tenant(request: Request) -> None:
        request.state.tenant_id = "0"
        request.state.tenant_role = "viewer"
        request.state.user_info = None
        request.state.is_system_admin = False
        request.state.api_key_scope = None

    client.app.dependency_overrides[require_auth] = _no_tenant

    response = client.get("/evaluation", params={"task_id": "task-1"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "evaluation.tenant_context_missing"


def test_get_evaluation_unknown_task_is_404(client: TestClient) -> None:
    service = evaluation_service(client)
    service.get.side_effect = NotFoundError(
        code="evaluation.task_not_found",
        message="evaluation task not found",
    )

    response = client.get("/evaluation", params={"task_id": "missing"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "evaluation.task_not_found"


__all__: list[Any] = []
