"""Web-layer tests for ``GET/PUT /initialization/config/{kb_id}``.

The save-and-close path writes models and chunking through this pair.
Services are overridden so routing, aliases, role gates, and the
exception handler stay in the loop without touching the database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.infra import ModelParameters
from src.core.infra.models.service.model_service import ModelService
from src.core.infra.models.types import ModelInfo
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.web.deps.infra_models import get_model_service
from src.web.deps.infra_storage_backends import get_storage_backend_service
from src.web.deps.knowledge import get_knowledge_service
from src.web.deps.knowledge_bases import get_kb_service

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_KB_ID = "kb-config-1"
_LLM_ID = "llm-1"
_EMB_ID = "emb-1"
_PUT_PATH = f"/api/v1/initialization/config/{_KB_ID}"
_GET_PATH = _PUT_PATH


def _kb_info(*, summary_model_id: str = "", embedding_model_id: str = "") -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=_KB_ID,
        name="docs",
        tenant_id=1,
        summary_model_id=summary_model_id,
        embedding_model_id=embedding_model_id,
        chunking_config={"chunk_size": 512, "chunk_overlap": 64, "separators": ["\n\n"]},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _model(model_id: str, *, name: str, source: str = "remote") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        tenant_id=1,
        name=name,
        type="KnowledgeQA",
        source=source,
        parameters=ModelParameters(),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _put_body() -> dict[str, object]:
    return {
        "llmModelId": _LLM_ID,
        "embeddingModelId": _EMB_ID,
        "documentSplitting": {
            "chunkSize": 512,
            "chunkOverlap": 64,
            "separators": ["\n\n"],
        },
        "multimodal": {"enabled": False},
        "nodeExtract": {
            "enabled": False,
            "text": "",
            "tags": [],
            "nodes": [],
            "relations": [],
        },
    }


@pytest.fixture
def kb_service() -> AsyncMock:
    return AsyncMock(spec=KBService)


@pytest.fixture
def model_service() -> AsyncMock:
    return AsyncMock(spec=ModelService)


@pytest.fixture
def knowledge_service() -> AsyncMock:
    return AsyncMock(spec=KnowledgeService)


@pytest.fixture
def storage_service() -> AsyncMock:
    return AsyncMock(spec=StorageBackendService)


@pytest.fixture
def app(
    web_app: FastAPI,
    kb_service: AsyncMock,
    model_service: AsyncMock,
    knowledge_service: AsyncMock,
    storage_service: AsyncMock,
) -> FastAPI:
    """Override the four services the config handlers call."""
    web_app.dependency_overrides[get_kb_service] = lambda: kb_service
    web_app.dependency_overrides[get_model_service] = lambda: model_service
    web_app.dependency_overrides[get_knowledge_service] = lambda: knowledge_service
    web_app.dependency_overrides[get_storage_backend_service] = lambda: storage_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    return web_authed_client


def test_get_kb_config_returns_slots(
    client: TestClient,
    kb_service: AsyncMock,
    knowledge_service: AsyncMock,
    model_service: AsyncMock,
) -> None:
    kb_service.get_knowledge_base_by_id_and_tenant.return_value = _kb_info(
        summary_model_id=_LLM_ID,
        embedding_model_id=_EMB_ID,
    )
    knowledge_service.count_documents.return_value = 2

    async def _get_model(*, tenant_id: int, model_id: str) -> ModelInfo:
        del tenant_id
        if model_id == _LLM_ID:
            return _model(_LLM_ID, name="chat")
        if model_id == _EMB_ID:
            return _model(_EMB_ID, name="embed", source="remote")
        raise NotFoundError(code="model.not_found", message="missing")

    model_service.get_model.side_effect = _get_model

    resp = client.get(_GET_PATH)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["hasFiles"] is True
    assert data["llm"]["modelName"] == "chat"
    assert data["documentSplitting"]["chunkSize"] == 512


def test_get_kb_config_unknown_kb_is_404(
    client: TestClient,
    kb_service: AsyncMock,
) -> None:
    kb_service.get_knowledge_base_by_id_and_tenant.side_effect = NotFoundError(
        code="knowledge_base.not_found",
        message="knowledge base missing",
    )
    resp = client.get(_GET_PATH)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "knowledge_base.not_found"


def test_put_kb_config_persists_chunking(
    client: TestClient,
    kb_service: AsyncMock,
    knowledge_service: AsyncMock,
    model_service: AsyncMock,
) -> None:
    info = _kb_info(summary_model_id=_LLM_ID, embedding_model_id=_EMB_ID)
    kb_service.get_knowledge_base_by_id_and_tenant.return_value = info
    kb_service.update_model_config.return_value = info
    knowledge_service.count_documents.return_value = 0
    model_service.get_model.return_value = _model(_LLM_ID, name="chat")

    resp = client.put(_PUT_PATH, json=_put_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "配置更新成功"
    kb_service.update_model_config.assert_awaited_once()
    kwargs = kb_service.update_model_config.await_args.kwargs
    assert kwargs["summary_model_id"] == _LLM_ID
    assert kwargs["embedding_model_id"] == _EMB_ID
    assert kwargs["chunking_config"]["chunk_size"] == 512
    assert kwargs["chunking_config"]["chunk_overlap"] == 64
    assert kwargs["document_count"] == 0


def test_put_kb_config_unknown_llm_is_422(
    client: TestClient,
    kb_service: AsyncMock,
    model_service: AsyncMock,
) -> None:
    kb_service.get_knowledge_base_by_id_and_tenant.return_value = _kb_info()
    model_service.get_model.side_effect = NotFoundError(
        code="model.not_found",
        message="missing",
    )
    resp = client.put(_PUT_PATH, json=_put_body())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "knowledge_base.llm_model_not_found"
    kb_service.update_model_config.assert_not_awaited()


def test_put_kb_config_embedding_lock_is_422(
    client: TestClient,
    kb_service: AsyncMock,
    knowledge_service: AsyncMock,
    model_service: AsyncMock,
) -> None:
    kb_service.get_knowledge_base_by_id_and_tenant.return_value = _kb_info(
        embedding_model_id="emb-old",
    )
    knowledge_service.count_documents.return_value = 3
    model_service.get_model.return_value = _model(_LLM_ID, name="chat")
    kb_service.update_model_config.side_effect = ValidationError(
        code="knowledge_base.embedding_model_locked",
        message="Cannot change the embedding model after documents exist",
    )
    resp = client.put(_PUT_PATH, json=_put_body())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "knowledge_base.embedding_model_locked"
