"""Unit tests for ``ChatModelService``.

Exercises the service against an ``AsyncMock(spec=ModelRepository)`` so
no DB session is involved. ``new_chat`` is mocked; factory routing is
covered in ``tests/ai``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.ai.llm.types import ChatConfig, ChatOptions, ChatResponse, Message
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import NotFoundError, ValidationError
from src.core.infra.models.chat_service import (
    ChatModelService,
    contract_model_from_row,
)
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeChat:
    """Minimal chat client returned by the mocked factory."""

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        return ChatResponse(content="ok")

    def chat_stream(self, messages: list[Message], opts: ChatOptions | None = None) -> None:
        return None

    def get_model_name(self) -> str:
        return "mimo"

    def get_model_id(self) -> str:
        return "mid"


def _model_row(
    *,
    id: str = "mid",
    tenant_id: int = 1,
    model_type: str = "KnowledgeQA",
    source: str = "remote",
) -> Model:
    return Model(
        id=id,
        tenant_id=tenant_id,
        name="mimo-v2.5",
        display_name=None,
        type=model_type,
        source=source,
        description=None,
        parameters={
            "base_url": "https://chat.test/v1",
            "api_key": "sk-test",
            "provider": "openai",
        },
        is_default=False,
        is_builtin=False,
        managed_by="",
        status="active",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _make_repo(rows: dict[str, Model]) -> AsyncMock:
    """``AsyncMock(spec=ModelRepository)`` with closure-captured rows."""
    repo = AsyncMock(spec=ModelRepository)

    async def _find_by_tenant_and_id(
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model | None:
        row = rows.get(id)
        if row is None:
            return None
        if row.tenant_id == tenant_id or (include_builtin and row.is_builtin):
            return row
        return None

    async def _find_by_tenant_and_id_or_fail(
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model:
        row = await _find_by_tenant_and_id(
            tenant_id=tenant_id,
            id=id,
            include_builtin=include_builtin,
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
        found: list[Model] = []
        for row in rows.values():
            visible = row.tenant_id == tenant_id or (include_builtin and row.is_builtin)
            if not visible:
                continue
            if model_type and row.type != model_type:
                continue
            if source and row.source != source:
                continue
            found.append(row)
        return found

    repo.find_by_tenant_and_id.side_effect = _find_by_tenant_and_id
    repo.find_by_tenant_and_id_or_fail.side_effect = _find_by_tenant_and_id_or_fail
    repo.list_by_tenant.side_effect = _list_by_tenant
    return repo


def test_contract_model_from_row_keeps_api_key() -> None:
    mapped = contract_model_from_row(_model_row())
    assert mapped.parameters.api_key == "sk-test"
    assert mapped.type == "KnowledgeQA"


async def test_get_chat_model_maps_row_through_new_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ChatConfig] = []
    fake = _FakeChat()

    def _new_chat(config: ChatConfig, ollama_service: OllamaService | None = None) -> _FakeChat:
        captured.append(config)
        return fake

    monkeypatch.setattr("src.core.infra.models.chat_service.new_chat", _new_chat)
    service = ChatModelService(models_repo=_make_repo({"mid": _model_row()}))
    chat = await service.get_chat_model(tenant_id=1, model_id="mid")
    assert chat.get_model_id() == "mid"
    assert captured[0].api_key == "sk-test"
    assert captured[0].model_id == "mid"


async def test_get_chat_model_rejects_non_knowledge_qa() -> None:
    service = ChatModelService(models_repo=_make_repo({"mid": _model_row(model_type="Embedding")}))
    with pytest.raises(ValidationError, match="not KnowledgeQA"):
        await service.get_chat_model(tenant_id=1, model_id="mid")


async def test_get_chat_model_invalid_tenant() -> None:
    service = ChatModelService(models_repo=_make_repo({}))
    with pytest.raises(ValidationError, match="Tenant ID must be positive"):
        await service.get_chat_model(tenant_id=0, model_id="mid")


async def test_first_knowledge_qa_id_returns_first_visible() -> None:
    rows = {
        "emb": _model_row(id="emb", model_type="Embedding"),
        "qa": _model_row(id="qa"),
    }
    service = ChatModelService(models_repo=_make_repo(rows))
    assert await service.first_knowledge_qa_id(tenant_id=1) == "qa"
