"""Unit tests for ``EmbeddingService``.

Exercises the service against an ``AsyncMock(spec=ModelRepository)`` with
closure-captured state (the same fake pattern as ``test_model_service_crud``)
so no DB session is involved. Factory routing itself is covered in
``tests/ai/test_embedding.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.ai.embedding import (
    ConcurrencyEmbedder,
    Config,
    OpenAIEmbedder,
    config_from_model,
    new_batch_embedder,
)
from src.common.exception import NotFoundError, ValidationError
from src.core.infra.models.embedding_service import EmbeddingService
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitelist the fake base URL so the SSRF guard skips DNS resolution."""
    monkeypatch.setenv("SSRF_WHITELIST", "embedding.test")


def _model_row(*, id: str = "mid", tenant_id: int = 1, source: str = "remote") -> Model:
    return Model(
        id=id,
        tenant_id=tenant_id,
        name="text-embedding-3-small",
        display_name=None,
        type="Embedding",
        source=source,
        description=None,
        parameters={
            "base_url": "https://embedding.test/v1",
            "api_key": "sk-test",
            "provider": "openai",
            "embedding_parameters": {
                "dimension": 768,
                "supports_dimension_override": True,
                "truncate_prompt_tokens": 256,
            },
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
            raise NotFoundError(
                code="model.not_found",
                message=f"Model {id} not found",
            )
        return row

    repo.find_by_tenant_and_id.side_effect = _find_by_tenant_and_id
    repo.find_by_tenant_and_id_or_fail.side_effect = _find_by_tenant_and_id_or_fail
    return repo


def test_config_from_model_accepts_db_row() -> None:
    config = config_from_model(_model_row())
    assert isinstance(config, Config)
    assert config.source == "remote"
    assert config.provider == "openai"
    assert config.dimensions == 768
    assert config.supports_dimension_override is True
    assert config.truncate_prompt_tokens == 256
    assert config.api_key == "sk-test"


async def test_get_embedding_model_builds_openai_embedder() -> None:
    rows = {"mid": _model_row()}
    service = EmbeddingService(
        models_repo=_make_repo(rows),
        pooler=new_batch_embedder(),
        ollama_service=None,
    )
    embedder = await service.get_embedding_model(tenant_id=1, model_id="mid")
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OpenAIEmbedder)
    assert embedder.get_model_id() == "mid"


async def test_get_embedding_model_for_tenant_scopes_to_tenant() -> None:
    rows = {"mid": _model_row(tenant_id=1)}
    service = EmbeddingService(models_repo=_make_repo(rows))
    with pytest.raises(NotFoundError, match="not found"):
        await service.get_embedding_model_for_tenant(tenant_id=2, model_id="mid")


async def test_get_embedding_model_invalid_tenant_id() -> None:
    service = EmbeddingService(models_repo=_make_repo({}))
    with pytest.raises(ValidationError, match="Tenant ID must be positive"):
        await service.get_embedding_model(tenant_id=0, model_id="mid")


async def test_get_embedding_model_not_found() -> None:
    service = EmbeddingService(models_repo=_make_repo({}))
    with pytest.raises(NotFoundError, match="not found"):
        await service.get_embedding_model(tenant_id=1, model_id="missing")
