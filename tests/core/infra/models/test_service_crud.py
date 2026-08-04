"""Unit tests for ``ModelService`` CRUD operations.

Exercises the service against an in-memory fake that mirrors the real
``ModelRepository`` contract (the same protocol is what the factory
hands the service at request time). The fake matches the repository's
method signatures so a drift between the two surfaces here rather
than at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.infra import (
    CreateModelRequest,
    EmbeddingParameters,
    ModelParameters,
    UpdateModelRequest,
)
from src.core.infra.models.service.model_service import ModelService
from src.core.infra.models.types import ModelInfo
from tests.core.infra.models.fake_model_repository import FakeModelRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> FakeModelRepository:
    return FakeModelRepository()


@pytest.fixture
def service(repo: FakeModelRepository) -> ModelService:
    return ModelService(models_repo=repo)  # type: ignore[arg-type]


def _make_parameters(
    *,
    base_url: str | None = "https://example.com/v1",
    provider: str = "openai",
    api_key: str | None = "sk-test",
    app_secret: str | None = None,
    embedding_parameters: EmbeddingParameters | None = None,
) -> ModelParameters:
    return ModelParameters(
        base_url=base_url,
        provider=provider,
        api_key=api_key,
        app_secret=app_secret,
        embedding_parameters=embedding_parameters,
    )


def _make_create_request(**overrides: Any) -> CreateModelRequest:
    base: dict[str, Any] = {
        "name": "gpt-4o",
        "type": "KnowledgeQA",
        "source": "openai",
        "description": "Default chat model",
        "parameters": _make_parameters(),
    }
    base.update(overrides)
    return CreateModelRequest.model_validate(base)


# ── create_model ─────────────────────────────────────────────────────


async def test_create_model_persists_a_new_row(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    body = _make_create_request()
    info = await service.create_model(tenant_id=1, body=body)

    assert isinstance(info, ModelInfo)
    assert info.tenant_id == 1
    assert info.name == "gpt-4o"
    assert info.type == "KnowledgeQA"
    assert info.source == "openai"
    assert info.status == "active"
    assert info.is_default is False
    assert info.is_builtin is False
    assert info.id in repo.rows
    stored = repo.rows[info.id]
    assert stored.tenant_id == 1
    params = stored.parameters
    assert isinstance(params, dict)
    assert params["provider"] == "openai"


async def test_create_model_generates_id_when_unspecified(
    service: ModelService,
) -> None:
    body = _make_create_request()
    info = await service.create_model(tenant_id=1, body=body)

    assert info.id != ""
    # UUID-shaped string
    assert len(info.id.split("-")) == 5


async def test_create_model_honours_supplied_id(
    service: ModelService,
) -> None:
    body = _make_create_request()
    info = await service.create_model(
        tenant_id=1,
        body=body,
        model_id="builtin-llm-001",
    )

    assert info.id == "builtin-llm-001"


async def test_create_model_strips_whitespace_from_name(
    service: ModelService,
) -> None:
    body = _make_create_request(name="  gpt-4o  ")
    info = await service.create_model(tenant_id=1, body=body)

    assert info.name == "gpt-4o"


async def test_create_model_rejects_blank_name(
    service: ModelService,
) -> None:
    body = _make_create_request(name="   ")

    with pytest.raises(ValidationError) as excinfo:
        await service.create_model(tenant_id=1, body=body)

    assert excinfo.value.code == "model.name_required"


async def test_create_model_rejects_non_positive_tenant_id(
    service: ModelService,
) -> None:
    body = _make_create_request()

    with pytest.raises(ValidationError) as excinfo:
        await service.create_model(tenant_id=0, body=body)

    assert excinfo.value.code == "model.invalid_tenant_id"


async def test_create_model_persists_parameters_as_json_object(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    body = _make_create_request(
        parameters=_make_parameters(
            provider="openai",
            base_url="https://api.openai.com/v1",
            embedding_parameters=EmbeddingParameters(
                dimension=1536, supports_dimension_override=True
            ),
        ),
    )
    info = await service.create_model(tenant_id=1, body=body)

    stored = repo.rows[info.id]
    assert stored.parameters["provider"] == "openai"
    embedding_params = stored.parameters["embedding_parameters"]
    assert isinstance(embedding_params, dict)
    assert embedding_params["dimension"] == 1536


# ── get_model ────────────────────────────────────────────────────────


async def test_get_model_returns_projection(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(tenant_id=1, body=_make_create_request(name="gpt-4o"))
    info = await service.get_model(tenant_id=1, model_id=created.id)

    assert info.id == created.id
    assert info.name == "gpt-4o"


async def test_get_model_raises_not_found_for_unknown_id(
    service: ModelService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get_model(tenant_id=1, model_id="does-not-exist")

    assert excinfo.value.code == "model.not_found"


async def test_get_model_rejects_non_positive_tenant_id(
    service: ModelService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.get_model(tenant_id=0, model_id="any")

    assert excinfo.value.code == "model.invalid_tenant_id"


# ── list_models ──────────────────────────────────────────────────────


async def test_list_models_returns_tenant_rows_with_builtins(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="chat-1"),
    )
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="chat-2"),
    )
    # Built-in row in the system tenant (visible to every tenant).
    builtin = await service.create_model(
        tenant_id=10000,
        body=_make_create_request(name="builtin-llm"),
        model_id="builtin-llm",
    )
    builtin_row = repo.rows[builtin.id]
    repo.rows[builtin.id] = builtin_row.model_copy(update={"is_builtin": True})

    infos = await service.list_models(tenant_id=1)

    ids = {i.id for i in infos}
    assert "builtin-llm" in ids
    assert len(infos) >= 2


async def test_list_models_filters_by_type(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="chat", type="KnowledgeQA"),
    )
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="embed", type="Embedding"),
    )

    infos = await service.list_models(tenant_id=1, model_type="Embedding")

    assert all(i.type == "Embedding" for i in infos)


async def test_list_models_filters_by_source(
    service: ModelService,
) -> None:
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="oai", source="openai"),
    )
    await service.create_model(
        tenant_id=1,
        body=_make_create_request(name="ollama", source="local"),
    )

    infos = await service.list_models(tenant_id=1, source="local")

    assert all(i.source == "local" for i in infos)


async def test_list_models_excludes_builtins_when_disabled(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    builtin = await service.create_model(
        tenant_id=10000,
        body=_make_create_request(name="builtin-llm"),
        model_id="builtin-llm",
    )
    builtin_row = repo.rows[builtin.id]
    repo.rows[builtin.id] = builtin_row.model_copy(update={"is_builtin": True})

    infos = await service.list_models(tenant_id=1, include_builtin=False)

    ids = {i.id for i in infos}
    assert "builtin-llm" not in ids


# ── update_model ─────────────────────────────────────────────────────


async def test_update_model_patches_supplied_columns(
    service: ModelService,
) -> None:
    created = await service.create_model(tenant_id=1, body=_make_create_request(name="gpt-4o"))
    updated = await service.update_model(
        tenant_id=1,
        model_id=created.id,
        body=UpdateModelRequest(name="gpt-4-turbo", description="new"),
    )

    assert updated.name == "gpt-4-turbo"
    assert updated.description == "new"


async def test_update_model_preserves_stored_credentials(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    """A credential in the stored parameters must survive a non-credential update."""
    created = await service.create_model(
        tenant_id=1,
        body=_make_create_request(
            parameters=_make_parameters(api_key="sk-secret"),
        ),
    )
    # Update with a body that carries no api_key at all (the typical
    # UI edit path). The stored value must be preserved.
    await service.update_model(
        tenant_id=1,
        model_id=created.id,
        body=UpdateModelRequest(name="renamed"),
    )

    stored = repo.rows[created.id]
    assert stored.parameters.get("api_key") == "sk-secret"


async def test_update_model_rejects_builtin_without_system_admin(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(
        tenant_id=10000, body=_make_create_request(), model_id="builtin-1"
    )
    repo.rows[created.id] = repo.rows[created.id].model_copy(update={"is_builtin": True})

    with pytest.raises(ValidationError) as excinfo:
        await service.update_model(
            tenant_id=1,
            model_id=created.id,
            body=UpdateModelRequest(name="renamed"),
            is_system_admin=False,
        )

    assert excinfo.value.code == "model.builtin_protected"


async def test_update_model_allows_system_admin_to_edit_builtin(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(
        tenant_id=10000, body=_make_create_request(), model_id="builtin-1"
    )
    repo.rows[created.id] = repo.rows[created.id].model_copy(update={"is_builtin": True})

    updated = await service.update_model(
        tenant_id=1,
        model_id=created.id,
        body=UpdateModelRequest(name="renamed"),
        is_system_admin=True,
    )

    assert updated.name == "renamed"


async def test_update_model_clears_managed_by_for_builtin_edits(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(
        tenant_id=10000, body=_make_create_request(), model_id="builtin-1"
    )
    repo.rows[created.id] = repo.rows[created.id].model_copy(
        update={"is_builtin": True, "managed_by": "yaml"}
    )

    await service.update_model(
        tenant_id=1,
        model_id=created.id,
        body=UpdateModelRequest(name="renamed"),
        is_system_admin=True,
    )

    stored = repo.rows[created.id]
    assert stored.managed_by == ""


async def test_update_model_rejects_blank_name(
    service: ModelService,
) -> None:
    created = await service.create_model(tenant_id=1, body=_make_create_request(name="gpt-4o"))

    with pytest.raises(ValidationError) as excinfo:
        await service.update_model(
            tenant_id=1,
            model_id=created.id,
            body=UpdateModelRequest(name="   "),
        )

    assert excinfo.value.code == "model.name_required"


async def test_update_model_raises_not_found_for_unknown_id(
    service: ModelService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.update_model(
            tenant_id=1,
            model_id="does-not-exist",
            body=UpdateModelRequest(name="x"),
        )

    assert excinfo.value.code == "model.not_found"


# ── delete_model ─────────────────────────────────────────────────────


async def test_delete_model_returns_true_when_row_deleted(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(tenant_id=1, body=_make_create_request(name="gpt-4o"))

    deleted = await service.delete_model(tenant_id=1, model_id=created.id)

    assert deleted is True
    assert created.id not in repo.rows


async def test_delete_model_rejects_builtin(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    created = await service.create_model(
        tenant_id=10000, body=_make_create_request(), model_id="builtin-1"
    )
    repo.rows[created.id] = repo.rows[created.id].model_copy(update={"is_builtin": True})

    with pytest.raises(ValidationError) as excinfo:
        await service.delete_model(tenant_id=1, model_id=created.id)

    assert excinfo.value.code == "model.builtin_protected"


async def test_delete_model_raises_not_found_for_unknown_id(
    service: ModelService,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.delete_model(tenant_id=1, model_id="does-not-exist")

    assert excinfo.value.code == "model.not_found"


async def test_delete_model_rejects_non_positive_tenant_id(
    service: ModelService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.delete_model(tenant_id=0, model_id="any")

    assert excinfo.value.code == "model.invalid_tenant_id"


# ── ModelInfo DTO ────────────────────────────────────────────────────


async def test_model_info_redacts_credential_fields(
    service: ModelService,
) -> None:
    body = _make_create_request(
        parameters=_make_parameters(api_key="sk-secret", app_secret="app-secret"),
    )
    info = await service.create_model(tenant_id=1, body=body)

    assert info.parameters.api_key is None
    assert info.parameters.app_secret is None


async def test_model_info_preserves_other_parameter_fields(
    service: ModelService,
) -> None:
    body = _make_create_request(
        parameters=_make_parameters(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-secret",
        ),
    )
    info = await service.create_model(tenant_id=1, body=body)

    assert info.parameters.provider == "openai"
    assert info.parameters.base_url == "https://api.openai.com/v1"


async def test_create_model_accepts_dict_parameters(
    service: ModelService,
    repo: FakeModelRepository,
) -> None:
    """Parameters can come pre-dumped as a dict (the Go wire shape)."""
    body = _make_create_request(
        parameters={
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
        },
    )
    info = await service.create_model(tenant_id=1, body=body)
    assert repo.rows[info.id].parameters["provider"] == "openai"


async def test_create_model_rejects_non_object_parameters(
    service: ModelService,
) -> None:
    """Non-dict, non-ModelParameters input is rejected.

    Built via :func:`_coerce_parameters` directly because Pydantic's
    frozen ``CreateModelRequest`` blocks mutating ``parameters``.
    """
    from src.core.infra.models.service.model_service import _coerce_parameters

    with pytest.raises(ValidationError) as excinfo:
        _coerce_parameters("not-an-object")  # type: ignore[arg-type]
    assert excinfo.value.code == "model.parameters_invalid"


async def test_update_model_blank_name_rejected(
    service: ModelService,
) -> None:
    created = await service.create_model(tenant_id=1, body=_make_create_request(name="gpt-4o"))
    with pytest.raises(ValidationError) as excinfo:
        await service.update_model(
            tenant_id=1,
            model_id=created.id,
            body=UpdateModelRequest(name="   "),
        )
    assert excinfo.value.code == "model.name_required"


async def test_update_model_rejects_non_positive_tenant_id(
    service: ModelService,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.update_model(
            tenant_id=0,
            model_id="any",
            body=UpdateModelRequest(name="x"),
        )
    assert excinfo.value.code == "model.invalid_tenant_id"
