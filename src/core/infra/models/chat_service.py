"""Chat-model service: resolve a stored model row to a live chat client.

Mirrors ``EmbeddingService``: a tenant-visible ``models`` row is mapped
through ``config_from_model`` and ``new_chat``. The service stays
request-scoped. The repository owns the session.

``ModelService.get_model`` redacts ``api_key``. This path keeps the
stored credentials so the client can authenticate.
"""

from __future__ import annotations

from src.ai.llm.base import config_from_model, new_chat
from src.ai.llm.types import Chat
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.contracts.infra import Model, ModelParameters
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model as ModelRow

MODEL_TYPE_KNOWLEDGE_QA: str = "KnowledgeQA"


def _require_positive_tenant(tenant_id: int) -> None:
    """Reject a missing or non-positive tenant id."""
    if tenant_id <= 0:
        raise ValidationError(
            code="model.invalid_tenant_id",
            message="Tenant ID must be positive",
        )


def _as_str_map(value: JsonValue) -> dict[str, str] | None:
    """Keep only string-to-string entries from a JSON object."""
    if not isinstance(value, dict):
        return None
    mapped: dict[str, str] = {
        key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)
    }
    return mapped or None


def _parameters_from_row(raw: JsonObject) -> ModelParameters:
    """Parse the stored parameters blob without redacting secrets."""
    payload: dict[str, JsonValue | dict[str, str]] = dict(raw)
    extra = _as_str_map(raw.get("extra_config"))
    if extra is not None:
        payload["extra_config"] = extra
    headers = _as_str_map(raw.get("custom_headers"))
    if headers is not None:
        payload["custom_headers"] = headers
    return ModelParameters.model_validate(payload)


def contract_model_from_row(row: ModelRow) -> Model:
    """Project a storage row onto the chat-factory ``Model`` shape."""
    raw = row.parameters if isinstance(row.parameters, dict) else {}
    return Model(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        display_name=row.display_name,
        type=row.type,
        source=row.source,
        description=row.description,
        parameters=_parameters_from_row(raw),
        is_default=row.is_default,
        is_builtin=row.is_builtin,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class ChatModelService:
    """Constructs live chat clients from stored KnowledgeQA rows."""

    def __init__(
        self,
        *,
        models_repo: ModelRepository,
        ollama_service: OllamaService | None = None,
    ) -> None:
        self._models_repo = models_repo
        self._ollama_service = ollama_service

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> Chat:
        """Resolve a tenant-visible KnowledgeQA row to a live chat client."""
        row = await self._load_row(tenant_id=tenant_id, model_id=model_id)
        if row.type != MODEL_TYPE_KNOWLEDGE_QA:
            raise ValidationError(
                code="model.not_chat",
                message=f"model {model_id} is type {row.type}, not {MODEL_TYPE_KNOWLEDGE_QA}",
            )
        config = config_from_model(contract_model_from_row(row))
        if config is None:
            raise ValidationError(
                code="model.chat_config_missing",
                message=f"model {model_id} could not be mapped to a chat client",
            )
        return new_chat(config, self._ollama_service)

    async def get_model_type(self, *, tenant_id: int, model_id: str) -> str | None:
        """Return the stored type of a visible model, or ``None`` if absent."""
        _require_positive_tenant(tenant_id)
        if not model_id:
            return None
        row = await self._models_repo.find_by_tenant_and_id(
            tenant_id=tenant_id,
            id=model_id,
        )
        return None if row is None else row.type

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None:
        """Return the first tenant-visible KnowledgeQA model id, if any."""
        _require_positive_tenant(tenant_id)
        rows = await self._models_repo.list_by_tenant(
            tenant_id=tenant_id,
            model_type=MODEL_TYPE_KNOWLEDGE_QA,
        )
        return rows[0].id if rows else None

    async def _load_row(self, *, tenant_id: int, model_id: str) -> ModelRow:
        """Load a visible model row or raise."""
        _require_positive_tenant(tenant_id)
        return await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
        )


__all__ = [
    "MODEL_TYPE_KNOWLEDGE_QA",
    "ChatModelService",
    "contract_model_from_row",
]
