"""Embedding-model service: resolve a stored model row to a live embedder.

Mirrors the upstream model-service inference path (``GetEmbeddingModel`` /
``GetEmbeddingModelForTenant``): a stored row is mapped through
``config_from_model`` and ``new_embedder`` into a ready embedder backed by
the shared batch pooler and Ollama service. The service is stateless and
built per request like ``ModelService``; the repository owns the
per-request session.

The upstream resolves managed-cloud tenant credentials here before calling
the factory; that resolution lands with the managed-cloud provider work,
so for now the config carries whatever ``app_id`` / ``app_secret`` the
model row stores.
"""

from __future__ import annotations

from src.ai.embedding import (
    Embedder,
    EmbedderPooler,
    config_from_model,
    new_batch_embedder,
    new_embedder,
)
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import ValidationError
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model


class EmbeddingService:
    """Constructs live embedders from stored model rows."""

    def __init__(
        self,
        *,
        models_repo: ModelRepository,
        pooler: EmbedderPooler | None = None,
        ollama_service: OllamaService | None = None,
    ) -> None:
        self._models_repo = models_repo
        self._pooler = pooler if pooler is not None else new_batch_embedder()
        self._ollama_service = ollama_service

    async def get_embedding_model(self, *, tenant_id: int, model_id: str) -> Embedder:
        """Resolve a model visible to ``tenant_id`` to a live embedder.

        Built-in rows from the system tenant are included, matching the
        upstream tenant-or-builtin visibility predicate.
        """
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        model = await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
        )
        return await self._build_embedder(model)

    async def get_embedding_model_for_tenant(
        self,
        *,
        tenant_id: int,
        model_id: str,
    ) -> Embedder:
        """Resolve a model owned strictly by ``tenant_id`` to an embedder.

        Cross-tenant sharing path (upstream ``GetEmbeddingModelForTenant``):
        the source tenant's embedding model is used to keep vectors
        compatible, so built-in rows are not consulted here.
        """
        if tenant_id <= 0:
            raise ValidationError(
                code="model.invalid_tenant_id",
                message="Tenant ID must be positive",
            )
        model = await self._models_repo.find_by_tenant_and_id_or_fail(
            tenant_id=tenant_id,
            id=model_id,
            include_builtin=False,
        )
        return await self._build_embedder(model)

    async def _build_embedder(self, model: Model) -> Embedder:
        config = config_from_model(model)
        return await new_embedder(config, self._pooler, self._ollama_service)


__all__ = ["EmbeddingService"]
