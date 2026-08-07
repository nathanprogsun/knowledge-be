"""Models-domain request-scoped service factories.

Repositories are built per request on the shared ``AsyncSession``;
``web`` never imports ``db``. Mirrors the pattern in
``src.core.tenants.factory``.
"""

from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.models.rerank_service import RerankService
from src.core.infra.models.service.model_service import ModelService
from src.core.infra.models.service.provider_service import WeKnoraCloudService
from src.db.dao.model_repository import ModelRepository
from src.db.dao.tenants_repository import TenantRepository


def build_model_service(session: AsyncSession) -> ModelService:
    """Per-request ``ModelService`` with a fresh repository."""
    return ModelService(models_repo=ModelRepository(session))


def build_rerank_service(
    session: AsyncSession,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> RerankService:
    """Per-request ``RerankService`` with a fresh repository.

    ``http_client`` may be an APP-scope pooled client from the lifespan
    registry; when omitted the reranker opens its own client.
    """
    return RerankService(
        models_repo=ModelRepository(session),
        http_client=http_client,
    )


def build_weknora_cloud_service(
    session: AsyncSession,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> WeKnoraCloudService:
    """Per-request ``WeKnoraCloudService`` with a fresh repository.

    ``http_client`` may be an APP-scope pooled client from the lifespan
    registry; when omitted the service opens a short-lived client for
    the single credential-verification call.
    """
    return WeKnoraCloudService(
        tenants_repo=TenantRepository(session),
        http_client=http_client,
    )


__all__ = ["build_model_service", "build_rerank_service", "build_weknora_cloud_service"]
