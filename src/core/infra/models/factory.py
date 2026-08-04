"""Infra-models request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.models.service.weknora_cloud_service import WeKnoraCloudService
from src.db.dao.tenants_repository import TenantRepository


def build_weknora_cloud_service(
    session: AsyncSession,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> WeKnoraCloudService:
    """Per-request ``WeKnoraCloudService`` with a fresh repository.

    ``http_client`` may be an APP-scope pooled client from the lifespan
    registry; when omitted the service opens a short-lived client for
    the single credential-verification call (Go builds a per-call
    ``http.Client`` too).
    """
    return WeKnoraCloudService(
        tenants_repo=TenantRepository(session),
        http_client=http_client,
    )


__all__ = ["build_weknora_cloud_service"]
