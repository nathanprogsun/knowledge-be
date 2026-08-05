"""Models-domain request-scoped service factory.

Repositories are built per request on the shared ``AsyncSession``;
``web`` never imports ``db``. Mirrors the pattern in
``src.core.tenants.factory``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.models.service.model_service import ModelService
from src.db.dao.model_repository import ModelRepository


def build_model_service(session: AsyncSession) -> ModelService:
    """Per-request ``ModelService`` with a fresh repository."""
    return ModelService(models_repo=ModelRepository(session))


__all__ = ["build_model_service"]
