"""Models-domain FastAPI dependency factories.

One-line forwarders to ``src.core.infra.models.factory``: repositories
are assembled in ``core`` on the request-scoped ``AsyncSession`` so
the request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.models.factory import build_model_service
from src.core.infra.models.service.model_service import ModelService
from src.web.deps.session import SessionDep


def get_model_service(session: SessionDep) -> ModelService:
    """Build a per-request ``ModelService`` on the shared session."""
    return build_model_service(session)


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]

__all__ = ["ModelServiceDep", "get_model_service"]
