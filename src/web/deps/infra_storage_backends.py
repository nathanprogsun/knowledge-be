"""Storage-backend FastAPI dependency factory.

One-line forwarder to ``src.core.infra.storage_backends.factory``: the
repository is assembled in ``core`` on the request-scoped ``AsyncSession``
so the request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.storage_backends.factory import build_storage_backend_service
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.web.deps.session import SessionDep


def get_storage_backend_service(session: SessionDep) -> StorageBackendService:
    """Build a per-request ``StorageBackendService`` on the shared session."""
    return build_storage_backend_service(session)


StorageBackendServiceDep = Annotated[StorageBackendService, Depends(get_storage_backend_service)]


__all__ = ["StorageBackendServiceDep", "get_storage_backend_service"]
