"""Storage-backend request-scoped service factory.

See ``src.core.system.factory`` for the pattern: the repository is built
per request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.db.dao.storage_backend_repository import StorageBackendRepository


def build_storage_backend_service(session: AsyncSession) -> StorageBackendService:
    """Per-request ``StorageBackendService`` with a fresh repository.

    The repository also owns the ``tenants.default_storage_backend_id``
    pointer, so setting a default and reading the backend row share one
    transactional unit of work.
    """
    return StorageBackendService(backend_repo=StorageBackendRepository(session))


__all__ = ["build_storage_backend_service"]
