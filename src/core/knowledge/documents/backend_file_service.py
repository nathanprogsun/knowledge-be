"""Resolve a knowledge-base storage backend into a ``FileService``.

Upload and the document-process worker share this resolver so a
``file_url`` job can call ``save_bytes`` without importing ``web``.
"""

from __future__ import annotations

from src.ai.storage.base import FileService
from src.ai.storage.factory import new_file_service_from_storage_config
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import StorageBackendConfigInfo
from src.core.knowledge.knowledge_bases.service.kb_service import KBService


class ResolvableBackendConfig(StorageBackendConfigInfo):
    """Storage-config view with the provider fallback the factory reads.

    The file-service factory falls back to ``config.default_provider``
    when no provider is given; the stored backend config carries no such
    field, so this adds the blank default the factory tolerates.
    """

    default_provider: str = ""


class BackendFileServiceResolver:
    """Resolve the storage file service for a knowledge base and tenant.

    Implements the ``StorageResolver`` seam: the knowledge base names its
    storage backend, the registry resolves that backend to a provider +
    config, and the storage factory builds the concrete file service.
    Returns ``None`` when no backend is configured.
    """

    def __init__(
        self,
        *,
        kb_service: KBService,
        storage_backend_service: StorageBackendService,
    ) -> None:
        self._kb_service = kb_service
        self._storage_backend_service = storage_backend_service

    async def resolve_file_service(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> FileService | None:
        """Return the file service for the knowledge base, or ``None``."""
        kb = await self._kb_service.get_knowledge_base_by_id(knowledge_base_id=knowledge_base_id)
        backend_id = (kb.storage_backend_id or "").strip()
        info = await self._storage_backend_service.resolve_backend(
            tenant_id=tenant_id,
            backend_id=backend_id,
        )
        if info is None:
            return None
        config = ResolvableBackendConfig(**info.config.model_dump())
        return new_file_service_from_storage_config(info.provider, config)[0]


__all__ = [
    "BackendFileServiceResolver",
    "ResolvableBackendConfig",
]
