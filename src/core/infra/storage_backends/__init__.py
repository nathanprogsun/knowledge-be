"""Storage-backend registry — a workspace's concrete storage instances."""

from __future__ import annotations

from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import (
    StorageBackendConfigInfo,
    StorageBackendInfo,
    StorageBackendListResult,
    StorageConnectivityResult,
    allowed_providers,
)

__all__ = [
    "StorageBackendConfigInfo",
    "StorageBackendInfo",
    "StorageBackendListResult",
    "StorageBackendService",
    "StorageConnectivityResult",
    "allowed_providers",
]
