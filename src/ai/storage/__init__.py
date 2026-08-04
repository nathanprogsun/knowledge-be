"""Storage-provider adapters — connectivity probes only.

Each module wraps one provider's HTTP/filesystem access behind the
``StorageAdapter`` protocol. ``core`` selects an adapter by provider name;
the adapters themselves know nothing about the database or the web layer.
"""

from __future__ import annotations

from src.ai.storage.base import StorageAdapter
from src.ai.storage.cos_backend import CosStorageAdapter
from src.ai.storage.local_backend import LocalStorageAdapter
from src.ai.storage.minio_backend import MinioStorageAdapter
from src.ai.storage.obs_backend import ObsStorageAdapter
from src.ai.storage.s3_backend import S3StorageAdapter

__all__ = [
    "CosStorageAdapter",
    "LocalStorageAdapter",
    "MinioStorageAdapter",
    "ObsStorageAdapter",
    "S3StorageAdapter",
    "StorageAdapter",
]
