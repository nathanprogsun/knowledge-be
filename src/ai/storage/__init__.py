"""Storage-provider adapters — full file services.

Each module wraps one provider's HTTP/filesystem access behind the
``FileService`` interface (connectivity probe + save/read/delete/copy).
``core`` selects an adapter by provider name; the adapters themselves
know nothing about the database or the web layer.
"""

from __future__ import annotations

from src.ai.storage.base import FileService, StorageAdapter
from src.ai.storage.cos_backend import CosStorageAdapter
from src.ai.storage.dummy_backend import DummyFileService
from src.ai.storage.ks3_backend import KS3FileService
from src.ai.storage.local_backend import LocalStorageAdapter
from src.ai.storage.minio_backend import MinioStorageAdapter
from src.ai.storage.obs_backend import ObsStorageAdapter
from src.ai.storage.oss_backend import OssFileService
from src.ai.storage.s3_backend import S3StorageAdapter
from src.ai.storage.tos_backend import TosFileService

__all__ = [
    "CosStorageAdapter",
    "DummyFileService",
    "FileService",
    "KS3FileService",
    "LocalStorageAdapter",
    "MinioStorageAdapter",
    "ObsStorageAdapter",
    "OssFileService",
    "S3StorageAdapter",
    "StorageAdapter",
    "TosFileService",
]
