"""Aliyun OSS storage adapter (full file interface).

OSS is an S3-compatible object store; the adapter wires the shared
``S3ObjectStore`` to the OSS path conventions:

- objects are addressed as ``oss://{bucket}/{objectKey}``;
- uploads land under ``{prefix}{tenant}/{knowledgeId}/{uuid}{ext}``;
- raw bytes land under ``{prefix}{tenant}/exports/{uuid}{ext}`` or, when
  ``temp`` is requested and a temp bucket is configured, under
  ``exports/{tenant}/{uuid}{ext}`` in that bucket.

Read/download operations pick the store by the bucket segment embedded
in the path, so temp-bucket objects resolve to the temp client.
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from typing import BinaryIO, Final

from src.ai.storage.base import (
    FileUpload,
    S3ObjectStore,
    content_type_for_ext,
    head_bucket,
    normalize_endpoint,
    parse_provider_path,
    safe_object_key,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError

PROVIDER_OSS: Final = "oss"
OSS_SCHEME: Final = "oss://"


class OssFileService:
    """Full file service for an Aliyun OSS backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        path_prefix: str = "",
        temp_bucket_name: str = "",
        temp_region: str = "",
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=True)
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket_name = bucket_name
        self._path_prefix = _trailing_slash(path_prefix)
        self._temp_bucket_name = temp_bucket_name.strip()
        self._temp_region = temp_region.strip() or region
        self._store_cache: S3ObjectStore | None = None
        self._temp_store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the configured bucket."""
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key,
            secret_access_key=self._secret_key,
            force_path_style=False,
            provider_label="OSS",
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Upload ``file`` to ``{prefix}{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_name = f"{self._path_prefix}{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        data = await file.read()
        content_type = file.content_type or content_type_for_ext(ext)
        await self._store.put_object(object_name, data, content_type)
        return f"{OSS_SCHEME}{self._bucket_name}/{object_name}"

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes.

        ``temp`` writes to the temp bucket (when configured) under
        ``exports/{tenant}/``; otherwise the main bucket stores
        ``{prefix}{tenant}/exports/``.
        """
        ext = os.path.splitext(file_name)[1]
        if temp and self._temp_store is not None:
            target_bucket = self._temp_bucket_name
            object_name = f"exports/{tenant_id}/{uuid.uuid4()}{ext}"
            await self._temp_store.put_object(object_name, data, content_type_for_ext(ext))
        else:
            target_bucket = self._bucket_name
            object_name = f"{self._path_prefix}{tenant_id}/exports/{uuid.uuid4()}{ext}"
            await self._store.put_object(object_name, data, content_type_for_ext(ext))
        return f"{OSS_SCHEME}{target_bucket}/{object_name}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object at ``oss://{bucket}/{key}``."""
        bucket, object_name = self._parse_path(file_path)
        data = await self._store_for(bucket).get_object(object_name)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h SigV4-presigned GET URL for the object."""
        bucket, object_name = self._parse_path(file_path)
        return self._store_for(bucket).presigned_get_url(object_name)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object at ``oss://{bucket}/{key}``."""
        bucket, object_name = self._parse_path(file_path)
        await self._store_for(bucket).delete_object(object_name)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be an ``oss://`` path; anything else is a
        cross-backend copy and is refused.
        """
        try:
            src_bucket, src_key = self._parse_path(src_path)
        except StorageBackendError:
            raise CrossBackendCopyError(message=f"oss copy rejected source {src_path!r}") from None
        ext = os.path.splitext(src_path)[1]
        dest_key = f"{self._path_prefix}{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        await self._store.copy_object(src_bucket, src_key, dest_key)
        return f"{OSS_SCHEME}{self._bucket_name}/{dest_key}"

    # ── Internals ───────────────────────────────────────────────────

    def _parse_path(self, file_path: str) -> tuple[str, str]:
        bucket, object_name = parse_provider_path(file_path, OSS_SCHEME)
        safe_object_key(object_name)
        return bucket, object_name

    def _store_for(self, bucket: str) -> S3ObjectStore:
        if self._temp_store is not None and bucket == self._temp_bucket_name:
            return self._temp_store
        return self._store

    @property
    def _store(self) -> S3ObjectStore:
        if self._store_cache is None:
            self._store_cache = S3ObjectStore(
                endpoint_url=self._endpoint_url,
                bucket_name=self._bucket_name,
                region=self._region,
                access_key_id=self._access_key,
                secret_access_key=self._secret_key,
                force_path_style=False,
                provider_label="OSS",
            )
        return self._store_cache

    @property
    def _temp_store(self) -> S3ObjectStore | None:
        if not self._temp_bucket_name:
            return None
        if self._temp_store_cache is None:
            self._temp_store_cache = S3ObjectStore(
                endpoint_url=self._endpoint_url,
                bucket_name=self._temp_bucket_name,
                region=self._temp_region,
                access_key_id=self._access_key,
                secret_access_key=self._secret_key,
                force_path_style=False,
                provider_label="OSS",
            )
        return self._temp_store_cache


def _trailing_slash(path_prefix: str) -> str:
    """Normalise a prefix so ``{prefix}{tenant}`` reads naturally."""
    prefix = path_prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


__all__ = ["OSS_SCHEME", "PROVIDER_OSS", "OssFileService"]
