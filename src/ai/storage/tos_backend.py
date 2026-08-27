"""Volcengine TOS storage adapter (full file interface).

TOS is an S3-compatible object store; the adapter wires the shared
``S3ObjectStore`` to the TOS path conventions:

- objects are addressed as ``tos://{bucket}/{objectKey}``;
- uploads land under ``{prefix}{tenant}/{knowledgeId}/{uuid}{ext}``;
- raw bytes land under ``{prefix}{tenant}/exports/{uuid}{ext}`` or, when
  ``temp`` is requested and a temp bucket is configured, under
  ``exports/{tenant}/{uuid}{ext}`` in that bucket (lifecycle rules
  auto-expire it).
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
    join_object_key,
    normalize_endpoint,
    parse_provider_path,
    safe_object_key,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError

PROVIDER_TOS: Final = "tos"
TOS_SCHEME: Final = "tos://"


class TosFileService:
    """Full file service for a Volcengine TOS backend."""

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
        self._path_prefix = path_prefix.strip().strip("/")
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
            provider_label="TOS",
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Upload ``file`` to ``{prefix}{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_name = join_object_key(
            [self._path_prefix, str(tenant_id), knowledge_id, f"{uuid.uuid4()}{ext}"]
        )
        data = await file.read()
        content_type = file.content_type or content_type_for_ext(ext)
        await self._store.put_object(object_name, data, content_type)
        return f"{TOS_SCHEME}{self._bucket_name}/{object_name}"

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes.

        ``temp`` writes to the temp bucket (when configured) under
        ``exports/{tenant}/``; otherwise the main bucket stores
        ``{prefix}{tenant}/exports/``.
        """
        ext = os.path.splitext(file_name)[1]
        if temp and self._temp_bucket_name:
            target_bucket = self._temp_bucket_name
            object_name = join_object_key(["exports", str(tenant_id), f"{uuid.uuid4()}{ext}"])
            await self._temp_store.put_object(object_name, data, content_type_for_ext(ext))
        else:
            target_bucket = self._bucket_name
            object_name = join_object_key(
                [self._path_prefix, str(tenant_id), "exports", f"{uuid.uuid4()}{ext}"]
            )
            await self._store.put_object(object_name, data, content_type_for_ext(ext))
        return f"{TOS_SCHEME}{target_bucket}/{object_name}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object at ``tos://{bucket}/{key}``."""
        _, object_name = self._parse_path(file_path)
        data = await self._store.get_object(object_name)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h SigV4-presigned GET URL for the object."""
        _, object_name = self._parse_path(file_path)
        return self._store.presigned_get_url(object_name)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object at ``tos://{bucket}/{key}``."""
        _, object_name = self._parse_path(file_path)
        await self._store.delete_object(object_name)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be a ``tos://`` path; anything else is a
        cross-backend copy and is refused.
        """
        try:
            src_bucket, src_key = self._parse_path(src_path)
        except StorageBackendError:
            raise CrossBackendCopyError(message=f"tos copy rejected source {src_path!r}") from None
        ext = os.path.splitext(src_path)[1]
        dest_key = join_object_key(
            [self._path_prefix, str(tenant_id), knowledge_id, f"{uuid.uuid4()}{ext}"]
        )
        await self._store.copy_object(src_bucket, src_key, dest_key)
        return f"{TOS_SCHEME}{self._bucket_name}/{dest_key}"

    # ── Internals ───────────────────────────────────────────────────

    def _parse_path(self, file_path: str) -> tuple[str, str]:
        bucket, object_name = parse_provider_path(file_path, TOS_SCHEME)
        safe_object_key(object_name)
        return bucket, object_name

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
                provider_label="TOS",
            )
        return self._store_cache

    @property
    def _temp_store(self) -> S3ObjectStore:
        if self._temp_store_cache is None:
            self._temp_store_cache = S3ObjectStore(
                endpoint_url=self._endpoint_url,
                bucket_name=self._temp_bucket_name,
                region=self._temp_region,
                access_key_id=self._access_key,
                secret_access_key=self._secret_key,
                force_path_style=False,
                provider_label="TOS",
            )
        return self._temp_store_cache


__all__ = ["PROVIDER_TOS", "TOS_SCHEME", "TosFileService"]
