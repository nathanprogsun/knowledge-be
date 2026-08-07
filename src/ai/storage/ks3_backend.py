"""Kingsoft Cloud KS3 storage adapter (full file interface).

KS3 is an S3-compatible object store addressed virtual-host style; the
adapter wires the shared ``S3ObjectStore`` to the KS3 path conventions:

- objects are addressed as ``ks3://{bucket}/{objectKey}``;
- uploads land under ``{prefix}{tenant}/{knowledgeId}/{uuid}{ext}``;
- raw bytes land under ``{prefix}{tenant}/exports/{uuid}{ext}``.

Reads and deletes always target this service's bucket and only the
object key is taken from the path, mirroring the upstream driver.
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

PROVIDER_KS3: Final = "ks3"
KS3_SCHEME: Final = "ks3://"


class KS3FileService:
    """Full file service for a Kingsoft Cloud KS3 backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        path_prefix: str = "",
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=True)
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket_name = bucket_name
        self._path_prefix = path_prefix.strip().strip("/")
        self._store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the configured bucket."""
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key,
            secret_access_key=self._secret_key,
            force_path_style=False,
            provider_label="KS3",
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(
        self, *, file: FileUpload, tenant_id: int, knowledge_id: str
    ) -> str:
        """Upload ``file`` to ``{prefix}{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_key = join_object_key(
            [self._path_prefix, str(tenant_id), knowledge_id, f"{uuid.uuid4()}{ext}"]
        )
        data = await file.read()
        content_type = file.content_type or content_type_for_ext(ext)
        await self._store.put_object(object_key, data, content_type)
        return f"{KS3_SCHEME}{self._bucket_name}/{object_key}"

    async def save_bytes(
        self, *, data: bytes, tenant_id: int, file_name: str, temp: bool
    ) -> str:
        """Upload raw bytes to ``{prefix}{tenant}/exports/``.

        ``temp`` is ignored — KS3 has no separate auto-expiring store.
        """
        ext = os.path.splitext(file_name)[1]
        object_key = join_object_key(
            [self._path_prefix, str(tenant_id), "exports", f"{uuid.uuid4()}{ext}"]
        )
        await self._store.put_object(object_key, data, content_type_for_ext(ext))
        return f"{KS3_SCHEME}{self._bucket_name}/{object_key}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object at ``ks3://{bucket}/{key}``."""
        _, object_key = self._parse_path(file_path)
        data = await self._store.get_object(object_key)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h SigV4-presigned GET URL for the object."""
        _, object_key = self._parse_path(file_path)
        return self._store.presigned_get_url(object_key)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object at ``ks3://{bucket}/{key}``."""
        _, object_key = self._parse_path(file_path)
        await self._store.delete_object(object_key)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be a ``ks3://`` path; anything else is a
        cross-backend copy and is refused.
        """
        try:
            src_bucket, src_key = self._parse_path(src_path)
        except StorageBackendError:
            raise CrossBackendCopyError(
                message=f"ks3 copy rejected source {src_path!r}"
            ) from None
        ext = os.path.splitext(src_path)[1]
        dest_key = join_object_key(
            [self._path_prefix, str(tenant_id), knowledge_id, f"{uuid.uuid4()}{ext}"]
        )
        await self._store.copy_object(src_bucket, src_key, dest_key)
        return f"{KS3_SCHEME}{self._bucket_name}/{dest_key}"

    # ── Internals ───────────────────────────────────────────────────

    def _parse_path(self, file_path: str) -> tuple[str, str]:
        bucket, object_key = parse_provider_path(file_path, KS3_SCHEME)
        safe_object_key(object_key)
        return bucket, object_key

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
                provider_label="KS3",
            )
        return self._store_cache


__all__ = ["KS3_SCHEME", "PROVIDER_KS3", "KS3FileService"]
