"""S3 storage adapter (full file interface).

The connectivity probe is a signed ``HeadBucket`` honouring the row's
``force_path_style`` flag. File operations go through the shared
``S3ObjectStore``: PUT/GET/DELETE are SigV4-signed against the endpoint,
``CopyFile`` uses a server-side ``CopyObject`` and ``GetFileURL`` returns
a 24h-presigned URL.

Stored objects are addressed as ``s3://{bucket}/{pathPrefix}{tenant}/{knowledgeId}/{uuid}{ext}``
for uploads and ``s3://{bucket}/{pathPrefix}{tenant}/exports/{uuid}{ext}``
for raw bytes.
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
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError

PROVIDER_S3: Final = "s3"
S3_SCHEME: Final = "s3://"

# Region AWS SigV4 falls back to when the row leaves it blank.
DEFAULT_S3_REGION: Final = "us-east-1"


class S3StorageAdapter:
    """Full file service for AWS S3 and endpoint-compatible object stores."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        use_ssl: bool = True,
        force_path_style: bool = False,
        provider_label: str = "S3",
        path_prefix: str = "",
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=use_ssl)
        self._region = region or DEFAULT_S3_REGION
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = bucket_name
        self._force_path_style = force_path_style
        self._provider_label = provider_label
        self._path_prefix = _trailing_slash(path_prefix)
        self._store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the bucket — 2xx means reachable + authorized."""
        if not self._endpoint_url:
            raise StorageBackendError(
                code="storage_backend.endpoint_required",
                message=f"{self._provider_label} connectivity check requires an endpoint",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=self._force_path_style,
            provider_label=self._provider_label,
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(
        self, *, file: FileUpload, tenant_id: int, knowledge_id: str
    ) -> str:
        """Upload ``file`` to ``{prefix}{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_key = self._object_key(tenant_id, knowledge_id, ext)
        data = await file.read()
        content_type = file.content_type or content_type_for_ext(ext)
        await self._store.put_object(object_key, data, content_type)
        return f"{S3_SCHEME}{self._bucket_name}/{object_key}"

    async def save_bytes(
        self, *, data: bytes, tenant_id: int, file_name: str, temp: bool
    ) -> str:
        """Upload raw bytes to ``{prefix}{tenant}/exports/``.

        ``temp`` is ignored — S3 has no separate auto-expiring store.
        """
        ext = os.path.splitext(file_name)[1]
        object_key = f"{self._path_prefix}{tenant_id}/exports/{uuid.uuid4()}{ext}"
        await self._store.put_object(object_key, data, content_type_for_ext(ext))
        return f"{S3_SCHEME}{self._bucket_name}/{object_key}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object at ``s3://{bucket}/{key}``."""
        _, object_key = parse_provider_path(
            file_path, S3_SCHEME, expected_bucket=self._bucket_name
        )
        data = await self._store.get_object(object_key)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h SigV4-presigned GET URL for the object."""
        _, object_key = parse_provider_path(
            file_path, S3_SCHEME, expected_bucket=self._bucket_name
        )
        return self._store.presigned_get_url(object_key)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object at ``s3://{bucket}/{key}``."""
        _, object_key = parse_provider_path(
            file_path, S3_SCHEME, expected_bucket=self._bucket_name
        )
        await self._store.delete_object(object_key)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be an ``s3://`` path of this service's bucket;
        anything else is a cross-backend copy and is refused.
        """
        try:
            _, src_key = parse_provider_path(
                src_path, S3_SCHEME, expected_bucket=self._bucket_name
            )
        except StorageBackendError:
            raise CrossBackendCopyError(
                message=f"s3 copy rejected source {src_path!r}"
            ) from None
        ext = os.path.splitext(src_path)[1]
        dest_key = self._object_key(tenant_id, knowledge_id, ext)
        await self._store.copy_object(self._bucket_name, src_key, dest_key)
        return f"{S3_SCHEME}{self._bucket_name}/{dest_key}"

    # ── Internals ───────────────────────────────────────────────────

    def _object_key(self, tenant_id: int, knowledge_id: str, ext: str) -> str:
        return f"{self._path_prefix}{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"

    @property
    def _store(self) -> S3ObjectStore:
        if self._store_cache is None:
            self._store_cache = S3ObjectStore(
                endpoint_url=self._endpoint_url,
                bucket_name=self._bucket_name,
                region=self._region,
                access_key_id=self._access_key_id,
                secret_access_key=self._secret_access_key,
                force_path_style=self._force_path_style,
                provider_label=self._provider_label,
            )
        return self._store_cache


def _trailing_slash(path_prefix: str) -> str:
    """Normalise a prefix so ``{prefix}{tenant}`` reads naturally.

    ``"weknora"`` becomes ``"weknora/"``; an empty prefix stays empty so
    keys start at the tenant segment.
    """
    prefix = path_prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


__all__ = [
    "DEFAULT_S3_REGION",
    "PROVIDER_S3",
    "S3_SCHEME",
    "S3StorageAdapter",
]
