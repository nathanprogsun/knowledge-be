"""Tencent COS storage adapter (full file interface).

COS is addressed by region rather than by a user-supplied endpoint, so
the host is derived as ``{bucket}[-{app_id}].cos.{region}.myqcloud.com``
and probed with a signed ``HEAD``. ``access_key_id`` / ``secret_access_key``
carry the COS ``SecretID`` / ``SecretKey`` pair (the normalized config
union renames them; the values are the same).

Stored objects are addressed as ``cos://{bucket}/{region}/{objectKey}``
(uploads and raw bytes in the main bucket). A configured temp bucket
receives ``save_bytes(temp=True)`` objects under ``exports/{tenant}/``
and reports them as plain ``https://`` URLs, which the lifecycle rules
auto-expire.
"""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from typing import BinaryIO, Final

from src.ai.storage.base import (
    FileUpload,
    S3ObjectStore,
    head_bucket,
    safe_object_key,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError

PROVIDER_COS: Final = "cos"
COS_SCHEME: Final = "cos://"

# COS buckets are always addressed virtual-host style on this suffix.
_COS_SERVICE_HOST_TEMPLATE: Final = "https://cos.{region}.myqcloud.com"

# Provider schemes a COS service never resolves (cross-backend sources).
_FOREIGN_SCHEMES: Final = ("local://", "minio://", "s3://", "tos://", "oss://", "ks3://", "obs://")

# Default prefix Go applies to COS object keys.
_DEFAULT_COS_PATH_PREFIX: Final = "kb"


class CosStorageAdapter:
    """Full file service for a Tencent Cloud COS backend."""

    def __init__(
        self,
        *,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        app_id: str = "",
        path_prefix: str = _DEFAULT_COS_PATH_PREFIX,
        temp_bucket_name: str = "",
        temp_region: str = "",
    ) -> None:
        self._region = region.strip()
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = _qualified_bucket(bucket_name, app_id)
        self._path_prefix = path_prefix.strip().strip("/")
        self._temp_bucket_name = temp_bucket_name.strip()
        self._temp_region = temp_region.strip() or self._region
        # Region-derived service host; the bucket becomes the leading DNS
        # label, giving ``{bucket}.cos.{region}.myqcloud.com``.
        self._endpoint_url = _COS_SERVICE_HOST_TEMPLATE.format(region=self._region)
        self._store_cache: S3ObjectStore | None = None
        self._temp_store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Signed ``HEAD`` on the region-derived bucket host."""
        if not self._region:
            raise StorageBackendError(
                code="storage_backend.region_required",
                message="COS connectivity check requires a region",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=False,
            provider_label="COS",
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Upload ``file`` to ``{prefix}/{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_name = f"{self._path_prefix}/{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        data = await file.read()
        await self._store.put_object(object_name, data, file.content_type or "")
        return f"{COS_SCHEME}{self._bucket_name}/{self._region}/{object_name}"

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes.

        ``temp`` writes to the temp bucket (when configured) and returns
        its auto-expiring ``https://`` URL; otherwise the main bucket
        stores ``{prefix}/{tenant}/exports/`` and reports a ``cos://``
        path.
        """
        ext = os.path.splitext(file_name)[1]
        if temp and self._temp_store is not None:
            object_name = f"exports/{tenant_id}/{uuid.uuid4()}{ext}"
            await self._temp_store.put_object(object_name, data, "")
            return f"{self._temp_bucket_url}{object_name}"
        object_name = f"{self._path_prefix}/{tenant_id}/exports/{uuid.uuid4()}{ext}"
        await self._store.put_object(object_name, data, "")
        return f"{COS_SCHEME}{self._bucket_name}/{self._region}/{object_name}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object referenced by ``file_path``."""
        object_name = self._parse_object_name(file_path)
        safe_object_key(object_name)
        data = await self._store.get_object(object_name)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h presigned GET URL for the object.

        Temp-bucket ``https://`` paths presign against the temp store;
        everything else against the main store.
        """
        if self._temp_store is not None and file_path.startswith(self._temp_bucket_url):
            object_name = file_path[len(self._temp_bucket_url) :]
            safe_object_key(object_name)
            return self._temp_store.presigned_get_url(object_name)
        object_name = self._parse_object_name(file_path)
        safe_object_key(object_name)
        return self._store.presigned_get_url(object_name)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object referenced by ``file_path``."""
        object_name = self._parse_object_name(file_path)
        safe_object_key(object_name)
        await self._store.delete_object(object_name)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be a ``cos://`` path (or a legacy bucket URL);
        a foreign provider scheme raises ``CrossBackendCopyError``.
        """
        try:
            src_object_key = self._parse_object_name(src_path)
            safe_object_key(src_object_key)
        except StorageBackendError:
            raise CrossBackendCopyError(message=f"cos copy rejected source {src_path!r}") from None
        ext = os.path.splitext(src_path)[1]
        dest_key = f"{self._path_prefix}/{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        # The copy source is the host + object key WITHOUT a scheme, per the
        # COS SDK contract; the store signs a server-side PUT on the dest.
        source_url = f"{self._bucket_name}.cos.{self._region}.myqcloud.com/{src_object_key}"
        await self._store.copy_object_source(dest_key, source_url)
        return f"{COS_SCHEME}{self._bucket_name}/{self._region}/{dest_key}"

    # ── Path parsing ────────────────────────────────────────────────

    def _parse_object_name(self, file_path: str) -> str:
        """Extract the object key from a ``cos://`` path or legacy URL.

        Legacy form: ``https://bucket.cos.region.myqcloud.com/{objectKey}``.
        """
        for other in _FOREIGN_SCHEMES:
            if file_path.startswith(other):
                provider = other.split("://", 1)[0]
                raise StorageBackendError(
                    code="storage_backend.cross_provider_path",
                    message=f"cos file service cannot resolve {provider} path",
                )
        if file_path.startswith(COS_SCHEME):
            rest = file_path[len(COS_SCHEME) :]
            parts = rest.split("/", 2)
            if len(parts) == 3:
                return parts[2]
            return rest
        return file_path[len(self._bucket_url) :]

    @property
    def _bucket_url(self) -> str:
        return f"https://{self._bucket_name}.cos.{self._region}.myqcloud.com/"

    @property
    def _temp_bucket_url(self) -> str:
        return f"https://{self._temp_bucket_name}.cos.{self._temp_region}.myqcloud.com/"

    @property
    def _store(self) -> S3ObjectStore:
        if self._store_cache is None:
            self._store_cache = S3ObjectStore(
                endpoint_url=_COS_SERVICE_HOST_TEMPLATE.format(region=self._region),
                bucket_name=self._bucket_name,
                region=self._region,
                access_key_id=self._access_key_id,
                secret_access_key=self._secret_access_key,
                force_path_style=False,
                provider_label="COS",
            )
        return self._store_cache

    @property
    def _temp_store(self) -> S3ObjectStore | None:
        if not self._temp_bucket_name:
            return None
        if self._temp_store_cache is None:
            self._temp_store_cache = S3ObjectStore(
                endpoint_url=_COS_SERVICE_HOST_TEMPLATE.format(region=self._temp_region),
                bucket_name=self._temp_bucket_name,
                region=self._temp_region,
                access_key_id=self._access_key_id,
                secret_access_key=self._secret_access_key,
                force_path_style=False,
                provider_label="COS",
            )
        return self._temp_store_cache


def _qualified_bucket(bucket_name: str, app_id: str) -> str:
    """Append the COS app id when the bucket name lacks the suffix.

    COS bucket names are globally ``{name}-{appid}``; the UI accepts either
    form, so the suffix is added only when missing.
    """
    name = bucket_name.strip()
    suffix = app_id.strip()
    if not suffix or not name or name.endswith(f"-{suffix}"):
        return name
    return f"{name}-{suffix}"


__all__ = ["COS_SCHEME", "PROVIDER_COS", "CosStorageAdapter"]
