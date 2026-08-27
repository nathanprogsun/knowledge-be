"""Huawei OBS storage adapter (full file interface).

OBS speaks the S3 API through a custom endpoint resolver and is probed
path-style (the upstream passes ``UsePathStyle: true``), so the adapter
fixes that flag rather than reading it from the row.

Stored objects are addressed as ``obs://{bucket}/{objectKey}`` or, when
``OBS_PROXY_DOMAIN`` is configured, ``{proxyDomain}/{objectKey}``. The
proxy-domain layout drops the bucket segment because the proxy already
routes to the concrete instance.
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
    normalize_endpoint,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.common.exception import StorageBackendError

PROVIDER_OBS: Final = "obs"
OBS_SCHEME: Final = "obs://"

# OBS is constructed path-style (``UsePathStyle: true``).
_FORCE_PATH_STYLE: Final = True

# Optional proxy domain that replaces the ``obs://`` scheme.
OBS_PROXY_DOMAIN_ENV: Final = "OBS_PROXY_DOMAIN"


class ObsStorageAdapter:
    """Full file service for a Huawei Cloud OBS backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        use_ssl: bool = True,
        path_prefix: str = "",
        proxy_domain: str | None = None,
    ) -> None:
        self._endpoint_url = normalize_endpoint(endpoint, use_ssl=use_ssl)
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket_name = bucket_name
        self._path_prefix = path_prefix.strip().strip("/")
        configured_proxy = (
            proxy_domain if proxy_domain is not None else os.environ.get(OBS_PROXY_DOMAIN_ENV, "")
        )
        self._proxy_domain = configured_proxy.strip().rstrip("/")
        self._store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Signed path-style ``HEAD`` on the configured bucket."""
        if not self._endpoint_url:
            raise StorageBackendError(
                code="storage_backend.endpoint_required",
                message="OBS connectivity check requires an endpoint",
            )
        await head_bucket(
            endpoint_url=self._endpoint_url,
            bucket_name=self._bucket_name,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            force_path_style=_FORCE_PATH_STYLE,
            provider_label="OBS",
        )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Upload ``file`` to ``{prefix}{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_key = self._knowledge_key(tenant_id, knowledge_id, ext)
        data = await file.read()
        content_type = file.content_type or "application/octet-stream"
        await self._store.put_object(object_key, data, content_type)
        return self._path(object_key)

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes.

        ``temp`` writes under ``{prefix}temp/{tenant}/`` (auto-expired by
        lifecycle rules); non-temp bytes land under ``{prefix}{tenant}/``.
        """
        ext = os.path.splitext(file_name)[1]
        if temp:
            object_key = self._scoped_key(f"temp/{tenant_id}", ext)
        else:
            object_key = self._scoped_key(str(tenant_id), ext)
        await self._store.put_object(object_key, data, "application/octet-stream")
        return self._path(object_key)

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object referenced by ``file_path``."""
        object_key = self._parse_object_key(file_path)
        data = await self._store.get_object(object_key)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A download URL for the object.

        ``http(s)://`` inputs pass through unchanged (already public);
        otherwise the proxy domain or the endpoint/bucket URL is
        returned.
        """
        if file_path.startswith(("http://", "https://")):
            return file_path
        object_key = self._parse_object_key(file_path)
        if self._proxy_domain:
            return f"{self._proxy_domain}/{object_key.lstrip('/')}"
        return f"{self._endpoint_url}/{self._bucket_name}/{object_key.lstrip('/')}"

    async def delete_file(self, file_path: str) -> None:
        """Remove the object referenced by ``file_path``."""
        object_key = self._parse_object_key(file_path)
        await self._store.delete_object(object_key)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must use this service's prefix (proxy domain or
        ``obs://``); anything else is a cross-backend copy and is
        refused.
        """
        if not src_path.startswith(self._prefix):
            raise CrossBackendCopyError(message=f"obs copy rejected source {src_path!r}")
        src_key = self._parse_object_key(src_path)
        ext = os.path.splitext(src_path)[1]
        dest_key = self._knowledge_key(tenant_id, knowledge_id, ext)
        await self._store.copy_object(self._bucket_name, src_key, dest_key)
        return self._path(dest_key)

    # ── Internals ───────────────────────────────────────────────────

    def _knowledge_key(self, tenant_id: int, knowledge_id: str, ext: str) -> str:
        return self._scoped_key(f"{tenant_id}/{knowledge_id}", ext)

    def _scoped_key(self, scope: str, ext: str) -> str:
        if self._path_prefix:
            return f"{self._path_prefix}/{scope}/{uuid.uuid4()}{ext}"
        return f"{scope}/{uuid.uuid4()}{ext}"

    @property
    def _prefix(self) -> str:
        if self._proxy_domain:
            return f"{self._proxy_domain}/"
        return OBS_SCHEME

    def _path(self, object_key: str) -> str:
        if self._proxy_domain:
            return f"{self._prefix}{object_key}"
        return f"{self._prefix}{self._bucket_name}/{object_key}"

    def _parse_object_key(self, file_path: str) -> str:
        """Extract the object key from a provider path.

        Non-prefixed paths are returned unchanged (legacy behavior),
        mirroring the upstream parser's fallback.
        """
        if not file_path.startswith(self._prefix):
            return file_path
        rest = file_path[len(self._prefix) :]
        if self._proxy_domain:
            rest = rest.lstrip("/")
            if not rest:
                raise StorageBackendError(
                    code="storage_backend.invalid_path",
                    message=f"invalid OBS file path: {file_path}",
                )
            return rest
        parts = rest.split("/", 1)
        if len(parts) == 2 and parts[0] == self._bucket_name and parts[1]:
            return parts[1]
        raise StorageBackendError(
            code="storage_backend.invalid_path",
            message=f"invalid OBS file path: {file_path}",
        )

    @property
    def _store(self) -> S3ObjectStore:
        if self._store_cache is None:
            self._store_cache = S3ObjectStore(
                endpoint_url=self._endpoint_url,
                bucket_name=self._bucket_name,
                region=self._region,
                access_key_id=self._access_key_id,
                secret_access_key=self._secret_access_key,
                force_path_style=_FORCE_PATH_STYLE,
                provider_label="OBS",
            )
        return self._store_cache


__all__ = ["OBS_PROXY_DOMAIN_ENV", "OBS_SCHEME", "PROVIDER_OBS", "ObsStorageAdapter"]
