"""MinIO storage adapter (full file interface).

The probe checks the configured bucket exists (``BucketExists``) via a
signed ``HEAD``, and falls back to listing buckets when no bucket is
configured. MinIO is always path-style — a self-hosted deployment has
no virtual-host DNS.

``docker`` mode reads endpoint and credentials from the process
environment instead of the row, matching the upstream ``Test`` branch
that overlays ``MINIO_*`` env vars before probing.

File operations are addressed as ``minio://{bucket}/{tenant}/{knowledgeId}/{uuid}{ext}``
(uploads) and ``minio://{bucket}/{tenant}/exports/{uuid}{ext}`` (raw
bytes); MinIO stores no path prefix, so the key starts at the tenant
segment.
"""

from __future__ import annotations

import os
import urllib.parse
import uuid
from io import BytesIO
from typing import BinaryIO, Final

import httpx

from src.ai.storage.base import (
    PROBE_TIMEOUT_SECONDS,
    FileUpload,
    S3ObjectStore,
    content_type_for_ext,
    head_bucket,
    normalize_endpoint,
    parse_provider_path,
)
from src.ai.storage.errors import CrossBackendCopyError
from src.ai.storage.sigv4 import sign_request
from src.common.exception import StorageBackendError

PROVIDER_MINIO: Final = "minio"
MINIO_SCHEME: Final = "minio://"

# Deployment mode where the endpoint/credentials come from the env.
MINIO_MODE_DOCKER: Final = "docker"
MINIO_MODE_REMOTE: Final = "remote"

# Env vars the docker mode overlays.
MINIO_ENDPOINT_ENV: Final = "MINIO_ENDPOINT"
MINIO_ACCESS_KEY_ID_ENV: Final = "MINIO_ACCESS_KEY_ID"
MINIO_SECRET_ACCESS_KEY_ENV: Final = "MINIO_SECRET_ACCESS_KEY"
MINIO_BUCKET_NAME_ENV: Final = "MINIO_BUCKET_NAME"

# MinIO has no bucket-per-subdomain DNS; every request is path-style.
_FORCE_PATH_STYLE: Final = True

# ``ListBuckets`` is a signed GET on the service root.
_SERVICE_ROOT: Final = "/"


class MinioStorageAdapter:
    """Full file service for a MinIO (S3-compatible, self-hosted) backend."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str = "",
        use_ssl: bool = False,
        region: str = "",
        mode: str = MINIO_MODE_REMOTE,
    ) -> None:
        resolved = _resolve_docker_overrides(
            mode=mode,
            endpoint=endpoint,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
        )
        self._endpoint_url = normalize_endpoint(resolved.endpoint, use_ssl=use_ssl)
        self._access_key_id = resolved.access_key_id
        self._secret_access_key = resolved.secret_access_key
        self._bucket_name = resolved.bucket_name
        self._region = region or "us-east-1"
        self._store_cache: S3ObjectStore | None = None

    async def check_connectivity(self) -> None:
        """Verify the bucket exists, or that the service answers at all.

        Read-only: a missing bucket is an error, never an implicit create.
        """
        if not self._endpoint_url:
            raise StorageBackendError(
                code="storage_backend.endpoint_required",
                message="MinIO connectivity check requires an endpoint",
            )
        if self._bucket_name:
            await head_bucket(
                endpoint_url=self._endpoint_url,
                bucket_name=self._bucket_name,
                region=self._region,
                access_key_id=self._access_key_id,
                secret_access_key=self._secret_access_key,
                force_path_style=_FORCE_PATH_STYLE,
                provider_label="MinIO",
            )
            return
        await self._list_buckets()

    async def _list_buckets(self) -> None:
        """Signed ``GET /`` — proves the credentials are accepted."""
        parsed = urllib.parse.urlsplit(self._endpoint_url)
        host = parsed.netloc
        headers = sign_request(
            method="GET",
            host=host,
            path=_SERVICE_ROOT,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
        )
        url = urllib.parse.urlunsplit((parsed.scheme, host, _SERVICE_ROOT, "", ""))
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise StorageBackendError(
                code="storage_backend.unreachable",
                message="MinIO connectivity check failed",
                details={"reason": type(exc).__name__},
            ) from exc
        if response.status_code >= 400:
            raise StorageBackendError(
                code="storage_backend.probe_failed",
                message="MinIO connectivity check failed",
                details={"status_code": response.status_code},
            )

    # ── File operations ─────────────────────────────────────────────

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Upload ``file`` to ``{tenant}/{knowledge}/{uuid}{ext}``."""
        ext = os.path.splitext(file.filename)[1]
        object_key = f"{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        data = await file.read()
        content_type = file.content_type or content_type_for_ext(ext)
        await self._store.put_object(object_key, data, content_type)
        return f"{MINIO_SCHEME}{self._bucket_name}/{object_key}"

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Upload raw bytes to ``{tenant}/exports/``.

        ``temp`` is ignored — MinIO has no separate auto-expiring store.
        """
        ext = os.path.splitext(file_name)[1]
        object_key = f"{tenant_id}/exports/{uuid.uuid4()}{ext}"
        await self._store.put_object(object_key, data, content_type_for_ext(ext))
        return f"{MINIO_SCHEME}{self._bucket_name}/{object_key}"

    async def get_file(self, file_path: str) -> BinaryIO:
        """Download the object at ``minio://{bucket}/{key}``."""
        _, object_key = parse_provider_path(
            file_path, MINIO_SCHEME, expected_bucket=self._bucket_name
        )
        data = await self._store.get_object(object_key)
        return BytesIO(data)

    async def get_file_url(self, file_path: str) -> str:
        """A 24h SigV4-presigned GET URL for the object."""
        _, object_key = parse_provider_path(
            file_path, MINIO_SCHEME, expected_bucket=self._bucket_name
        )
        return self._store.presigned_get_url(object_key)

    async def delete_file(self, file_path: str) -> None:
        """Remove the object at ``minio://{bucket}/{key}``."""
        _, object_key = parse_provider_path(
            file_path, MINIO_SCHEME, expected_bucket=self._bucket_name
        )
        await self._store.delete_object(object_key)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Server-side copy to a new knowledge-owned object.

        The source must be a ``minio://`` path of this service's bucket;
        anything else is a cross-backend copy and is refused.
        """
        try:
            _, src_key = parse_provider_path(
                src_path, MINIO_SCHEME, expected_bucket=self._bucket_name
            )
        except StorageBackendError:
            raise CrossBackendCopyError(
                message=f"minio copy rejected source {src_path!r}"
            ) from None
        ext = os.path.splitext(src_path)[1]
        dest_key = f"{tenant_id}/{knowledge_id}/{uuid.uuid4()}{ext}"
        await self._store.copy_object(self._bucket_name, src_key, dest_key)
        return f"{MINIO_SCHEME}{self._bucket_name}/{dest_key}"

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
                provider_label="MinIO",
            )
        return self._store_cache


class _MinioCredentials:
    """Endpoint + credential set after the docker-mode env overlay."""

    __slots__ = ("access_key_id", "bucket_name", "endpoint", "secret_access_key")

    def __init__(
        self,
        *,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
    ) -> None:
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name


def _resolve_docker_overrides(
    *,
    mode: str,
    endpoint: str,
    access_key_id: str,
    secret_access_key: str,
    bucket_name: str,
) -> _MinioCredentials:
    """Overlay ``MINIO_*`` env vars when the row is in ``docker`` mode.

    The bucket is only taken from the env when the row leaves it blank,
    matching the upstream branch (endpoint/credentials always come from
    the env, the bucket only as a fallback).
    """
    if mode != MINIO_MODE_DOCKER:
        return _MinioCredentials(
            endpoint=endpoint,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
        )
    return _MinioCredentials(
        endpoint=os.environ.get(MINIO_ENDPOINT_ENV, ""),
        access_key_id=os.environ.get(MINIO_ACCESS_KEY_ID_ENV, ""),
        secret_access_key=os.environ.get(MINIO_SECRET_ACCESS_KEY_ENV, ""),
        bucket_name=bucket_name or os.environ.get(MINIO_BUCKET_NAME_ENV, ""),
    )


__all__ = [
    "MINIO_MODE_DOCKER",
    "MINIO_MODE_REMOTE",
    "MINIO_SCHEME",
    "PROVIDER_MINIO",
    "MinioStorageAdapter",
]
