"""MinIO storage adapter.

Mirrors ``internal/application/service/file/minio.go``: the probe checks
the configured bucket exists (``BucketExists``) via a signed ``HEAD``, and
falls back to listing buckets when no bucket is configured. MinIO is
always path-style — a self-hosted deployment has no virtual-host DNS.

``docker`` mode reads endpoint and credentials from the process
environment instead of the row, matching the Go ``Test`` branch that
overlays ``MINIO_*`` env vars before probing.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Final

import httpx

from src.ai.storage.base import (
    PROBE_TIMEOUT_SECONDS,
    head_bucket,
    normalize_endpoint,
)
from src.ai.storage.sigv4 import sign_request
from src.common.exception import StorageBackendError

PROVIDER_MINIO: Final = "minio"

# Deployment mode where the endpoint/credentials come from the env.
MINIO_MODE_DOCKER: Final = "docker"
MINIO_MODE_REMOTE: Final = "remote"

# Env vars the docker mode overlays (Go ``storagebackend.go::Test``).
MINIO_ENDPOINT_ENV: Final = "MINIO_ENDPOINT"
MINIO_ACCESS_KEY_ID_ENV: Final = "MINIO_ACCESS_KEY_ID"
MINIO_SECRET_ACCESS_KEY_ENV: Final = "MINIO_SECRET_ACCESS_KEY"
MINIO_BUCKET_NAME_ENV: Final = "MINIO_BUCKET_NAME"

# MinIO has no bucket-per-subdomain DNS; every request is path-style.
_FORCE_PATH_STYLE: Final = True

# ``ListBuckets`` is a signed GET on the service root.
_SERVICE_ROOT: Final = "/"


class MinioStorageAdapter:
    """Probe for a MinIO (S3-compatible, self-hosted) backend."""

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
    matching the Go branch (endpoint/credentials always come from the env,
    the bucket only as a fallback).
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
    "PROVIDER_MINIO",
    "MinioStorageAdapter",
]
