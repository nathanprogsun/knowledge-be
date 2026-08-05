"""Storage-backend adapter protocol + shared S3-compatible probe.

The ``ai`` layer holds the provider SDK/HTTP wiring; it never imports
``core``, ``db`` or ``web``. Each adapter exposes one coroutine —
``check_connectivity`` — mirroring the Go ``FileService.CheckConnectivity``
probe used by ``storageBackendService.Test``. The upload/download half of
the Go ``FileService`` interface lands with the file domain in a later PR;
this module deliberately stops at the read-only probe.

Failures are raised as ``StorageBackendError`` so the service can turn
them into a ``success=false`` response without inspecting HTTP internals.
"""

from __future__ import annotations

import urllib.parse
from typing import Final, Protocol, runtime_checkable

import httpx

from src.ai.storage.sigv4 import sign_request
from src.common.exception import StorageBackendError

# Go bounds every probe at 10s (``context.WithTimeout``).
PROBE_TIMEOUT_SECONDS: Final = 10.0

# Buckets answer a HEAD with 200; 404/301 are "missing"/"wrong region".
_BUCKET_MISSING_STATUSES: Final = frozenset({403, 404})


@runtime_checkable
class StorageAdapter(Protocol):
    """One storage provider's connectivity probe."""

    async def check_connectivity(self) -> None:
        """Raise ``StorageBackendError`` when the backend is unusable."""
        ...


def normalize_endpoint(endpoint: str, *, use_ssl: bool) -> str:
    """Return ``endpoint`` as an absolute URL with no trailing slash.

    A bare ``host:port`` gains a scheme chosen by ``use_ssl``, matching
    ``validateStorageBackendEndpoint``'s scheme inference.
    """
    cleaned = endpoint.strip().rstrip("/")
    if not cleaned:
        return ""
    if "://" not in cleaned:
        scheme = "https" if use_ssl else "http"
        cleaned = f"{scheme}://{cleaned}"
    return cleaned


def endpoint_host(endpoint_url: str) -> str:
    """Extract the ``Host`` header value (with port) from an absolute URL."""
    parsed = urllib.parse.urlsplit(endpoint_url)
    return parsed.netloc


async def head_bucket(
    *,
    endpoint_url: str,
    bucket_name: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    force_path_style: bool,
    provider_label: str,
) -> None:
    """Probe an S3-compatible bucket with a signed ``HEAD``.

    ``force_path_style`` selects ``endpoint/bucket`` over the virtual-host
    form ``bucket.endpoint``. Raises ``StorageBackendError`` when the
    bucket is absent or the credentials are rejected.
    """
    if not bucket_name:
        raise StorageBackendError(
            code="storage_backend.bucket_required",
            message=f"{provider_label} connectivity check requires a bucket name",
        )
    parsed = urllib.parse.urlsplit(endpoint_url)
    if force_path_style:
        host = parsed.netloc
        path = f"/{bucket_name}"
    else:
        host = f"{bucket_name}.{parsed.netloc}"
        path = "/"
    url = urllib.parse.urlunsplit((parsed.scheme, host, path, "", ""))
    headers = sign_request(
        method="HEAD",
        host=host,
        path=path,
        region=region or "us-east-1",
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            response = await client.head(url, headers=headers)
    except httpx.HTTPError as exc:
        raise StorageBackendError(
            code="storage_backend.unreachable",
            message=f"{provider_label} connectivity check failed",
            details={"reason": type(exc).__name__},
        ) from exc
    if response.status_code in _BUCKET_MISSING_STATUSES:
        raise StorageBackendError(
            code="storage_backend.bucket_unavailable",
            message=f'bucket "{bucket_name}" does not exist or is not accessible',
        )
    if response.status_code >= 400:
        raise StorageBackendError(
            code="storage_backend.probe_failed",
            message=f"{provider_label} connectivity check failed",
            details={"status_code": response.status_code},
        )


__all__ = [
    "PROBE_TIMEOUT_SECONDS",
    "StorageAdapter",
    "endpoint_host",
    "head_bucket",
    "normalize_endpoint",
]
