"""Storage-backend adapter protocols, shared helpers and S3-compatible client.

The ``ai`` layer holds the provider SDK/HTTP wiring; it never imports
``core``, ``db`` or ``web``. Every adapter exposes the full file
interface (``FileService`` — check connectivity, save, read, delete,
copy, URL) mirroring the upstream contract. ``StorageAdapter`` is kept
as the probe-only view so the registry service can dial a backend
without touching file I/O.

Failures are raised as ``StorageBackendError`` so callers can turn them
into API errors without inspecting HTTP internals.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from typing import BinaryIO, Final, Protocol, runtime_checkable

import httpx

from src.ai.storage.sigv4 import presign_get_url, sign_request
from src.common.exception import StorageBackendError, ValidationError

# Go bounds every probe at 10s (``context.WithTimeout``).
PROBE_TIMEOUT_SECONDS: Final = 10.0

# Buckets answer a HEAD with 200; 404/301 are "missing"/"wrong region".
_BUCKET_MISSING_STATUSES: Final = frozenset({403, 404})

# Prefix wrapping a provider path with the concrete backend instance id:
# ``storage://<backendID>/<provider>://...``.
STORAGE_BACKEND_SCHEME: Final = "storage://"

# 24h presigned-URL validity, matching the providers' GetFileURL.
_PRESIGNED_URL_TTL_SECONDS: Final = 24 * 3600

# Default TTL for the local presigned download URLs (2h upstream).
_LOCAL_PRESIGN_TTL_SECONDS: Final = 2 * 3600


@runtime_checkable
class StorageAdapter(Protocol):
    """One storage provider's connectivity probe."""

    async def check_connectivity(self) -> None:
        """Raise ``StorageBackendError`` when the backend is unusable."""
        ...


@runtime_checkable
class FileUpload(Protocol):
    """The uploaded-file surface ``save_file`` consumes.

    ``fastapi.UploadFile`` satisfies this protocol (``filename``,
    ``size``, ``content_type`` plus an async ``read``).
    """

    filename: str
    size: int
    content_type: str

    async def read(self) -> bytes:
        """Return the whole uploaded payload."""
        ...


@runtime_checkable
class FileService(Protocol):
    """Full storage-backend file interface (mirrors the upstream contract)."""

    async def check_connectivity(self) -> None:
        """Verify the backend is reachable and correctly configured."""
        ...

    async def save_file(self, *, file: FileUpload, tenant_id: int, knowledge_id: str) -> str:
        """Persist an uploaded file and return its provider:// path."""
        ...

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        """Persist raw bytes and return their provider:// path.

        ``temp`` requests the temporary store when the backend has one.
        """
        ...

    async def get_file(self, file_path: str) -> BinaryIO:
        """Open a stored object for reading (caller closes the handle)."""
        ...

    async def get_file_url(self, file_path: str) -> str:
        """Return a download URL for the object (presigned when supported)."""
        ...

    async def delete_file(self, file_path: str) -> None:
        """Remove a stored object."""
        ...

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        """Copy an object to a new knowledge-owned object.

        Returns the new provider:// path. Raises
        ``CrossBackendCopyError`` when ``src_path`` belongs to a
        different storage provider.
        """
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


# ── Path and name helpers ──────────────────────────────────────────────


def join_object_key(parts: list[str]) -> str:
    """Join object-key segments, dropping empties and slashes.

    Mirrors the ``join*Key`` helpers the object providers share: each
    part is trimmed of ``/`` and blank parts are skipped, so a
    ``path_prefix`` with or without trailing slashes yields one key.
    """
    filtered: list[str] = []
    for part in parts:
        cleaned = part.strip("/")
        if cleaned:
            filtered.append(cleaned)
    return "/".join(filtered)


def safe_file_name(file_name: str) -> str:
    """Return the safe basename of ``file_name`` or raise ``ValidationError``.

    Rejects empty names, ``.``/``..``, path traversal and over-long
    names, mirroring ``SafeFileName`` used by the SaveBytes call sites.
    """
    if not file_name:
        raise ValidationError(
            code="storage_backend.empty_file_name", message="file name cannot be empty"
        )
    base = os.path.basename(os.path.normpath(file_name))
    if base in ("", ".", "..") or ".." in base:
        raise ValidationError(
            code="storage_backend.invalid_file_name",
            message="invalid file name: path traversal or empty name",
        )
    if len(base) > 255:
        raise ValidationError(
            code="storage_backend.file_name_too_long", message="file name too long"
        )
    return base


def safe_object_key(object_key: str) -> None:
    """Raise ``ValidationError`` when an object key contains traversal.

    Mirrors ``SafeObjectKey``: keys are namespaced by the provider
    layout, but a ``..`` segment is never legitimate.
    """
    if not object_key:
        raise ValidationError(
            code="storage_backend.empty_object_key", message="object key cannot be empty"
        )
    if ".." in object_key:
        raise ValidationError(
            code="storage_backend.object_key_traversal",
            message="object key contains path traversal",
        )


def parse_provider_path(
    file_path: str, scheme: str, *, expected_bucket: str | None = None
) -> tuple[str, str]:
    """Split ``{scheme}//{bucket}/{objectKey}`` into ``(bucket, key)``.

    ``expected_bucket`` enforces the single-bucket rule the S3-shaped
    providers apply to their own paths. Raises ``StorageBackendError``
    for malformed paths and ``ValidationError`` for traversal keys.
    """
    if not file_path.startswith(scheme):
        raise StorageBackendError(
            code="storage_backend.invalid_path",
            message=f"invalid {scheme} file path: {file_path}",
        )
    rest = file_path[len(scheme) :]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise StorageBackendError(
            code="storage_backend.invalid_path",
            message=f"invalid {scheme} file path: {file_path}",
        )
    bucket, key = parts[0], parts[1]
    if expected_bucket is not None and bucket != expected_bucket:
        raise StorageBackendError(
            code="storage_backend.bucket_mismatch",
            message=f"bucket mismatch in path: got {bucket}, want {expected_bucket}",
        )
    safe_object_key(key)
    return bucket, key


def content_type_for_ext(ext: str) -> str:
    """Map a file extension to a content type (octet-stream fallback).

    Mirrors ``GetContentTypeByExt``: active browser-content extensions
    (svg/html/js/...) are always served as ``application/octet-stream``
    so a stored object can never be executed inline.
    """
    lowered = ext.lower()
    if lowered in (
        ".svg",
        ".svgz",
        ".html",
        ".htm",
        ".xhtml",
        ".xml",
        ".js",
        ".mjs",
        ".css",
    ):
        return "application/octet-stream"
    content_types: dict[str, str] = {
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".txt": "text/plain; charset=utf-8",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".zip": "application/zip",
        ".tar": "application/x-tar",
        ".gz": "application/gzip",
        ".md": "text/markdown; charset=utf-8",
    }
    return content_types.get(lowered, "application/octet-stream")


def build_storage_backend_path(backend_id: str, provider_path: str) -> str:
    """Wrap a provider:// path with the concrete backend instance id."""
    return f"{STORAGE_BACKEND_SCHEME}{backend_id.strip()}/{provider_path}"


def parse_storage_backend_path(path: str) -> tuple[str, str] | None:
    """Split ``storage://<backendID>/<providerPath>``.

    Returns ``None`` when the path is not instance-wrapped.
    """
    if not path.startswith(STORAGE_BACKEND_SCHEME):
        return None
    rest = path[len(STORAGE_BACKEND_SCHEME) :]
    parts = rest.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def parse_tenant_id_from_storage_path(file_path: str) -> int:
    """Extract the first numeric tenant segment from a provider path.

    ``storage://<backendID>/`` wrappers are unwrapped first so the scan
    anchors on the provider path, not the backend id. Returns 0 when no
    numeric segment exists.
    """
    parsed = parse_storage_backend_path(file_path)
    if parsed is not None:
        file_path = parsed[1]
    if "://" not in file_path:
        return 0
    rest = file_path.split("://", 1)[1]
    for segment in rest.split("/"):
        if segment.isdigit():
            return int(segment)
    return 0


def sign_file_url(*, base_url: str, file_path: str, tenant_id: int, ttl: int = 0) -> str:
    """Build an HMAC-signed download URL for a stored file.

    Requires ``SYSTEM_AES_KEY`` (at least 16 bytes) to be configured;
    raises ``StorageBackendError`` otherwise. ``ttl`` seconds, defaulting
    to 2 hours when not positive. The signed URL carries the provider
    path, the owning tenant and an expiry, mirroring the upstream
    presign util.
    """
    key = os.environ.get("SYSTEM_AES_KEY", "").encode("utf-8")
    if len(key) < 16:
        raise StorageBackendError(
            code="storage_backend.presign_key_missing",
            message="presigning a file URL requires SYSTEM_AES_KEY",
        )
    ttl_seconds = ttl if ttl > 0 else _LOCAL_PRESIGN_TTL_SECONDS
    expires = int(time.time()) + ttl_seconds
    payload = f"file_path={file_path}&tenant_id={tenant_id}&expires={expires}"
    signature = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    base = base_url.rstrip("/")
    query = urllib.parse.urlencode(
        {
            "file_path": file_path,
            "tenant_id": tenant_id,
            "expires": expires,
            "sig": signature,
        }
    )
    return f"{base}/api/v1/files/presigned?{query}"


# ── S3-compatible object client ────────────────────────────────────────


class S3ObjectStore:
    """Minimal S3-compatible object store over httpx + SigV4 signing.

    One instance addresses one bucket; a backend with a temporary bucket
    holds a second instance. ``force_path_style`` selects
    ``endpoint/bucket/key`` over virtual-host addressing. All file
    operations are async; failures surface as ``StorageBackendError``.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        bucket_name: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        force_path_style: bool,
        provider_label: str,
    ) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._bucket_name = bucket_name
        self._region = region or "us-east-1"
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._force_path_style = force_path_style
        self._provider_label = provider_label
        self._parsed = urllib.parse.urlsplit(self._endpoint_url)

    @property
    def bucket_name(self) -> str:
        """The bucket this instance addresses."""
        return self._bucket_name

    def object_url(self, key: str) -> str:
        """The absolute HTTP URL for an object key."""
        host, raw_path = self._resolve(key)
        path = urllib.parse.quote(raw_path, safe="/~")
        return urllib.parse.urlunsplit((self._parsed.scheme, host, path, "", ""))

    def presigned_get_url(self, key: str, expires_seconds: int = _PRESIGNED_URL_TTL_SECONDS) -> str:
        """A SigV4-presigned GET URL valid for ``expires_seconds``."""
        host, raw_path = self._resolve(key)
        return presign_get_url(
            scheme=self._parsed.scheme,
            host=host,
            path=raw_path,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            expires_seconds=expires_seconds,
        )

    async def put_object(self, key: str, body: bytes, content_type: str) -> None:
        """Upload ``body`` to ``key`` with the given content type."""
        self._require_bucket()
        headers = self._signed_headers(
            "PUT",
            key,
            payload=body,
            extra={"Content-Type": content_type or "application/octet-stream"},
        )
        await self._request(
            "PUT", self.object_url(key), headers=headers, content=body, action="upload"
        )

    async def get_object(self, key: str) -> bytes:
        """Download the object content at ``key``."""
        self._require_bucket()
        headers = self._signed_headers("GET", key)
        response = await self._request(
            "GET", self.object_url(key), headers=headers, action="download"
        )
        return response.content

    async def delete_object(self, key: str) -> None:
        """Remove the object at ``key``."""
        self._require_bucket()
        headers = self._signed_headers("DELETE", key)
        await self._request("DELETE", self.object_url(key), headers=headers, action="delete")

    async def copy_object(self, src_bucket: str, src_key: str, dest_key: str) -> None:
        """Server-side copy ``src_bucket/src_key`` onto ``dest_key``.

        The destination lives in this instance's bucket; the source may
        be any bucket of the same provider (e.g. the temporary bucket).
        The ``x-amz-copy-source`` value keeps its ``/`` unescaped, as
        the upstream S3 clients require.
        """
        await self.copy_object_source(dest_key, f"{src_bucket}/{src_key}")

    async def copy_object_source(self, dest_key: str, source: str) -> None:
        """Server-side copy with an explicit ``x-amz-copy-source`` value.

        ``source`` is the provider-specific source reference (``bucket/key``
        for S3-compatible stores, ``bucket.region.host/key`` without a
        scheme for COS). The header is passed through unescaped so the
        bucket/key split survives.
        """
        self._require_bucket()
        headers = self._signed_headers(
            "PUT",
            dest_key,
            payload=b"",
            extra={"x-amz-copy-source": source},
        )
        await self._request(
            "PUT", self.object_url(dest_key), headers=headers, content=b"", action="copy"
        )

    def _resolve(self, key: str) -> tuple[str, str]:
        """Return ``(host, raw_request_path)`` for an object key."""
        if self._force_path_style:
            return self._parsed.netloc, f"/{self._bucket_name}/{key}"
        return f"{self._bucket_name}.{self._parsed.netloc}", f"/{key}"

    def _signed_headers(
        self,
        method: str,
        key: str,
        *,
        payload: bytes | None = None,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        host, raw_path = self._resolve(key)
        headers = sign_request(
            method=method,
            host=host,
            path=raw_path,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            payload=payload,
        )
        if extra:
            headers.update(extra)
        return headers

    def _require_bucket(self) -> None:
        if not self._bucket_name:
            raise StorageBackendError(
                code="storage_backend.bucket_required",
                message=f"{self._provider_label} file operations require a bucket name",
            )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        content: bytes | None = None,
        action: str,
    ) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
                response = await client.request(method, url, content=content, headers=headers)
        except httpx.HTTPError as exc:
            raise StorageBackendError(
                code="storage_backend.unreachable",
                message=f"{self._provider_label} {action} failed",
                details={"reason": type(exc).__name__},
            ) from exc
        if response.status_code >= 300:
            raise StorageBackendError(
                code="storage_backend.request_failed",
                message=f"{self._provider_label} {action} failed",
                details={"status_code": response.status_code},
            )
        return response


__all__ = [
    "PROBE_TIMEOUT_SECONDS",
    "STORAGE_BACKEND_SCHEME",
    "FileService",
    "FileUpload",
    "S3ObjectStore",
    "StorageAdapter",
    "build_storage_backend_path",
    "content_type_for_ext",
    "endpoint_host",
    "head_bucket",
    "join_object_key",
    "normalize_endpoint",
    "parse_provider_path",
    "parse_storage_backend_path",
    "parse_tenant_id_from_storage_path",
    "safe_file_name",
    "safe_object_key",
    "sign_file_url",
]
