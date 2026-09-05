"""Persist downloadable ``file_url`` bytes onto the knowledge row.

Create already classifies downloadable URLs as type ``file_url`` and
ordinary web pages as type ``url``. Preview and download stream
``file_path``; a ``file_url`` row with an empty path must download,
``save_bytes``, and write that path before parse. Type ``url`` stays
without bytes so those pages keep returning ``knowledge.file_unavailable``.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import httpx

from src.ai.storage.base import FileService
from src.common.exception import ExternalServiceError, ValidationError
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.knowledge.documents.create_common import (
    is_safe_url,
    is_valid_http_url,
    normalize_file_extension,
)
from src.core.knowledge.documents.create_file import StorageResolver
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

KNOWLEDGE_TYPE_FILE_URL = "file_url"

_STORAGE_ENGINE_REQUIRED_CODE = "knowledge.storage_engine_required"
_INVALID_URL_CODE = "knowledge.invalid_url"
_FETCH_FAILED_CODE = "knowledge.file_url_fetch_failed"
_HTML_REJECTED_CODE = "knowledge.file_url_html_rejected"
_TOO_LARGE_CODE = "knowledge.file_url_too_large"

# Same 50 MB cap as temporary uploads. A file URL is a downloadable
# object, not an unbounded web crawl.
_MAX_FILE_URL_BYTES = 50 * 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_MAX_REDIRECTS = 10

_HTML_MEDIA_TYPES: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})
_HTML_FILE_TYPES: frozenset[str] = frozenset({"html", "htm", "mhtml"})

_DOWNLOAD_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


UrlGuard = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class FileUrlDownload:
    """Bytes and media type from one file-URL GET."""

    data: bytes
    content_type: str


class FileUrlDownloader(Protocol):
    """Download seam so tests never open the network."""

    async def download(self, *, url: str) -> FileUrlDownload:
        """Return the response body or raise a typed application error."""
        ...


@dataclass(frozen=True)
class FileUrlStoreResult:
    """Row plus the parse source after an optional ``save_bytes``."""

    row: Document
    file_path: str
    file_content: bytes | None
    url: str


def _effective_file_path(row: Document, file_path: str) -> str:
    incoming = file_path.strip()
    if incoming:
        return incoming
    return (row.file_path or "").strip()


def _effective_url(row: Document, url: str) -> str:
    incoming = url.strip()
    if incoming:
        return incoming
    return (row.source or "").strip()


def _is_html_web_page(*, content_type: str, file_type: str) -> bool:
    media = content_type.split(";", 1)[0].strip().lower()
    if media not in _HTML_MEDIA_TYPES:
        return False
    return normalize_file_extension(file_type) not in _HTML_FILE_TYPES


def _reject_html_page(*, content_type: str, file_type: str) -> None:
    if _is_html_web_page(content_type=content_type, file_type=file_type):
        raise ValidationError(
            code=_HTML_REJECTED_CODE,
            message="HTML web pages are not stored as files",
        )


async def _guard_url(url: str, ssrf_guard: UrlGuard) -> None:
    try:
        await ssrf_guard(url)
    except ValidationError as exc:
        raise ValidationError(code=_INVALID_URL_CODE, message=exc.message) from exc


def _next_redirect(current: str, response: httpx.Response) -> str:
    location = response.headers.get("location")
    if not location:
        raise ExternalServiceError(
            code=_FETCH_FAILED_CODE,
            message="file URL redirect is missing a Location header",
        )
    joined = str(urllib.parse.urljoin(current, location))
    if not is_valid_http_url(joined) or not is_safe_url(joined):
        raise ValidationError(
            code=_INVALID_URL_CODE,
            message="invalid or unsafe file URL redirect",
        )
    return joined


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ValidationError(
                code=_TOO_LARGE_CODE,
                message="file URL exceeds the download size limit",
            )
        chunks.append(chunk)
    if total == 0:
        raise ExternalServiceError(
            code=_FETCH_FAILED_CODE,
            message="file URL download returned an empty body",
        )
    return b"".join(chunks)


async def _download_with_redirects(
    *,
    url: str,
    client: httpx.AsyncClient,
    ssrf_guard: UrlGuard,
    max_bytes: int,
) -> FileUrlDownload:
    current = url
    for _ in range(_MAX_REDIRECTS):
        await _guard_url(current, ssrf_guard)
        async with client.stream("GET", current, headers=_DOWNLOAD_HEADERS) as response:
            if response.is_redirect:
                current = _next_redirect(current, response)
                continue
            if response.status_code >= 400:
                raise ExternalServiceError(
                    code=_FETCH_FAILED_CODE,
                    message=f"file URL download failed with HTTP {response.status_code}",
                )
            data = await _read_capped(response, max_bytes)
            content_type = response.headers.get("content-type", "")
            return FileUrlDownload(data=data, content_type=content_type)
    raise ExternalServiceError(
        code=_FETCH_FAILED_CODE,
        message="file URL download exceeded the redirect limit",
    )


async def download_file_url(
    *,
    url: str,
    client: httpx.AsyncClient | None = None,
    ssrf_guard: UrlGuard | None = None,
    max_bytes: int = _MAX_FILE_URL_BYTES,
) -> FileUrlDownload:
    """GET ``url`` with per-hop SSRF checks and a body-size cap."""
    if not is_valid_http_url(url) or not is_safe_url(url):
        raise ValidationError(
            code=_INVALID_URL_CODE,
            message="invalid or unsafe file URL",
        )
    guard = ssrf_guard if ssrf_guard is not None else validate_ssrf_safe_url
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        follow_redirects=False,
    )
    try:
        return await _download_with_redirects(
            url=url,
            client=http,
            ssrf_guard=guard,
            max_bytes=max_bytes,
        )
    except httpx.HTTPError as exc:
        raise ExternalServiceError(
            code=_FETCH_FAILED_CODE,
            message=f"file URL download failed: {exc}",
        ) from exc
    finally:
        if owns_client:
            await http.aclose()


class HttpxFileUrlDownloader:
    """Default downloader backed by httpx."""

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def download(self, *, url: str) -> FileUrlDownload:
        return await download_file_url(url=url, client=self._client)


async def _resolve_file_service(
    resolver: StorageResolver | None,
    row: Document,
) -> FileService:
    if resolver is None:
        raise ValidationError(
            code=_STORAGE_ENGINE_REQUIRED_CODE,
            message="storage engine is not configured",
        )
    service = await resolver.resolve_file_service(
        knowledge_base_id=row.knowledge_base_id,
        tenant_id=row.tenant_id,
    )
    if service is None:
        raise ValidationError(
            code=_STORAGE_ENGINE_REQUIRED_CODE,
            message="storage engine is not configured",
        )
    return service


async def _download_and_persist(
    *,
    row: Document,
    source_url: str,
    file_name: str,
    file_type: str,
    resolver: StorageResolver | None,
    downloader: FileUrlDownloader | None,
    knowledge_repo: KnowledgeRepository,
    now: datetime,
) -> FileUrlStoreResult:
    if not source_url:
        raise ValidationError(
            code=_INVALID_URL_CODE,
            message="file URL is required",
        )
    service = await _resolve_file_service(resolver, row)
    fetcher = downloader if downloader is not None else HttpxFileUrlDownloader()
    downloaded = await fetcher.download(url=source_url)
    _reject_html_page(content_type=downloaded.content_type, file_type=file_type)
    name = file_name.strip() or (row.file_name or "").strip() or "download"
    saved = await service.save_bytes(
        data=downloaded.data,
        tenant_id=row.tenant_id,
        file_name=name,
        temp=False,
    )
    updated = row.model_copy(update={"file_path": saved, "updated_at": now})
    persisted = await knowledge_repo.update(updated)
    return FileUrlStoreResult(
        row=persisted,
        file_path=saved,
        file_content=downloaded.data,
        url="",
    )


async def store_file_url_bytes(
    *,
    row: Document,
    file_path: str,
    url: str,
    file_name: str,
    file_type: str,
    resolver: StorageResolver | None,
    downloader: FileUrlDownloader | None,
    knowledge_repo: KnowledgeRepository,
    now: datetime,
) -> FileUrlStoreResult:
    """Download and persist bytes only for an empty-path ``file_url`` row.

    Type ``url`` is left untouched. A ``file_url`` row that already has
    ``file_path`` is idempotent and skips ``save_bytes``.
    """
    source_url = _effective_url(row, url)
    existing = _effective_file_path(row, file_path)
    if row.type != KNOWLEDGE_TYPE_FILE_URL:
        return FileUrlStoreResult(
            row=row,
            file_path=existing,
            file_content=None,
            url=source_url,
        )
    if existing:
        return FileUrlStoreResult(
            row=row,
            file_path=existing,
            file_content=None,
            url=source_url,
        )
    return await _download_and_persist(
        row=row,
        source_url=source_url,
        file_name=file_name,
        file_type=file_type,
        resolver=resolver,
        downloader=downloader,
        knowledge_repo=knowledge_repo,
        now=now,
    )


__all__ = [
    "KNOWLEDGE_TYPE_FILE_URL",
    "FileUrlDownload",
    "FileUrlDownloader",
    "FileUrlStoreResult",
    "HttpxFileUrlDownloader",
    "download_file_url",
    "store_file_url_bytes",
]
