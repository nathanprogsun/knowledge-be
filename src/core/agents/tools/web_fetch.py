"""Web page fetch agent tool.

Fetches web page content (returned by ``web_search``) and optionally
summarizes it with an injected chat model. Each URL in a batch is
fetched independently so a partial failure never invalidates the pages
that did succeed.

The fetch path is served by :class:`WebPageFetcher` — an injectable,
httpx-backed fetcher with a body-size cap, per-hop SSRF validation, and
a stable machine-readable error taxonomy (``FetchError`` carries an
error code and a retryable flag, mirroring the upstream fetch client).
Tests inject a stub fetcher (or a fetcher bound to an
``httpx.MockTransport``) so no tool unit test touches the network.
"""

from __future__ import annotations

import asyncio
import re
import ssl
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser
from socket import gaierror
from typing import Protocol

import httpx

from src.ai.llm.types import Chat, ChatOptions, Message
from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.agents.tools.types import ToolResult

# ── Fetch constants (mirror the upstream fetch client) ───────────────

# Overall per-request timeout (seconds).
DEFAULT_FETCH_TIMEOUT_SECONDS: int = 60
# Hard cap on the response body bytes kept (100 KiB).
DEFAULT_MAX_BODY_BYTES: int = 100 * 1024
# Maximum redirect hops before the fetch is rejected.
MAX_REDIRECTS: int = 10

# Stable error codes emitted by :class:`WebPageFetcher`.
ERROR_INVALID_URL = "invalid_url"
ERROR_DNS = "dns_failed"
ERROR_TIMEOUT = "connection_timeout"
ERROR_TLS = "tls_failed"
ERROR_HTTP_403 = "http_403"
ERROR_HTTP_429 = "http_429"
ERROR_HTTP_5XX = "http_5xx"
ERROR_HTTP_STATUS = "http_status"
ERROR_SSRF_REJECTED = "ssrf_rejected"
ERROR_REDIRECT_REJECTED = "redirect_rejected"
ERROR_READ = "read_failed"
ERROR_EMPTY_CONTENT = "empty_content"
ERROR_CONNECTION = "connection_failed"

_TOOL_NAME = "web_fetch"

_DESCRIPTION = """Fetch detailed content from web pages returned by web_search and analyze it with an LLM.

## Usage
- Receive one or more {url: "wN", prompt} combinations; the field name stays url, but its value is the short page ID
- Fetch each page independently and return a structured status for every URL
- Successful pages remain usable when other pages fail

## When to Use
- Use when a search snippet is insufficient or a claim needs full-page verification
- web_search titles, URLs, and snippets remain usable evidence if fetching fails
- Do not repeatedly fetch a non-retryable URL or expand searches after all fetches fail
- When page verification is unavailable, answer from search summaries, disclose that limitation, and lower confidence for dynamic facts"""

# LLM-facing JSON Schema for the tool's ``items`` parameter.
_WEB_FETCH_SCHEMA: JsonValue = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Batch fetch tasks, each containing a short wN web page ID and prompt",
            "items": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Short wN web page ID from web_search results",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt for analyzing the fetched web page content",
                    },
                },
                "required": ["url", "prompt"],
            },
        }
    },
    "required": ["items"],
}

# Elements whose content is dropped during HTML-to-text extraction.
_SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "nav", "footer", "header", "iframe", "noscript", "svg", "img"}
)

_FETCH_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


class FetchError(Exception):
    """Stable, machine-readable fetch failure.

    ``code`` identifies the failure stage/class; ``retryable`` tells the
    caller whether retrying the same URL is worthwhile (non-retryable
    codes are ``invalid_url``, ``ssrf_rejected``, ``redirect_rejected``,
    ``http_403``, ``http_status``, ``tls_failed``, ``empty_content``).
    """

    def __init__(self, *, code: str, retryable: bool, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.message = message


class WebContentFetcher(Protocol):
    """Minimal fetcher seam the tool depends on."""

    async def fetch(self, url: str) -> str:
        """Return the readable text content of ``url`` or raise ``FetchError``."""
        ...


class SSRFGuard(Protocol):
    """Injected URL-safety guard; rejects unsafe (private/reserved) hosts."""

    async def __call__(self, url: str) -> None:
        """Raise ``ValidationError`` when ``url`` is not SSRF-safe."""
        ...


class WebPageFetcher:
    """SSRF-guarded, size-capped HTTP page fetcher backed by httpx.

    Redirects are followed manually (up to ``MAX_REDIRECTS``) so every
    hop passes the SSRF guard before it is requested. The response body
    is streamed and capped at ``max_body_bytes``; the extracted text is
    returned.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: int = DEFAULT_FETCH_TIMEOUT_SECONDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        ssrf_guard: SSRFGuard | None = None,
    ) -> None:
        self._client = client if client is not None else httpx.AsyncClient()
        self._timeout_seconds = max(1, timeout_seconds)
        self._max_body_bytes = max(1024, max_body_bytes)
        self._ssrf_guard = ssrf_guard if ssrf_guard is not None else validate_ssrf_safe_url

    async def fetch(self, url: str) -> str:
        """Fetch ``url`` and return its readable text content."""
        raw = url.strip()
        if not raw:
            raise FetchError(
                code=ERROR_INVALID_URL,
                retryable=False,
                message="url is empty",
            )
        if not _is_http_url(raw):
            raise FetchError(
                code=ERROR_INVALID_URL,
                retryable=False,
                message="invalid URL format",
            )
        try:
            body = await self._get_body(raw)
        except FetchError:
            raise
        except httpx.HTTPError as exc:
            raise _classify_httpx_error(exc) from exc
        text = html_to_text(body.decode("utf-8", errors="replace"))
        if not text.strip():
            raise FetchError(
                code=ERROR_EMPTY_CONTENT,
                retryable=False,
                message="page contains no readable text",
            )
        return text

    async def _get_body(self, url: str) -> bytes:
        current = url
        for _ in range(MAX_REDIRECTS):
            try:
                await self._ssrf_guard(current)
            except ValidationError as exc:
                raise _classify_guard_error(exc) from exc
            timeout = httpx.Timeout(self._timeout_seconds)
            async with self._client.stream(
                "GET",
                current,
                timeout=timeout,
                follow_redirects=False,
                headers=_FETCH_HEADERS,
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError(
                            code=ERROR_REDIRECT_REJECTED,
                            retryable=False,
                            message="redirect with no location header",
                        )
                    current = str(httpx.URL(current).join(location))
                    continue
                if response.status_code != 200:
                    raise _classify_http_status(response.status_code)
                return await _read_body(response, self._max_body_bytes)
        raise FetchError(
            code=ERROR_REDIRECT_REJECTED,
            retryable=False,
            message=f"stopped after {MAX_REDIRECTS} redirects",
        )


# ── HTML text extraction ─────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Collect text outside ``<head>`` and the skipped-tag set."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._head_depth = 0
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "head":
            self._head_depth += 1
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "head" and self._head_depth > 0:
            self._head_depth -= 1
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._head_depth == 0 and self._skip_depth == 0:
            self._parts.append(data)


def html_to_text(raw: str) -> str:
    """Extract readable body text from ``raw`` HTML (or plain text)."""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return _strip_tags(raw)
    lines = [line.strip() for line in "".join(parser._parts).splitlines() if line.strip()]
    return "\n".join(lines)


def _strip_tags(raw: str) -> str:
    without_tags = re.sub(r"<[^>]*>", "", raw)
    lines = [line.strip() for line in without_tags.splitlines() if line.strip()]
    return "\n".join(lines)


# ── URL normalization (matches the fetch tool semantics) ─────────────


def normalize_github_url(source: str) -> str:
    """Rewrite a github.com blob URL to its raw.githubusercontent.com form."""
    if "github.com" in source and "/blob/" in source:
        source = source.replace("github.com", "raw.githubusercontent.com", 1)
        source = source.replace("/blob/", "/", 1)
    return source


def canonical_fetch_url(raw_url: str) -> str:
    """Lowercase the host and drop the fragment for duplicate detection."""
    trimmed = normalize_github_url(raw_url.strip())
    parsed = urllib.parse.urlsplit(trimmed)
    if not parsed.scheme or not parsed.netloc:
        return trimmed
    try:
        port = parsed.port
    except ValueError:
        return trimmed
    hostname = (parsed.hostname or "").lower()
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


# ── Fetch error classification ───────────────────────────────────────


def _classify_guard_error(exc: ValidationError) -> FetchError:
    message = str(exc).lower()
    if "dns resolution failed" in message or "dns lookup failed" in message:
        return FetchError(
            code=ERROR_DNS,
            retryable=True,
            message=f"DNS lookup failed: {exc}",
        )
    return FetchError(
        code=ERROR_SSRF_REJECTED,
        retryable=False,
        message=f"URL rejected: {exc}",
    )


def _classify_http_status(status_code: int) -> FetchError:
    if status_code == 403:
        return FetchError(code=ERROR_HTTP_403, retryable=False, message=f"HTTP {status_code}")
    if status_code == 429:
        return FetchError(code=ERROR_HTTP_429, retryable=True, message=f"HTTP {status_code}")
    if status_code >= 500:
        return FetchError(code=ERROR_HTTP_5XX, retryable=True, message=f"HTTP {status_code}")
    return FetchError(code=ERROR_HTTP_STATUS, retryable=False, message=f"HTTP {status_code}")


def _classify_httpx_error(exc: httpx.HTTPError) -> FetchError:
    if isinstance(exc, httpx.TimeoutException):
        return FetchError(code=ERROR_TIMEOUT, retryable=True, message=f"fetch timed out: {exc}")
    if _chain_has(exc, gaierror):
        return FetchError(code=ERROR_DNS, retryable=True, message=f"DNS lookup failed: {exc}")
    if _chain_has(exc, ssl.SSLError):
        return FetchError(code=ERROR_TLS, retryable=False, message=f"TLS validation failed: {exc}")
    if isinstance(exc, httpx.InvalidURL):
        return FetchError(code=ERROR_INVALID_URL, retryable=False, message=f"invalid URL: {exc}")
    if isinstance(exc, (httpx.DecodingError, httpx.ReadError)):
        return FetchError(code=ERROR_READ, retryable=True, message=f"read failed: {exc}")
    return FetchError(code=ERROR_CONNECTION, retryable=True, message=f"fetch failed: {exc}")


def _chain_has(exc: BaseException, target: type[BaseException]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, target):
            return True
        current = current.__cause__ or current.__context__
    return False


# ── Tool ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WebFetchItem:
    """One batch fetch task: a page id/URL and an analysis prompt."""

    url: str
    prompt: str


@dataclass(frozen=True)
class _ItemResult:
    output: str
    data: JsonObject
    status: str


class WebFetchTool:
    """Agent tool that fetches web pages and optionally summarizes them."""

    name = _TOOL_NAME

    def __init__(
        self,
        *,
        fetcher: WebContentFetcher | None = None,
        chat_model: Chat | None = None,
    ) -> None:
        self._fetcher = fetcher if fetcher is not None else WebPageFetcher()
        self._chat_model = chat_model
        self.description = _DESCRIPTION
        self.parameters_schema: JsonValue = _WEB_FETCH_SCHEMA

    async def execute(self, args: JsonObject) -> ToolResult:
        """Fetch each requested page; keep successful items when some fail."""
        items = _extract_items(args)
        if not items:
            return ToolResult(success=False, error="missing required parameter: items")

        results: list[_ItemResult | None] = [None] * len(items)
        seen: set[str] = set()
        pending: list[tuple[int, WebFetchItem]] = []
        for index, item in enumerate(items):
            canonical = canonical_fetch_url(item.url)
            if canonical in seen:
                results[index] = _duplicate_item_result(item)
                continue
            seen.add(canonical)
            pending.append((index, item))
        if pending:
            fetched = await asyncio.gather(*(self._fetch_item(item) for _index, item in pending))
            for (index, _item), item_result in zip(pending, fetched, strict=True):
                results[index] = item_result
        return _build_tool_result(results)

    async def _fetch_item(self, item: WebFetchItem) -> _ItemResult:
        display_url = item.url.strip()
        if not item.prompt.strip():
            return _failed_item_result(
                display_url, False, "invalid_arguments", "prompt is required"
            )

        fetch_url = normalize_github_url(display_url)
        try:
            content = await self._fetcher.fetch(fetch_url)
        except FetchError as exc:
            return _failed_item_result(display_url, exc.retryable, exc.code, exc.message)
        except Exception as exc:
            return _failed_item_result(display_url, True, ERROR_CONNECTION, str(exc))

        summary, summary_error = await self._summarize(item, content)
        data: JsonObject = {
            "url": display_url,
            "status": "success",
            "retryable": False,
            "prompt": item.prompt,
            "raw_content": content,
            "content_length": len(content),
            "evidence_type": "fetched_page",
            "summary_status": "not_requested",
        }
        if summary_error is not None:
            data["summary_status"] = "failed"
            data["summary_error_code"] = "summary_failed"
            data["summary_error_message"] = summary_error
        elif summary:
            data["summary_status"] = "success"
            data["summary"] = summary
        return _ItemResult(
            output=_build_item_output(item, content, summary, summary_error),
            data=data,
            status="success",
        )

    async def _summarize(self, item: WebFetchItem, content: str) -> tuple[str, str | None]:
        chat_model = self._chat_model
        if chat_model is None:
            return "", "chat model not available for web_fetch summary"
        messages = [
            Message(
                role="system",
                content=(
                    "Answer the request from the supplied web page text. "
                    "Never fabricate information that is absent from the page."
                ),
            ),
            Message(
                role="user",
                content=f"User request:\n{item.prompt}\n\nWeb page content:\n{content}",
            ),
        ]
        opts = ChatOptions(temperature=0.3, max_tokens=1024)
        try:
            response = await chat_model.chat(messages, opts)
        except Exception as exc:
            return "", str(exc)
        return response.content.strip(), None


# ── Tool helpers ─────────────────────────────────────────────────────


def _extract_items(args: JsonObject) -> list[WebFetchItem]:
    raw_items = args.get("items")
    if not isinstance(raw_items, list):
        return []
    items: list[WebFetchItem] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        prompt = entry.get("prompt")
        items.append(
            WebFetchItem(
                url=url if isinstance(url, str) else "",
                prompt=prompt if isinstance(prompt, str) else "",
            )
        )
    return items


def _is_http_url(raw: str) -> bool:
    parsed = urllib.parse.urlsplit(raw)
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.hostname)


async def _read_body(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        remaining = max_bytes - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _failed_item_result(raw_url: str, retryable: bool, code: str, message: str) -> _ItemResult:
    return _ItemResult(
        output=(
            f"URL: {raw_url}\nStatus: failed\nRetryable: {retryable}\n"
            f"Error code: {code}\nError: {message}\n"
        ),
        data={
            "url": raw_url,
            "status": "failed",
            "retryable": retryable,
            "error_code": code,
            "error_message": message,
        },
        status="failed",
    )


def _duplicate_item_result(item: WebFetchItem) -> _ItemResult:
    message = "duplicate URL skipped in this batch"
    return _ItemResult(
        output=f"URL: {item.url}\nStatus: skipped\nRetryable: false\nReason: {message}\n",
        data={
            "url": item.url,
            "status": "skipped",
            "retryable": False,
            "error_code": "duplicate_url",
            "error_message": message,
        },
        status="skipped",
    )


def _build_item_output(
    item: WebFetchItem,
    content: str,
    summary: str,
    summary_error: str | None,
) -> str:
    lines = [f"URL: {item.url}", "Status: success", f"Prompt: {item.prompt}"]
    if summary:
        lines.append("Summary:")
        lines.append(summary)
        return "\n".join(lines) + "\n"
    if summary_error is not None:
        lines.append(
            f"Summary status: failed ({summary_error}); fetched page content remains usable."
        )
    lines.append("Content Preview:")
    lines.append(content)
    return "\n".join(lines) + "\n"


def _build_tool_result(results: list[_ItemResult | None]) -> ToolResult:
    parts: list[str] = ["=== Web Fetch Results ===\n\n"]
    aggregated: list[JsonValue] = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    for index, result in enumerate(results):
        if result is None:
            result = _failed_item_result(
                "", False, "internal_error", "fetch item returned no result"
            )
        parts.append(f"#{index + 1}:\n{result.output}\n")
        aggregated.append(result.data)
        if result.status == "success":
            success_count += 1
        elif result.status == "failed":
            failed_count += 1
        else:
            skipped_count += 1
    all_failed = success_count == 0 and failed_count > 0
    parts.append("=== Next Steps ===\n")
    if all_failed:
        parts.append(
            "- All page fetches failed. Stop expanding web searches and answer from existing "
            "web_search titles, URLs, and snippets.\n"
        )
        parts.append(
            "- Explicitly state that page content was not verified. Treat prices, inventory, "
            "and other dynamic facts as uncertain.\n"
        )
    elif failed_count > 0:
        parts.append(
            "- Use successful page content together with existing search snippets; failed URLs "
            "do not invalidate successful evidence.\n"
        )
        parts.append(
            "- Do not retry non-retryable failures. If evidence is sufficient, answer now.\n"
        )
    else:
        parts.append("- Synthesize the fetched evidence and answer when it is sufficient.\n")

    tool_result = ToolResult(
        success=success_count > 0,
        output="".join(parts),
        data={
            "results": aggregated,
            "count": len(aggregated),
            "successful_count": success_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "all_failed": all_failed,
            "display_type": "web_fetch_results",
        },
    )
    if all_failed:
        return tool_result.model_copy(update={"error": "all page fetches failed"})
    return tool_result


__all__ = [
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BODY_BYTES",
    "MAX_REDIRECTS",
    "FetchError",
    "SSRFGuard",
    "WebContentFetcher",
    "WebFetchItem",
    "WebFetchTool",
    "WebPageFetcher",
    "canonical_fetch_url",
    "html_to_text",
    "normalize_github_url",
]
