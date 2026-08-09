"""Unit + integration tests for the web_search and web_fetch agent tools.

Unit tests drive each tool through injected seams — a stub search
service, a stub fetcher, a stub chat model, and an httpx
``MockTransport``-bound ``WebPageFetcher`` — so no test touches the
network. They cover validation, error classification, duplicate
handling, output formatting, and the optional LLM summary.

Integration tests run against the real applied schema: they seed a
``tenants`` row (id is DB-assigned) and a ``web_search_providers`` row,
then execute the tool through the real ``WebSearchSearchService`` with a
stub client registry. They require a reachable database — run with
``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from typing import cast

import httpx
import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.agents.tools.types import ToolResult
from src.core.agents.tools.web_fetch import (
    ERROR_HTTP_5XX,
    ERROR_HTTP_403,
    ERROR_HTTP_429,
    ERROR_INVALID_URL,
    ERROR_SSRF_REJECTED,
    ERROR_TIMEOUT,
    FetchError,
    WebFetchTool,
    WebPageFetcher,
    canonical_fetch_url,
    html_to_text,
    normalize_github_url,
)
from src.core.agents.tools.web_search import WebSearchTool
from src.core.infra.web_search.provider_service import WebSearchClient
from src.core.infra.web_search.search_service import SearchResult, WebSearchSearchService
from src.db.dao.tenants_repository import TenantRepository
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository
from src.db.models.infra.web_search_provider import WebSearchProvider
from src.db.models.tenants.tenants import Tenant
from src.settings import get_settings, reset_settings_cache

_FAKER_SEED_MAX = 100_000_000


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Protocol doubles (non-network collaborators) ─────────────────────


class FakeSearchService:
    """Stub of the tool's ``WebSearchService`` seam."""

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._results = list(results or [])
        self._error = error
        self.calls: list[tuple[int, str, str, int]] = []

    async def search(
        self,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        self.calls.append((tenant_id, provider_id, query, max_results))
        if self._error is not None:
            raise self._error
        return list(self._results)


class FakeFetcher:
    """Stub of the tool's ``WebContentFetcher`` seam."""

    def __init__(self, content: str = "page text", error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.urls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.urls.append(url)
        if self._error is not None:
            raise self._error
        return self._content


class FakeChat:
    """Stub chat model returning a canned summary."""

    def __init__(self, response: str = "canned summary", error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return ChatResponse(content=self._response)


class FakeClient:
    """Stub ``WebSearchClient`` returning canned hits."""

    def __init__(self, hits: list[dict[str, str]]) -> None:
        self._hits = hits

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[dict[str, str]]:
        return list(self._hits)


class FakeRegistry:
    """Stub ``WebSearchClientRegistry`` for the real search-service path."""

    def __init__(self, hits: list[dict[str, str]]) -> None:
        self._hits = hits

    def create_provider(self, provider_type: str, params: JsonObject) -> WebSearchClient:
        return FakeClient(self._hits)


# ── Shared fixtures / helpers ────────────────────────────────────────


async def _noop_guard(url: str) -> None:
    return None


def _result(
    *,
    title: str = "Alpha",
    url: str = "https://example.com/a",
    snippet: str = "snippet a",
    content: str = "",
    source: str = "tavily",
    published_at: datetime | None = None,
) -> SearchResult:
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        content=content,
        source=source,
        published_at=published_at,
    )


def _fetcher_client(handler: httpx.MockTransportHandler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

# ═══════════════════════════════════════════════════════════════════════
# WebSearchTool unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchTool:
    async def test_missing_query_fails(self) -> None:
        tool = WebSearchTool(search_service=FakeSearchService())

        result = await tool.execute({}, tenant_id=1)

        assert result.success is False
        assert result.error == "query parameter is required"

    async def test_blank_query_fails(self) -> None:
        tool = WebSearchTool(search_service=FakeSearchService())

        result = await tool.execute({"query": ""}, tenant_id=1)

        assert result.success is False
        assert result.error == "query parameter is required"

    async def test_missing_tenant_fails(self) -> None:
        tool = WebSearchTool(search_service=FakeSearchService())

        result = await tool.execute({"query": "hello"}, tenant_id=0)

        assert result.success is False
        assert result.error == "workspace ID not found in context"

    async def test_happy_path_formats_results(self) -> None:
        service = FakeSearchService(
            results=[
                _result(title="Alpha", url="https://example.com/a", snippet="snip a", content="body a"),
                _result(title="Beta", url="https://example.com/b", snippet="", source="bing"),
            ]
        )
        tool = WebSearchTool(search_service=service, provider_id="prov-1", max_results=3)

        result = await tool.execute({"query": "hello"}, tenant_id=7)

        assert result.success is True
        assert "=== Web Search Results ===" in result.output
        assert "Query: hello" in result.output
        assert "Found 2 result(s)" in result.output
        assert "Result #1:" in result.output
        assert "Title: Alpha" in result.output
        assert "URL: https://example.com/a" in result.output
        assert "Snippet: snip a" in result.output
        assert "Content: body a" in result.output
        assert "=== Next Steps ===" in result.output
        assert "Result #2:" in result.output
        # Only the first result carries a snippet line (the second has none).
        assert result.output.count("Snippet:") == 1

        assert service.calls == [(7, "prov-1", "hello", 3)]
        data = result.data
        assert data is not None
        assert data["count"] == 2
        assert data["query"] == "hello"
        assert data["display_type"] == "web_search_results"
        items = data["results"]
        assert isinstance(items, list)
        assert len(items) == 2
        first = items[0]
        assert isinstance(first, dict)
        assert first["title"] == "Alpha"
        assert first["evidence_type"] == "search_summary"
        assert first["page_verified"] is False
        assert first["result_index"] == 1

    async def test_content_truncated_at_500_chars(self) -> None:
        long_content = "x" * 600
        tool = WebSearchTool(search_service=FakeSearchService(results=[_result(content=long_content)]))

        result = await tool.execute({"query": "hello"}, tenant_id=7)

        assert "x" * 500 + "..." in result.output
        # The structured payload keeps the full content; only the preview is truncated.
        data = result.data
        assert data is not None
        items = data["results"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        assert item["content"] == long_content

    async def test_empty_results(self) -> None:
        tool = WebSearchTool(search_service=FakeSearchService(results=[]))

        result = await tool.execute({"query": "nothing"}, tenant_id=7)

        assert result.success is True
        assert result.output == "No web search results found for query: nothing"
        data = result.data
        assert data is not None
        assert data["count"] == 0
        assert data["results"] == []

    async def test_published_at_rendered_as_rfc3339(self) -> None:
        service = FakeSearchService(
            results=[_result(published_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))]
        )
        tool = WebSearchTool(search_service=service)

        result = await tool.execute({"query": "hello"}, tenant_id=7)

        assert "Published: 2026-01-02T03:04:05Z" in result.output
        data = result.data
        assert data is not None
        items = data["results"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        assert item["published_at"] == "2026-01-02T03:04:05Z"

    async def test_service_error_surfaces_as_failed_result(self) -> None:
        service = FakeSearchService(
            error=ValidationError(
                code="web_search_provider.not_found",
                message="web search provider nope not found",
            )
        )
        tool = WebSearchTool(search_service=service, provider_id="nope")

        result = await tool.execute({"query": "hello"}, tenant_id=7)

        assert result.success is False
        assert result.error == "web search failed: web search provider nope not found"


# ═══════════════════════════════════════════════════════════════════════
# WebFetchTool unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebFetchTool:
    async def test_missing_items_fails(self) -> None:
        tool = WebFetchTool(fetcher=FakeFetcher())

        result = await tool.execute({})

        assert result.success is False
        assert result.error == "missing required parameter: items"

    async def test_single_success(self) -> None:
        fetcher = FakeFetcher(content="page body text")
        tool = WebFetchTool(fetcher=fetcher)

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        assert result.success is True
        assert "=== Web Fetch Results ===" in result.output
        assert "#1:" in result.output
        assert "URL: https://example.com/a" in result.output
        assert "Status: success" in result.output
        assert "Content Preview:" in result.output
        assert fetcher.urls == ["https://example.com/a"]
        data = result.data
        assert data is not None
        assert data["successful_count"] == 1
        assert data["failed_count"] == 0
        assert data["all_failed"] is False
        assert data["display_type"] == "web_fetch_results"
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["status"] == "success"
        assert first["evidence_type"] == "fetched_page"
        assert first["content_length"] == len("page body text")
        # No chat model injected: the summary is marked failed, content stays usable.
        assert first["summary_status"] == "failed"

    async def test_empty_prompt_fails_item(self) -> None:
        tool = WebFetchTool(fetcher=FakeFetcher())

        result = await tool.execute({"items": [{"url": "https://example.com/a", "prompt": "  "}]})

        assert result.success is False
        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["status"] == "failed"
        assert first["error_code"] == "invalid_arguments"
        assert first["error_message"] == "prompt is required"
        assert first["retryable"] is False

    async def test_fetch_error_maps_to_failed_item(self) -> None:
        fetcher = FakeFetcher(
            error=FetchError(code=ERROR_HTTP_403, retryable=False, message="HTTP 403 Forbidden")
        )
        tool = WebFetchTool(fetcher=fetcher)

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        assert result.success is False
        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["status"] == "failed"
        assert first["error_code"] == ERROR_HTTP_403
        assert first["retryable"] is False
        assert "Error code: http_403" in result.output

    async def test_unexpected_fetch_error_classifies_as_connection_failed(self) -> None:
        fetcher = FakeFetcher(error=RuntimeError("boom"))
        tool = WebFetchTool(fetcher=fetcher)

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["status"] == "failed"
        assert first["error_code"] == "connection_failed"
        assert first["retryable"] is True

    async def test_all_failures_mark_tool_failed(self) -> None:
        tool = WebFetchTool(
            fetcher=FakeFetcher(
                error=FetchError(code=ERROR_TIMEOUT, retryable=True, message="timed out")
            )
        )

        result = await tool.execute(
            {
                "items": [
                    {"url": "https://ok.example.com/a", "prompt": "summarize"},
                    {"url": "https://ok.example.com/b", "prompt": "summarize"},
                ]
            }
        )

        assert result.success is False
        assert result.error == "all page fetches failed"
        data = result.data
        assert data is not None
        assert data["successful_count"] == 0
        assert data["failed_count"] == 2
        assert data["all_failed"] is True

    async def test_mixed_success_and_failure(self) -> None:
        class _MixedFetcher:
            async def fetch(self, url: str) -> str:
                if "ok" in url:
                    return "content ok"
                raise FetchError(code=ERROR_HTTP_429, retryable=True, message="rate limited")

        tool = WebFetchTool(fetcher=_MixedFetcher())

        result = await tool.execute(
            {
                "items": [
                    {"url": "https://ok.example.com/a", "prompt": "summarize"},
                    {"url": "https://bad.example.com/b", "prompt": "summarize"},
                ]
            }
        )

        assert result.success is True
        assert result.error is None
        data = result.data
        assert data is not None
        assert data["successful_count"] == 1
        assert data["failed_count"] == 1
        assert data["skipped_count"] == 0
        assert data["all_failed"] is False
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        second = results[1]
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        assert first["status"] == "success"
        assert second["status"] == "failed"

    async def test_duplicate_url_skipped(self) -> None:
        tool = WebFetchTool(fetcher=FakeFetcher(content="content"))

        result = await tool.execute(
            {
                "items": [
                    {"url": "https://example.com/a", "prompt": "summarize"},
                    {"url": "https://example.com/a", "prompt": "summarize"},
                ]
            }
        )

        assert result.success is True
        data = result.data
        assert data is not None
        assert data["successful_count"] == 1
        assert data["skipped_count"] == 1
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["status"] == "success"
        skipped = results[1]
        assert isinstance(skipped, dict)
        assert skipped["status"] == "skipped"
        assert skipped["error_code"] == "duplicate_url"
        assert "Reason: duplicate URL skipped in this batch" in result.output

    async def test_duplicate_detection_ignores_case_and_fragment(self) -> None:
        tool = WebFetchTool(fetcher=FakeFetcher(content="content"))

        result = await tool.execute(
            {
                "items": [
                    {"url": "https://Example.com/a#frag", "prompt": "summarize"},
                    {"url": "https://example.com/a", "prompt": "summarize"},
                ]
            }
        )

        data = result.data
        assert data is not None
        assert data["successful_count"] == 1
        assert data["skipped_count"] == 1

    async def test_github_blob_url_normalized_before_fetch(self) -> None:
        fetcher = FakeFetcher(content="readme")
        tool = WebFetchTool(fetcher=fetcher)

        result = await tool.execute(
            {
                "items": [
                    {
                        "url": "https://github.com/org/repo/blob/main/readme.md",
                        "prompt": "summarize",
                    }
                ]
            }
        )

        assert result.success is True
        assert fetcher.urls == ["https://raw.githubusercontent.com/org/repo/main/readme.md"]

    async def test_summary_generated_with_chat_model(self) -> None:
        chat = FakeChat(response="short summary")
        tool = WebFetchTool(fetcher=FakeFetcher(content="content"), chat_model=cast("Chat", chat))

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        assert chat.calls == [
            [
                Message(
                    role="system",
                    content=(
                        "Answer the request from the supplied web page text. "
                        "Never fabricate information that is absent from the page."
                    ),
                ),
                Message(
                    role="user",
                    content="User request:\nsummarize\n\nWeb page content:\ncontent",
                ),
            ]
        ]
        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["summary_status"] == "success"
        assert first["summary"] == "short summary"
        assert "Summary:" in result.output
        assert "Content Preview:" not in result.output

    async def test_summary_marked_failed_without_chat_model(self) -> None:
        tool = WebFetchTool(fetcher=FakeFetcher(content="content"))

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        assert result.success is True
        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["summary_status"] == "failed"
        assert first["summary_error_code"] == "summary_failed"
        assert "chat model not available for web_fetch summary" in first["summary_error_message"]
        assert "fetched page content remains usable" in result.output
        assert "Content Preview:" in result.output

    async def test_summary_error_from_chat_model(self) -> None:
        chat = FakeChat(error=RuntimeError("provider down"))
        tool = WebFetchTool(fetcher=FakeFetcher(content="content"), chat_model=cast("Chat", chat))

        result = await tool.execute(
            {"items": [{"url": "https://example.com/a", "prompt": "summarize"}]}
        )

        data = result.data
        assert data is not None
        results = data["results"]
        assert isinstance(results, list)
        first = results[0]
        assert isinstance(first, dict)
        assert first["summary_status"] == "failed"
        assert first["summary_error_code"] == "summary_failed"
        assert "provider down" in first["summary_error_message"]


# ═══════════════════════════════════════════════════════════════════════
# WebPageFetcher unit tests (httpx.MockTransport, no network)
# ═══════════════════════════════════════════════════════════════════════


class TestWebPageFetcher:
    async def test_extracts_body_text_and_skips_noise_tags(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text=(
                    "<html><head><title>head title</title></head>"
                    "<body><nav>menu</nav><p>Hello world</p>"
                    "<script>var x = 1;</script><footer>foot</footer></body></html>"
                ),
            )

        fetcher = WebPageFetcher(client=_fetcher_client(_handler), ssrf_guard=_noop_guard)

        text = await fetcher.fetch("https://example.com/page")

        assert text == "Hello world"
        assert "menu" not in text
        assert "head title" not in text
        assert "var x" not in text

    async def test_plain_text_passes_through(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="just text, no html")

        fetcher = WebPageFetcher(client=_fetcher_client(_handler), ssrf_guard=_noop_guard)

        text = await fetcher.fetch("https://example.com/plain")

        assert text == "just text, no html"

    async def test_body_capped_at_max_bytes(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="a" * 200_000)

        fetcher = WebPageFetcher(
            client=_fetcher_client(_handler),
            ssrf_guard=_noop_guard,
            max_body_bytes=1024,
        )

        text = await fetcher.fetch("https://example.com/big")

        assert len(text) <= 1024

    async def test_empty_page_raises_empty_content(self) -> None:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body><script>var x=1;</script></body></html>")

        fetcher = WebPageFetcher(client=_fetcher_client(_handler), ssrf_guard=_noop_guard)

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/empty")

        assert exc_info.value.code == "empty_content"
        assert exc_info.value.retryable is False

    async def test_http_403_classified_non_retryable(self) -> None:
        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(403)),
            ssrf_guard=_noop_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/x")

        assert exc_info.value.code == ERROR_HTTP_403
        assert exc_info.value.retryable is False

    async def test_http_429_classified_retryable(self) -> None:
        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(429)),
            ssrf_guard=_noop_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/x")

        assert exc_info.value.code == ERROR_HTTP_429
        assert exc_info.value.retryable is True

    async def test_http_5xx_classified_retryable(self) -> None:
        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(502)),
            ssrf_guard=_noop_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/x")

        assert exc_info.value.code == ERROR_HTTP_5XX
        assert exc_info.value.retryable is True

    async def test_invalid_scheme_rejected(self) -> None:
        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(200)),
            ssrf_guard=_noop_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("ftp://example.com/file")

        assert exc_info.value.code == ERROR_INVALID_URL
        assert exc_info.value.retryable is False

    async def test_empty_url_rejected(self) -> None:
        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(200)),
            ssrf_guard=_noop_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("   ")

        assert exc_info.value.code == ERROR_INVALID_URL

    async def test_redirects_followed_and_each_hop_guarded(self) -> None:
        guarded: list[str] = []

        async def _recording_guard(url: str) -> None:
            guarded.append(url)

        def _handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "example.com":
                return httpx.Response(302, headers={"location": "https://cdn.example.com/page"})
            return httpx.Response(200, text="final content")

        fetcher = WebPageFetcher(client=_fetcher_client(_handler), ssrf_guard=_recording_guard)

        text = await fetcher.fetch("https://example.com/start")

        assert text == "final content"
        assert guarded == ["https://example.com/start", "https://cdn.example.com/page"]

    async def test_ssrf_guard_rejection_maps_to_ssrf_rejected(self) -> None:
        async def _blocking_guard(url: str) -> None:
            raise ValidationError(
                "hostname example.com resolves to restricted IP 10.0.0.1",
                code="oidc.ssrf_blocked",
            )

        fetcher = WebPageFetcher(
            client=_fetcher_client(lambda req: httpx.Response(200)),
            ssrf_guard=_blocking_guard,
        )

        with pytest.raises(FetchError) as exc_info:
            await fetcher.fetch("https://example.com/x")

        assert exc_info.value.code == ERROR_SSRF_REJECTED
        assert exc_info.value.retryable is False


# ═══════════════════════════════════════════════════════════════════════
# URL-normalization unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestUrlNormalization:
    def test_github_blob_rewritten_to_raw(self) -> None:
        source = "https://github.com/org/repo/blob/main/readme.md"
        assert normalize_github_url(source) == "https://raw.githubusercontent.com/org/repo/main/readme.md"

    def test_non_github_url_unchanged(self) -> None:
        assert normalize_github_url("https://example.com/a") == "https://example.com/a"

    def test_canonical_drops_fragment_and_lowercases_host(self) -> None:
        raw = "https://Example.com/a#section"
        assert canonical_fetch_url(raw) == "https://example.com/a"

    def test_canonical_keeps_port(self) -> None:
        assert canonical_fetch_url("https://example.com:8080/a#f") == "https://example.com:8080/a"

    def test_canonical_invalid_url_returns_trimmed(self) -> None:
        assert canonical_fetch_url("not a url") == "not a url"


# ═══════════════════════════════════════════════════════════════════════
# HTML extraction unit tests
# ═══════════════════════════════════════════════════════════════════════


class TestHtmlToText:
    def test_whitespace_normalized(self) -> None:
        raw = "<html><body><p>line one</p>\n\n<p>line two</p></body></html>"
        assert html_to_text(raw) == "line one\nline two"

    def test_skip_tags_removed(self) -> None:
        raw = "<body><script>js()</script><style>.a{}</style><p>keep</p></body>"
        assert html_to_text(raw) == "keep"


# ═══════════════════════════════════════════════════════════════════════
# Integration (real applied schema)
# ═══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def test_integration_web_search_tool_runs_through_real_provider_row(
    session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    tenant = await TenantRepository(session).insert(
        Tenant(
            name=f"workspace-{uuid.uuid4().hex[:8]}",
            description="per-test workspace",
            status="active",
            business="",
            retriever_engines={"engines": []},
            created_at=now,
            updated_at=now,
        )
    )
    provider_id = f"ws-{uuid.uuid4().hex[:8]}"
    await WebSearchProviderRepository(session).insert(
        WebSearchProvider(
            id=provider_id,
            tenant_id=tenant.id,
            name="tavily",
            provider="tavily",
            description=None,
            parameters={"api_key": "test-key"},
            is_default=True,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
    )
    await session.commit()

    registry = FakeRegistry(
        hits=[{"title": "One", "url": "https://example.com/1", "snippet": "snip", "content": "body"}]
    )
    service = WebSearchSearchService(
        provider_repo=WebSearchProviderRepository(session),
        registry=registry,
        timeout_seconds=5,
    )
    tool = WebSearchTool(search_service=service, max_results=3, provider_id=provider_id)

    result: ToolResult = await tool.execute({"query": "hello"}, tenant_id=tenant.id)

    assert result.success is True
    assert "Query: hello" in result.output
    data = result.data
    assert data is not None
    assert data["count"] == 1
    items = data["results"]
    assert isinstance(items, list)
    first = items[0]
    assert isinstance(first, dict)
    assert first["title"] == "One"
    assert first["source"] == "tavily"
    assert first["evidence_type"] == "search_summary"
