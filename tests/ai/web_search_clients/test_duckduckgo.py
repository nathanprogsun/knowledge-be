"""Unit tests for the DuckDuckGo web-search client.

The httpx transport is mocked (``httpx.MockTransport``) so the tests
exercise request construction, response parsing, the HTML-first /
API-fallback flow and error normalization without contacting the live
service.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.duckduckgo import (
    DuckDuckGoProvider,
    build_duckduckgo_client,
    clean_ddg_url,
    extract_title,
)
from src.common.exception import ExternalServiceError, ValidationError

_HTML_URL = "https://html.duckduckgo.com/html/"
_API_URL = "https://api.duckduckgo.com/"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _duckduckgo(handler: Callable[[httpx.Request], httpx.Response]) -> DuckDuckGoProvider:
    return DuckDuckGoProvider(client=_client(handler))


def _hit_html(title: str, url: str, snippet: str) -> str:
    return (
        '<div class="web-result">'
        f'<a class="result__a" href="{url}">{title}</a>'
        f'<a class="result__snippet">{snippet}</a>'
        "</div>"
    )


# ── HTML path ─────────────────────────────────────────────────────────


def test_html_search_returns_hits_and_builds_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = (
            _hit_html("Title 1", "https://example.com/1", "Snippet 1")
            + _hit_html("Title 2", "https://example.com/2", "Snippet 2")
        )
        return httpx.Response(200, text=body)

    results = _duckduckgo(handler).search("hello", 5, False)
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert str(request.url).startswith(_HTML_URL + "?")
    assert dict(request.url.params) == {"q": "hello", "kl": "cn-zh"}
    assert request.headers["User-Agent"].startswith("Mozilla/5.0")
    assert results == [
        {"title": "Title 1", "url": "https://example.com/1", "snippet": "Snippet 1", "source": "duckduckgo"},
        {"title": "Title 2", "url": "https://example.com/2", "snippet": "Snippet 2", "source": "duckduckgo"},
    ]


def test_html_search_accepts_status_202() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, text=_hit_html("T", "https://e/1", "s"))

    results = _duckduckgo(handler).search("hello", 3, False)
    assert len(results) == 1
    assert results[0]["title"] == "T"


def test_html_search_cleans_redirect_url() -> None:
    real_url = "https%3A%2F%2Fexample.com%2F1"
    redirect = "//duckduckgo.com/l/?uddg=" + real_url + "&rut=foo"
    body = _hit_html("T", redirect, "s")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    results = _duckduckgo(handler).search("hello", 3, False)
    assert results[0]["url"] == "https://example.com/1"


def test_html_search_returns_empty_when_no_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body></body></html>")

    # Empty HTML result falls back to the API; both endpoints are hit
    # in this scenario.
    api_calls: list[httpx.Request] = []

    def handler_with_api(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html></html>")
        api_calls.append(request)
        return httpx.Response(200, json={"AbstractText": "", "AbstractURL": "", "RelatedTopics": []})

    provider = _duckduckgo(handler_with_api)
    results = provider.search("hello", 3, False)
    assert results == []
    assert len(api_calls) == 1


def test_html_search_respects_max_results_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(
            _hit_html(f"T{i}", f"https://e/{i}", f"s{i}") for i in range(10)
        )
        return httpx.Response(200, text=body)

    results = _duckduckgo(handler).search("hello", 3, False)
    assert len(results) == 3


# ── API fallback path ─────────────────────────────────────────────────


def test_api_fallback_used_when_html_returns_no_hits() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html><body></body></html>")
        return httpx.Response(
            200,
            json={
                "AbstractText": "abstract",
                "AbstractURL": "https://example.com/abstract",
                "Heading": "Heading",
                "RelatedTopics": [],
            },
        )

    provider = _duckduckgo(handler)
    results = provider.search("hello", 3, False)
    assert len(requests) == 2
    assert str(requests[1].url).startswith(_API_URL + "?")
    assert dict(requests[1].url.params) == {
        "q": "hello",
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    assert results == [
        {
            "title": "Heading",
            "url": "https://example.com/abstract",
            "snippet": "abstract",
            "source": "duckduckgo",
        }
    ]


def test_api_search_includes_related_topics_and_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html></html>",
            request=request,
        ) if str(request.url).startswith(_HTML_URL) else httpx.Response(
            200,
            json={
                "AbstractText": "",
                "AbstractURL": "",
                "RelatedTopics": [
                    {"Text": "R-Title-1\nrest", "FirstURL": "https://e/r/1"},
                    {"Text": "", "FirstURL": "https://e/skip"},
                    {"Text": "only-text"},
                ],
                "Results": [
                    {"Text": "Result-Text-1", "FirstURL": "https://e/res/1"},
                ],
            },
        )

    results = _duckduckgo(handler).search("hello", 10, False)
    assert [r["title"] for r in results] == ["R-Title-1", "Result-Text-1"]
    assert results[0]["snippet"] == "R-Title-1\nrest"
    assert results[0]["url"] == "https://e/r/1"
    assert results[1]["url"] == "https://e/res/1"


def test_api_search_uses_first_line_as_title_and_caps_at_100() -> None:
    long_line = "x" * 150

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(
            200,
            json={
                "AbstractText": "",
                "AbstractURL": "",
                "RelatedTopics": [
                    {"Text": f"{long_line}\nrest", "FirstURL": "https://e/1"},
                ],
            },
        )

    results = _duckduckgo(handler).search("hello", 5, False)
    assert results[0]["title"].endswith("...")
    assert len(results[0]["title"]) == 103  # 100 chars + ellipsis


# ── Error handling ────────────────────────────────────────────────────


def test_html_network_error_falls_back_to_api() -> None:
    api_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal api_called
        if str(request.url).startswith(_HTML_URL):
            raise httpx.ConnectError("boom")
        api_called = True
        return httpx.Response(
            200,
            json={
                "AbstractText": "abs",
                "AbstractURL": "https://e/abs",
                "RelatedTopics": [],
            },
        )

    results = _duckduckgo(handler).search("hello", 3, False)
    assert api_called is True
    assert len(results) == 1


def test_html_non_200_falls_back_to_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(500, text="server error")
        return httpx.Response(
            200,
            json={
                "AbstractText": "abs",
                "AbstractURL": "https://e/abs",
                "RelatedTopics": [],
            },
        )

    results = _duckduckgo(handler).search("hello", 3, False)
    assert len(results) == 1


def test_both_endpoints_failing_raises_html_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(500, text="html boom")
        return httpx.Response(500, text="api boom")

    with pytest.raises(ExternalServiceError) as excinfo:
        _duckduckgo(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "HTML search failed" in excinfo.value.message


def test_api_non_200_after_empty_html_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(503, text="down")

    with pytest.raises(ExternalServiceError) as excinfo:
        _duckduckgo(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "API search failed" in excinfo.value.message


def test_api_unparseable_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(200, text="not json")

    with pytest.raises(ExternalServiceError) as excinfo:
        _duckduckgo(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"


def test_api_non_object_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(_HTML_URL):
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(200, json=["array", "instead", "of", "object"])

    with pytest.raises(ExternalServiceError) as excinfo:
        _duckduckgo(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"


def test_search_empty_query_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(ValidationError) as excinfo:
        _duckduckgo(handler).search("   ", 3, False)
    assert excinfo.value.code == "web_search_provider.query_required"


def test_search_uses_default_count_when_max_results_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="".join(_hit_html(f"T{i}", f"https://e/{i}", "s") for i in range(10)),
        )

    results = _duckduckgo(handler).search("hello", 0, False)
    assert len(results) == 5  # default = 5


# ── URL cleaning helpers ──────────────────────────────────────────────


def test_clean_ddg_url_strips_uddg_protocol_relative() -> None:
    url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath&rut=xyz"
    assert clean_ddg_url(url) == "https://example.com/path"


def test_clean_ddg_url_strips_uddg_absolute() -> None:
    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath"
    assert clean_ddg_url(url) == "https://example.com/path"


def test_clean_ddg_url_returns_unchanged_when_not_ddg() -> None:
    assert clean_ddg_url("https://example.com/direct") == "https://example.com/direct"


def test_clean_ddg_url_returns_unchanged_when_no_rut_marker() -> None:
    url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath"
    assert clean_ddg_url(url) == url


def test_extract_title_truncates_long_titles() -> None:
    long_line = "x" * 200
    assert extract_title(long_line).endswith("...")
    assert len(extract_title(long_line)) == 103


def test_extract_title_takes_first_line() -> None:
    assert extract_title("first line\nsecond line") == "first line"
    assert extract_title("  spaced  \n  other  ") == "spaced"


# ── Builder ──────────────────────────────────────────────────────────


def test_build_duckduckgo_client() -> None:
    provider = build_duckduckgo_client({})
    assert isinstance(provider, DuckDuckGoProvider)


def test_build_rejects_invalid_proxy() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_duckduckgo_client({"proxy_url": "http://127.0.0.1:3128"})
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_build_accepts_public_proxy() -> None:
    provider = build_duckduckgo_client({"proxy_url": "https://proxy.example.com:3128"})
    assert isinstance(provider, DuckDuckGoProvider)
