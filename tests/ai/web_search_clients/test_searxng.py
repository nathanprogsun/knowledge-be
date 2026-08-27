"""Unit tests for the SearXNG metasearch client.

The instance URL is tenant-supplied, so the SSRF guard is exercised
directly; the HTTP transport is mocked with ``httpx.MockTransport``.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from pytest import MonkeyPatch

from src.ai.web_search_clients.searxng import (
    SearxngProvider,
    build_searxng_client,
    validate_searxng_base_url,
)
from src.common.exception import ExternalServiceError, ValidationError

_LOOPBACK_BASE = "http://127.0.0.1:8888"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _searxng(handler: Callable[[httpx.Request], httpx.Response]) -> SearxngProvider:
    return SearxngProvider(client=_client(handler), base_url=_LOOPBACK_BASE)


# ── base_url validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "searxng:8080",
        "ftp://searxng:8080",
        "http://searxng.example.com/?x=1",
        "http://searxng.example.com/#frag",
        "http://127.0.0.1:8080",
    ],
)
def test_validate_base_url_rejects_invalid(raw: str, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(ValidationError):
        validate_searxng_base_url(raw)


def test_validate_base_url_accepts_whitelisted_loopback(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "127.0.0.1,localhost")
    assert validate_searxng_base_url(_LOOPBACK_BASE) == _LOOPBACK_BASE


def test_validate_base_url_strips_trailing_slash(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "127.0.0.1")
    assert validate_searxng_base_url(_LOOPBACK_BASE + "/") == _LOOPBACK_BASE


def test_validate_base_url_accepts_public_domain() -> None:
    assert validate_searxng_base_url("https://searxng.example.com") == "https://searxng.example.com"


# ── search ────────────────────────────────────────────────────────────


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T1",
                        "url": "https://e/1",
                        "content": "c1",
                        "publishedDate": "2024-05-01",
                    },
                    {"title": "", "url": "https://e/skip"},
                    {"title": "T2", "url": "https://e/2", "content": "c2"},
                ]
            },
        )

    provider = _searxng(handler)
    results = provider.search("hello", 5, True)
    request = seen["request"]

    assert request.url.path == "/search"
    assert dict(request.url.params) == {"q": "hello", "format": "json", "language": "all"}
    assert str(request.url).startswith(_LOOPBACK_BASE + "/search?")
    assert request.headers.get("Accept") == "application/json"
    assert results == [
        {
            "title": "T1",
            "url": "https://e/1",
            "snippet": "c1",
            "source": "searxng",
            "published_at": "2024-05-01T00:00:00",
        },
        {"title": "T2", "url": "https://e/2", "snippet": "c2", "source": "searxng"},
    ]


def test_search_respects_max_results_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"T{i}", "url": f"https://e/{i}", "content": "c"} for i in range(10)
                ]
            },
        )

    results = _searxng(handler).search("hello", 3, False)
    assert len(results) == 3


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(ExternalServiceError) as excinfo:
        _searxng(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "searxng returned status 500" in excinfo.value.message


def test_search_unparseable_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ExternalServiceError) as excinfo:
        _searxng(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_empty_result_diagnostics_reports_unresponsive_engines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [],
                "unresponsive_engines": [["google", "timeout"], ["bing", "down"]],
            },
        )

    provider = _searxng(handler)
    assert provider.search("hello", 5, False) == []
    assert "google" in provider.empty_result_diagnostics()


def test_build_requires_base_url() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_searxng_client({})
    assert excinfo.value.code == "web_search_provider.base_url_required"


def test_build_rejects_ssrf_unsafe_base_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        build_searxng_client({"base_url": "http://169.254.169.254/latest/meta-data"})
    assert excinfo.value.code == "web_search_provider.invalid_base_url"
