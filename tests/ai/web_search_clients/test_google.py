"""Unit tests for the Google Custom Search web-search client."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.google import GoogleProvider, build_google_client
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://www.googleapis.com/customsearch/v1"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _google(handler: Callable[[httpx.Request], httpx.Response]) -> GoogleProvider:
    return GoogleProvider(client=_client(handler), api_key="key", cx="engine-1")


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "items": [
                    {"title": "T1", "link": "https://example.com/1", "snippet": "s1"},
                    {"title": "T2", "link": "https://example.com/2"},
                ]
            },
        )

    provider = _google(handler)
    results = provider.search("hello", 5, False)
    request = seen["request"]

    assert str(request.url).startswith(_DEFAULT_URL + "?")
    assert dict(request.url.params) == {
        "key": "key",
        "cx": "engine-1",
        "q": "hello",
        "num": "5",
        "hl": "ch-zh",
    }
    assert results == [
        {"title": "T1", "url": "https://example.com/1", "snippet": "s1", "source": "google"},
        {"title": "T2", "url": "https://example.com/2", "snippet": "", "source": "google"},
    ]


def test_search_uses_default_count_when_max_results_zero() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"items": []})

    _google(handler).search("hello", 0, False)
    assert dict(seen["request"].url.params)["num"] == "5"


def test_search_empty_items_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert _google(handler).search("hello", 5, False) == []


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    with pytest.raises(ExternalServiceError) as excinfo:
        _google(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "google API returned status 403" in excinfo.value.message


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_google_client({"cx": "engine-1"})
    assert excinfo.value.code == "web_search_provider.api_key_required"


def test_build_requires_cx() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_google_client({"api_key": "key"})
    assert excinfo.value.code == "web_search_provider.cx_required"


def test_build_accepts_legacy_engine_id_alias() -> None:
    provider = build_google_client({"api_key": "key", "engine_id": "e1"})
    assert isinstance(provider, GoogleProvider)
    assert provider._cx == "e1"
