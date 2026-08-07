"""Unit tests for the Tavily web-search client."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.tavily import TavilyProvider, build_tavily_client
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://api.tavily.com/search"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _tavily(handler: Callable[[httpx.Request], httpx.Response]) -> TavilyProvider:
    return TavilyProvider(client=_client(handler), api_key="test-key")


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "query": "hello",
                "results": [
                    {
                        "title": "T1",
                        "url": "https://example.com/1",
                        "content": "c1",
                        "score": 0.9,
                        "published_date": "2024-05-01T12:00:00Z",
                    },
                    {"title": "T2", "url": "https://example.com/2", "content": "c2"},
                ],
            },
        )

    provider = _tavily(handler)
    results = provider.search("hello", 5, True)
    request = seen["request"]

    assert str(request.url) == _DEFAULT_URL
    assert request.method == "POST"
    assert json.loads(request.read()) == {
        "api_key": "test-key",
        "query": "hello",
        "max_results": 5,
    }
    assert results == [
        {
            "title": "T1",
            "url": "https://example.com/1",
            "snippet": "c1",
            "source": "tavily",
            "published_at": "2024-05-01T12:00:00+00:00",
        },
        {"title": "T2", "url": "https://example.com/2", "snippet": "c2", "source": "tavily"},
    ]


def test_search_omits_published_at_when_include_date_false() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T1",
                        "url": "https://example.com/1",
                        "content": "c1",
                        "published_date": "2024-05-01T12:00:00Z",
                    }
                ]
            },
        )

    results = _tavily(handler).search("hello", 5, False)
    assert "published_at" not in results[0]


def test_search_empty_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert _tavily(handler).search("hello", 5, False) == []


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    with pytest.raises(ExternalServiceError) as excinfo:
        _tavily(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "tavily API returned status 400" in excinfo.value.message


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_tavily_client({})
    assert excinfo.value.code == "web_search_provider.api_key_required"
