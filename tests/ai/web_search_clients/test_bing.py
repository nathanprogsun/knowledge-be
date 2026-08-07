"""Unit tests for the Bing web-search client.

The httpx transport is mocked (``httpx.MockTransport``) so the tests
exercise request construction, response parsing and error
normalization without contacting the live API.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.bing import BingProvider, build_bing_client
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://api.bing.microsoft.com/v7.0/search"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _bing(handler: Callable[[httpx.Request], httpx.Response]) -> BingProvider:
    return BingProvider(client=_client(handler), api_key="test-key")


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "webPages": {
                    "value": [
                        {
                            "name": "Result 1",
                            "url": "https://example.com/1",
                            "snippet": "Summary 1",
                            "dateLastCrawled": "2024-05-01T12:00:00Z",
                        },
                        {"name": "Result 2", "url": "https://example.com/2", "snippet": "Summary 2"},
                    ]
                }
            },
        )

    provider = _bing(handler)
    results = provider.search("hello world", 3, True)
    request = seen["request"]

    assert str(request.url).startswith(_DEFAULT_URL + "?")
    assert dict(request.url.params) == {"q": "hello world", "count": "3"}
    assert request.headers.get("Ocp-Apim-Subscription-Key") == "test-key"
    assert results == [
        {
            "title": "Result 1",
            "url": "https://example.com/1",
            "snippet": "Summary 1",
            "source": "bing",
            "published_at": "2024-05-01T12:00:00+00:00",
        },
        {
            "title": "Result 2",
            "url": "https://example.com/2",
            "snippet": "Summary 2",
            "source": "bing",
        },
    ]


def test_search_empty_webpages_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"webPages": {"value": []}})

    assert _bing(handler).search("hello", 3, False) == []


def test_search_missing_webpages_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert _bing(handler).search("hello", 3, False) == []


def test_search_non_200_raises_external_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ExternalServiceError) as excinfo:
        _bing(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "bing API returned status 401" in excinfo.value.message


def test_search_unparseable_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ExternalServiceError) as excinfo:
        _bing(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_search_empty_query_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(ValidationError) as excinfo:
        _bing(handler).search("   ", 3, False)
    assert excinfo.value.code == "web_search_provider.query_required"


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_bing_client({})
    assert excinfo.value.code == "web_search_provider.api_key_required"


def test_build_rejects_invalid_proxy() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_bing_client({"api_key": "k", "proxy_url": "http://127.0.0.1:3128"})
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"
