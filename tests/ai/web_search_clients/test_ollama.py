"""Unit tests for the Ollama Cloud web-search client.

The httpx transport is mocked (``httpx.MockTransport``) so the tests
exercise request construction, header composition, response parsing
and error normalization without contacting the live API.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients._base import DEFAULT_USER_AGENT
from src.ai.web_search_clients.ollama import OllamaProvider, build_ollama_client
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://ollama.com/api/web_search"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ollama(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-key",
    base_url: str = _DEFAULT_URL,
) -> OllamaProvider:
    return OllamaProvider(client=_client(handler), api_key=api_key, base_url=base_url)


# ── Request construction ──────────────────────────────────────────────


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
                        "snippet": "s1",
                        "content": "c1",
                    },
                    {
                        "title": "T2",
                        "url": "https://e/2",
                        "snippet": "s2",
                    },
                ]
            },
        )

    provider = _ollama(handler)
    results = provider.search("hello", 5, False)
    request = seen["request"]

    assert request.method == "POST"
    assert str(request.url) == _DEFAULT_URL
    assert json.loads(request.read()) == {"query": "hello", "max_results": 5}
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["User-Agent"] == DEFAULT_USER_AGENT
    assert results == [
        {
            "title": "T1",
            "url": "https://e/1",
            "snippet": "s1",
            "content": "c1",
            "source": "ollama",
        },
        {
            "title": "T2",
            "url": "https://e/2",
            "snippet": "s2",
            "content": "",
            "source": "ollama",
        },
    ]


def test_search_caps_max_results_at_ten() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _ollama(handler).search("hello", 1000, False)
    assert json.loads(seen["request"].read())["max_results"] == 10


def test_search_uses_default_count_when_max_results_zero() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _ollama(handler).search("hello", 0, False)
    assert json.loads(seen["request"].read())["max_results"] == 5


def test_search_skips_hits_without_title_or_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "", "url": ""},
                    {"title": "T1", "url": "https://e/1", "snippet": "s1"},
                    {"title": "T2"},
                ]
            },
        )

    # An empty title AND empty url is skipped; the rest are kept even
    # when one of (title, url) is empty.
    results = _ollama(handler).search("hello", 5, False)
    assert len(results) == 2
    assert results[0]["title"] == "T1"
    assert results[1]["title"] == "T2"
    assert results[1]["url"] == ""


def test_search_skips_non_dict_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    "not a dict",
                    {"title": "T1", "url": "https://e/1"},
                ]
            },
        )

    results = _ollama(handler).search("hello", 3, False)
    assert len(results) == 1


def test_custom_base_url_is_honored() -> None:
    custom = "https://ollama.example.com/api/web_search"
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _ollama(handler, base_url=custom).search("hello", 3, False)
    assert str(seen["request"].url) == custom


def test_search_emits_provider_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    provider = _ollama(handler)
    assert provider.provider_type == "ollama"


# ── Response handling ─────────────────────────────────────────────────


def test_search_empty_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert _ollama(handler).search("hello", 3, False) == []


def test_search_missing_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert _ollama(handler).search("hello", 3, False) == []


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ExternalServiceError) as excinfo:
        _ollama(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "ollama API returned status 401" in excinfo.value.message


def test_search_unparseable_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ExternalServiceError) as excinfo:
        _ollama(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_search_non_object_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["array", "instead", "of", "object"])

    with pytest.raises(ExternalServiceError) as excinfo:
        _ollama(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_search_empty_query_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(ValidationError) as excinfo:
        _ollama(handler).search("   ", 3, False)
    assert excinfo.value.code == "web_search_provider.query_required"


# ── Builder ──────────────────────────────────────────────────────────


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_ollama_client({})
    assert excinfo.value.code == "web_search_provider.api_key_required"


def test_build_accepts_api_key() -> None:
    provider = build_ollama_client({"api_key": "k"})
    assert isinstance(provider, OllamaProvider)
    assert provider._api_key == "k"
    assert provider._base_url == _DEFAULT_URL


def test_build_ignores_proxy_url() -> None:
    # Ollama does not honor an explicit proxy per the contract; the
    # builder must still produce a usable client.
    provider = build_ollama_client({"api_key": "k", "proxy_url": "https://proxy.example.com:3128"})
    assert isinstance(provider, OllamaProvider)
    assert provider._api_key == "k"
