"""Unit tests for the Keenable web-search client.

The httpx transport is mocked (``httpx.MockTransport``) so the tests
exercise request construction, header composition, response parsing and
error normalization without contacting the live API.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.keenable import KeenableProvider, build_keenable_client
from src.common.exception import ExternalServiceError, ValidationError

_BASE_URL = "https://api.keenable.ai"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _keenable(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-key",
    base_url: str = _BASE_URL,
) -> KeenableProvider:
    return KeenableProvider(client=_client(handler), api_key=api_key, base_url=base_url)


# ── Request construction ──────────────────────────────────────────────


def test_search_with_api_key_uses_authenticated_endpoint() -> None:
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
                        "description": "d1",
                        "snippet": "s1",
                        "published_at": "2024-05-01T12:00:00Z",
                    },
                    {
                        "title": "T2",
                        "url": "https://e/2",
                        "description": "d2",
                        "snippet": "s2",
                    },
                ]
            },
        )

    provider = _keenable(handler)
    results = provider.search("hello", 5, True)
    request = seen["request"]

    assert request.method == "POST"
    assert str(request.url) == f"{_BASE_URL}/v1/search"
    assert json.loads(request.read()) == {"query": "hello", "mode": "pro"}
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["X-Keenable-Title"] == "knowledge-be"
    assert request.headers["X-API-Key"] == "test-key"
    assert results == [
        {
            "title": "T1",
            "url": "https://e/1",
            "snippet": "d1",
            "content": "s1",
            "source": "keenable",
            "published_at": "2024-05-01T12:00:00+00:00",
        },
        {
            "title": "T2",
            "url": "https://e/2",
            "snippet": "d2",
            "content": "s2",
            "source": "keenable",
        },
    ]


def test_search_without_api_key_uses_public_endpoint() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _keenable(handler, api_key="").search("hello", 3, False)
    request = seen["request"]
    assert str(request.url) == f"{_BASE_URL}/v1/search/public"
    assert "X-API-Key" not in request.headers


def test_search_omits_published_at_when_include_date_false() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T1",
                        "url": "https://e/1",
                        "snippet": "s1",
                        "published_at": "2024-05-01",
                    }
                ]
            },
        )

    results = _keenable(handler).search("hello", 3, False)
    assert "published_at" not in results[0]


def test_search_skips_hits_without_title_or_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "", "url": ""},
                    {"title": "T1", "url": "https://e/1", "snippet": "s1"},
                    {"url": "https://e/2"},
                ]
            },
        )

    results = _keenable(handler).search("hello", 5, False)
    assert len(results) == 2
    assert results[0]["title"] == "T1"
    assert results[1]["title"] == ""


def test_search_uses_description_fallback_when_snippet_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T1",
                        "url": "https://e/1",
                        "description": "desc-only",
                    }
                ]
            },
        )

    results = _keenable(handler).search("hello", 3, False)
    assert results[0]["snippet"] == "desc-only"


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

    results = _keenable(handler).search("hello", 3, False)
    assert len(results) == 1


def test_search_respects_max_results_cap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"T{i}", "url": f"https://e/{i}"} for i in range(10)
                ]
            },
        )

    results = _keenable(handler).search("hello", 3, False)
    assert len(results) == 3


def test_search_uses_default_count_when_max_results_zero() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _keenable(handler).search("hello", 0, False)
    # Default cap = 5; with no items returned we just confirm the call
    # was issued.
    assert "request" in seen


def test_search_ignores_unparseable_published_at() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "T1",
                        "url": "https://e/1",
                        "published_at": "not a date",
                    }
                ]
            },
        )

    results = _keenable(handler).search("hello", 3, True)
    assert "published_at" not in results[0]


# ── Response handling ─────────────────────────────────────────────────


def test_search_empty_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert _keenable(handler).search("hello", 3, False) == []


def test_search_missing_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert _keenable(handler).search("hello", 3, False) == []


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ExternalServiceError) as excinfo:
        _keenable(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "keenable API returned status 401" in excinfo.value.message


def test_search_unparseable_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ExternalServiceError) as excinfo:
        _keenable(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_search_non_object_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["array", "not", "object"])

    with pytest.raises(ExternalServiceError) as excinfo:
        _keenable(handler).search("hello", 3, False)
    assert excinfo.value.code == "web_search_provider.invalid_response"


def test_search_empty_query_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    with pytest.raises(ValidationError) as excinfo:
        _keenable(handler).search("   ", 3, False)
    assert excinfo.value.code == "web_search_provider.query_required"


def test_custom_base_url_is_honored() -> None:
    custom = "https://keenable.example.com"
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"results": []})

    _keenable(handler, base_url=custom).search("hello", 3, False)
    assert str(seen["request"].url).startswith(custom + "/")


# ── Builder ──────────────────────────────────────────────────────────


def test_build_keenable_client_with_key() -> None:
    provider = build_keenable_client({"api_key": "k"})
    assert isinstance(provider, KeenableProvider)
    assert provider._api_key == "k"


def test_build_keenable_client_keyless() -> None:
    provider = build_keenable_client({})
    assert isinstance(provider, KeenableProvider)
    assert provider._api_key == ""


def test_build_rejects_invalid_proxy() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_keenable_client({"api_key": "k", "proxy_url": "http://127.0.0.1:3128"})
    assert excinfo.value.code == "web_search_provider.ssrf_blocked"


def test_build_accepts_public_proxy() -> None:
    provider = build_keenable_client(
        {"api_key": "k", "proxy_url": "https://proxy.example.com:3128"}
    )
    assert isinstance(provider, KeenableProvider)
