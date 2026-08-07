"""Unit tests for the Zhipu web-search client."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.zhipu import (
    ZhipuProvider,
    build_zhipu_client,
    normalize_zhipu_query,
)
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _zhipu(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    search_engine: str = "search_std",
    content_size: str = "medium",
) -> ZhipuProvider:
    return ZhipuProvider(
        client=_client(handler),
        api_key="test-key",
        search_engine=search_engine,
        content_size=content_size,
    )


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "id": "search-id",
                "request_id": "request-id",
                "search_result": [
                    {
                        "title": "Result 1",
                        "link": "https://example.com/1",
                        "content": "Summary 1",
                        "publish_date": "2026-07-16",
                    },
                    {"title": "Result 2", "link": "https://example.com/2", "content": "Summary 2"},
                    {"title": "", "link": ""},
                ],
            },
        )

    provider = _zhipu(handler)
    results = provider.search("hello", 10, True)
    request = seen["request"]

    assert str(request.url) == _DEFAULT_URL
    assert request.method == "POST"
    assert json.loads(request.read()) == {
        "search_query": "hello",
        "search_engine": "search_std",
        "search_intent": False,
        "count": 10,
        "content_size": "medium",
    }
    assert request.headers.get("Authorization") == "Bearer test-key"
    assert results == [
        {
            "title": "Result 1",
            "url": "https://example.com/1",
            "snippet": "Summary 1",
            "source": "zhipu",
            "published_at": "2026-07-16T00:00:00",
        },
        {
            "title": "Result 2",
            "url": "https://example.com/2",
            "snippet": "Summary 2",
            "source": "zhipu",
        },
    ]


def test_search_uses_configured_engine_and_content_size() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"search_result": []})

    provider = _zhipu(handler, search_engine="search_pro_sogou", content_size="high")
    provider.search("hello", 10, False)
    body = json.loads(seen["request"].read())
    assert body["search_engine"] == "search_pro_sogou"
    assert body["content_size"] == "high"


def test_search_caps_max_results_and_defaults() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"search_result": []})

    _zhipu(handler).search("hello", 1000, False)
    assert json.loads(seen["request"].read())["count"] == 50
    _zhipu(handler).search("hello", 0, False)
    assert json.loads(seen["request"].read())["count"] == 10


def test_search_api_level_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"search_result": [], "error": {"code": "1111", "message": "bad key"}},
        )

    with pytest.raises(ExternalServiceError) as excinfo:
        _zhipu(handler).search("hello", 10, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "Zhipu API error (1111): bad key" in excinfo.value.message


def test_search_non_200_prefers_error_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": "1002", "message": "auth failed"}},
        )

    with pytest.raises(ExternalServiceError) as excinfo:
        _zhipu(handler).search("hello", 10, False)
    assert "Zhipu API returned status 401 (1002): auth failed" in excinfo.value.message


def test_search_non_200_falls_back_to_body_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    with pytest.raises(ExternalServiceError) as excinfo:
        _zhipu(handler).search("hello", 10, False)
    assert "Zhipu API returned status 503: service unavailable" in excinfo.value.message


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_zhipu_client({})
    assert excinfo.value.code == "web_search_provider.api_key_required"


def test_build_rejects_invalid_engine() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_zhipu_client({"api_key": "k", "extra_config": {"search_engine": "unknown"}})
    assert excinfo.value.code == "web_search_provider.invalid_config"


def test_build_rejects_invalid_content_size() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_zhipu_client({"api_key": "k", "extra_config": {"content_size": "large"}})
    assert excinfo.value.code == "web_search_provider.invalid_config"


def test_build_accepts_custom_options() -> None:
    provider = build_zhipu_client(
        {
            "api_key": "k",
            "extra_config": {"search_engine": "search_pro", "content_size": "high"},
        }
    )
    assert isinstance(provider, ZhipuProvider)
    assert provider._search_engine == "search_pro"
    assert provider._content_size == "high"


def test_normalize_zhipu_query_truncates_runes() -> None:
    query = "中" * 75
    normalized = normalize_zhipu_query(query)
    assert len(normalized) == 70
    assert normalize_zhipu_query("  hello  ") == "hello"
