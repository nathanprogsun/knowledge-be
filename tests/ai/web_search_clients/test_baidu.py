"""Unit tests for the Baidu AI Search web-search client."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.web_search_clients.baidu import (
    BaiduProvider,
    build_baidu_client,
    normalize_baidu_query,
)
from src.common.exception import ExternalServiceError, ValidationError

_DEFAULT_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _baidu(handler: Callable[[httpx.Request], httpx.Response]) -> BaiduProvider:
    return BaiduProvider(client=_client(handler), api_key="test-key")


def test_search_returns_hits_and_builds_request() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "code": 0,
                "references": [
                    {
                        "id": 1,
                        "title": "T1",
                        "url": "https://e/1",
                        "content": "c1",
                        "date": "2025-4-24",
                        "type": "web",
                    },
                    {"id": 2, "title": "T2", "url": "https://e/2", "content": "c2"},
                ],
            },
        )

    provider = _baidu(handler)
    results = provider.search("hello", 5, True)
    request = seen["request"]

    assert str(request.url) == _DEFAULT_URL
    assert request.method == "POST"
    assert json.loads(request.read()) == {
        "messages": [{"role": "user", "content": "hello"}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": 5}],
    }
    assert request.headers.get("Authorization") == "Bearer test-key"
    assert results == [
        {
            "title": "T1",
            "url": "https://e/1",
            "content": "c1",
            "source": "baidu",
            "published_at": "2025-04-24T00:00:00",
        },
        {"title": "T2", "url": "https://e/2", "content": "c2", "source": "baidu"},
    ]


def test_search_api_level_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 17, "message": "rate limited"})

    with pytest.raises(ExternalServiceError) as excinfo:
        _baidu(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "baidu API error (code 17): rate limited" in excinfo.value.message


def test_search_non_200_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ExternalServiceError) as excinfo:
        _baidu(handler).search("hello", 5, False)
    assert excinfo.value.code == "web_search_provider.search_failed"
    assert "baidu API returned status 401" in excinfo.value.message


def test_search_empty_references_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "references": []})

    assert _baidu(handler).search("hello", 5, False) == []


def test_search_caps_max_results() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"code": 0, "references": []})

    _baidu(handler).search("hello", 1000, False)
    body = json.loads(seen["request"].read())
    assert body["resource_type_filter"] == [{"type": "web", "top_k": 50}]


def test_build_requires_api_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        build_baidu_client({})
    assert excinfo.value.code == "web_search_provider.api_key_required"


def test_normalize_baidu_query_truncates_cjk() -> None:
    query = "汉" * 40  # 80 units > 72
    normalized = normalize_baidu_query(query)
    assert len(normalized) == 36  # 72 units / 2 per rune
    assert normalize_baidu_query("  hello  ") == "hello"
