"""Tests for the dedicated rerank provider backends.

Covers the wire shape of each dedicated backend (Aliyun, Zhipu, Jina,
NVIDIA, the kb, LKEAP, Volcengine) and its factory route. All HTTP
is faked through ``httpx.MockTransport`` and the LKEAP SDK is replaced by
a fake client — no network. The Volcengine IAM signature is pinned
against a known answer generated from the upstream Go signer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from src.ai.rerank.aliyun import AliyunReranker, new_aliyun_reranker
from src.ai.rerank.base import RerankerConfig, new_reranker
from src.ai.rerank.cloud import new_cloud_reranker
from src.ai.rerank.jina import new_jina_reranker
from src.ai.rerank.lkeap import LKEAPReranker, new_lkeap_reranker
from src.ai.rerank.nvidia import new_nvidia_reranker
from src.ai.rerank.volcengine import (
    new_volcengine_reranker,
    sign_rerank_request,
)
from src.ai.rerank.zhipu import new_zhipu_reranker
from src.ai.utils.signer import sign_request
from src.common.exception import ExternalServiceError, ValidationError

_BASE_URL = "http://rerank.test/v1"

# Default provider endpoints are whitelisted so constructor SSRF gates do
# not perform DNS resolution when the default endpoint is exercised.
_SSRF_WHITELIST = ",".join(
    [
        "rerank.test",
        "dashscope.aliyuncs.com",
        "open.bigmodel.cn",
        "api.jina.ai",
        "ai.api.nvidia.com",
        "api-knowledgebase.mlp.cn-beijing.volces.com",
    ]
)


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", _SSRF_WHITELIST)


def _json_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _capture(
    handler: Callable[[httpx.Request, dict[str, Any]], httpx.Response],
) -> tuple[httpx.AsyncClient, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        return handler(request, captured)

    return _json_client(wrapped), captured


# ── Aliyun ───────────────────────────────────────────────────────────


async def test_aliyun_reranker_sends_expected_request_and_parses_response() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"document": {"text": "A"}, "index": 1, "relevance_score": 0.9},
                        {"document": {"text": "B"}, "index": 0, "relevance_score": 0.4},
                    ]
                },
                "usage": {"total_tokens": 5},
            },
        )

    client, captured = _capture(handler)
    reranker = await new_aliyun_reranker(
        RerankerConfig(
            model_name="gte-rerank",
            model_id="al-1",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", ["A", "B", "C"])

    assert captured["path"] == "/v1"
    assert captured["authorization"] == "Bearer sk-test"
    body = captured["body"]
    assert body["model"] == "gte-rerank"
    assert body["input"] == {"query": "query", "documents": ["A", "B", "C"]}
    assert body["parameters"] == {"return_documents": True, "top_n": 3}

    assert [r.index for r in results] == [1, 0]
    assert results[0].document.text == "A"
    assert results[0].relevance_score == 0.9


async def test_aliyun_reranker_uses_default_endpoint() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"output": {"results": []}})

    client, captured = _capture(handler)
    reranker = await new_aliyun_reranker(
        RerankerConfig(model_name="m", model_id="al-2", api_key="sk-test"),
        client=client,
    )
    await reranker.rerank("query", ["A"])
    assert captured["url"].startswith(
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )


async def test_aliyun_reranker_raises_on_non_200() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client, _ = _capture(handler)
    reranker = await new_aliyun_reranker(
        RerankerConfig(
            model_name="m",
            model_id="al-3",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    with pytest.raises(ExternalServiceError, match="Aliyun rerank API error"):
        await reranker.rerank("query", ["A"])


async def test_aliyun_reranker_applies_custom_headers_via_factory() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["x_gateway"] = request.headers.get("X-Gateway")
        return httpx.Response(200, json={"output": {"results": []}})

    client, captured = _capture(handler)
    reranker = await new_reranker(
        RerankerConfig(
            model_name="m",
            model_id="al-4",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="aliyun",
            custom_headers={"X-Gateway": "g"},
        ),
        client=client,
    )
    await reranker.rerank("query", ["A"])
    assert captured["x_gateway"] == "g"
    assert isinstance(reranker, AliyunReranker)


# ── Zhipu ───────────────────────────────────────────────────────────


async def test_zhipu_reranker_sends_expected_request_and_parses_response() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "id": "task-1",
                "results": [
                    {"index": 0, "relevance_score": 0.95, "document": "A"},
                    {"index": 1, "relevance_score": 0.05, "document": "B"},
                ],
                "usage": {"total_tokens": 6, "prompt_tokens": 3},
            },
        )

    client, captured = _capture(handler)
    reranker = await new_zhipu_reranker(
        RerankerConfig(
            model_name="rerank",
            model_id="zp-1",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", ["A", "B"])

    assert captured["authorization"] == "Bearer sk-test"
    body = captured["body"]
    assert body == {
        "model": "rerank",
        "query": "query",
        "documents": ["A", "B"],
        "return_documents": True,
    }
    assert "top_n" not in body
    assert "return_raw_scores" not in body

    assert results[0].document.text == "A"
    assert results[0].relevance_score == 0.95


async def test_zhipu_reranker_raises_on_non_200() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    client, _ = _capture(handler)
    reranker = await new_zhipu_reranker(
        RerankerConfig(
            model_name="m",
            model_id="zp-2",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    with pytest.raises(ExternalServiceError, match="Zhipu rerank API error"):
        await reranker.rerank("query", ["A"])


# ── Jina ────────────────────────────────────────────────────────────


async def test_jina_reranker_posts_to_rerank_path() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jina-reranker-v2",
                "results": [
                    {"index": 2, "relevance_score": 0.98, "document": {"text": "C"}},
                    {"index": 0, "relevance_score": 0.1, "document": {"text": "A"}},
                ],
                "usage": {"total_tokens": 4},
            },
        )

    client, captured = _capture(handler)
    reranker = await new_jina_reranker(
        RerankerConfig(
            model_name="jina-reranker-v2",
            model_id="ji-1",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", ["A", "B", "C"])

    assert captured["path"] == "/v1/rerank"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["body"] == {
        "model": "jina-reranker-v2",
        "query": "query",
        "documents": ["A", "B", "C"],
        "return_documents": True,
    }
    assert [r.index for r in results] == [2, 0]
    assert results[0].document.text == "C"


# ── NVIDIA ──────────────────────────────────────────────────────────


async def test_nvidia_reranker_sends_expected_request_and_parses_response() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "nvidia-rerank",
                "rankings": [
                    {"index": 1, "logit": 0.9},
                    {"index": 0, "logit": 0.4},
                ],
            },
        )

    client, captured = _capture(handler)
    reranker = await new_nvidia_reranker(
        RerankerConfig(
            model_name="nvidia-rerank",
            model_id="nv-1",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", ["A", "B"])

    assert captured["path"] == "/v1"
    assert captured["body"] == {
        "model": "nvidia-rerank",
        "query": {"text": "query"},
        "passages": [{"text": "A"}, {"text": "B"}],
    }
    # Document text is recovered from the input list by index.
    assert [r.index for r in results] == [1, 0]
    assert results[0].document.text == "B"
    assert results[0].relevance_score == 0.9
    assert results[1].document.text == "A"


async def test_nvidia_reranker_out_of_range_index_degrades_to_empty() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            200,
            json={"rankings": [{"index": 5, "logit": 0.8}]},
        )

    client, _ = _capture(handler)
    reranker = await new_nvidia_reranker(
        RerankerConfig(
            model_name="m",
            model_id="nv-2",
            api_key="sk-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", ["A"])
    assert results[0].index == 5
    assert results[0].document.text == ""


# ── The kb ───────────────────────────────────────────────────


async def test_cloud_reranker_signs_and_sends_expected_request() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.8, "document": {"text": "不会保留"}},
                    {"index": 0, "relevance_score": 0.2, "document": {"text": "会保留"}},
                ]
            },
        )

    client, captured = _capture(handler)
    reranker = await new_cloud_reranker(
        RerankerConfig(
            model_name="rerank",
            model_id="wc-1",
            base_url=_BASE_URL,
            app_id="app-1",
            app_secret="secret-1",
        ),
        client=client,
    )
    results = await reranker.rerank("保留对话数据吗", ["会保留", "不会保留"])

    assert captured["path"] == "/v1/api/v1/rerank"
    headers = captured["headers"]
    assert headers["X-APPID"] == "app-1"
    assert headers["X-API-Key"] == "secret-1"
    assert headers["X-Request-ID"]
    assert headers["X-Timestamp"]
    assert headers["X-Nonce"]
    assert headers["X-Signature"]

    # The signature must match a recomputation over the exact wire body —
    # non-ASCII text stays raw UTF-8 on the wire, so the signature must
    # cover the unescaped representation.
    request_id = headers["X-Request-ID"]
    expected = sign_request(
        "app-1",
        "secret-1",
        request_id,
        captured["body"],
        timestamp=headers["X-Timestamp"],
        nonce=headers["X-Nonce"],
    )
    assert headers["X-Signature"] == expected["X-Signature"]

    body = json.loads(captured["body"])
    assert body == {
        "model": "rerank",
        "query": "保留对话数据吗",
        "documents": ["会保留", "不会保留"],
    }

    assert [r.index for r in results] == [1, 0]
    assert results[0].document.text == "不会保留"


async def test_cloud_reranker_uses_remote_model_name() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    client, captured = _capture(handler)
    reranker = await new_cloud_reranker(
        RerankerConfig(
            model_name="local-model",
            model_id="wc-2",
            base_url=_BASE_URL,
            app_id="app",
            app_secret="secret",
            extra_config={"remote_model_name": "remote-model"},
        ),
        client=client,
    )
    await reranker.rerank("query", ["A"])
    assert captured["body"]["model"] == "remote-model"


async def test_cloud_reranker_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="app id is required"):
        await new_cloud_reranker(
            RerankerConfig(model_name="m", model_id="wc-3", app_secret="secret")
        )
    with pytest.raises(ValidationError, match="app secret is required"):
        await new_cloud_reranker(RerankerConfig(model_name="m", model_id="wc-4", app_id="app"))


# ── LKEAP ───────────────────────────────────────────────────────────


class _FakeLkeapResponse:
    def __init__(self, scores: list[float | None]) -> None:
        self.ScoreList = scores


class _FakeLkeapClient:
    def __init__(self, response: _FakeLkeapResponse | None = None) -> None:
        self.requests: list[Any] = []
        self.response = response
        self.error: Exception | None = None

    async def RunRerank(self, request: Any) -> _FakeLkeapResponse | None:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


async def test_lkeap_reranker_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="secret_id and secret_key"):
        await new_lkeap_reranker(RerankerConfig(model_name="lke-reranker-base"))


async def test_lkeap_reranker_secret_key_from_extra_config() -> None:
    reranker = await new_lkeap_reranker(
        RerankerConfig(
            model_name="lke-reranker-base",
            api_key="secret-id",
            extra_config={"secret_key": "sk-test", "region": "ap-beijing"},
        ),
        client=_FakeLkeapClient(),
    )
    assert isinstance(reranker, LKEAPReranker)
    assert reranker.get_model_name() == "lke-reranker-base"


async def test_lkeap_reranker_default_model_name() -> None:
    reranker = await new_lkeap_reranker(
        RerankerConfig(api_key="secret-id", app_secret="sk-test"),
        client=_FakeLkeapClient(),
    )
    assert reranker.get_model_name() == "lke-reranker-base"


async def test_lkeap_reranker_empty_documents_skips_call() -> None:
    fake = _FakeLkeapClient()
    reranker = LKEAPReranker(model_name="m", model_id="lk-1", client=fake)
    results = await reranker.rerank("query", [])
    assert results == []
    assert fake.requests == []


async def test_lkeap_reranker_rejects_too_many_documents() -> None:
    fake = _FakeLkeapClient()
    reranker = LKEAPReranker(model_name="m", model_id="lk-2", client=fake)
    with pytest.raises(ValidationError, match="at most 60 documents"):
        await reranker.rerank("query", ["doc"] * 61)
    assert fake.requests == []


async def test_lkeap_reranker_sends_request_and_parses_response() -> None:
    fake = _FakeLkeapClient(response=_FakeLkeapResponse([0.9, None, 0.3]))
    reranker = LKEAPReranker(model_name="lke-reranker-base", model_id="lk-3", client=fake)
    results = await reranker.rerank("query", ["A", "B", "C"])

    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.Query == "query"
    assert request.Docs == ["A", "B", "C"]
    assert request.Model == "lke-reranker-base"

    assert [r.index for r in results] == [0, 1, 2]
    assert results[0].relevance_score == 0.9
    assert results[0].document.text == "A"
    assert results[1].relevance_score == 0.0  # nil score keeps the zero value
    assert results[2].relevance_score == 0.3


async def test_lkeap_reranker_raises_on_empty_score_list() -> None:
    fake = _FakeLkeapClient(response=_FakeLkeapResponse([]))
    reranker = LKEAPReranker(model_name="m", model_id="lk-4", client=fake)
    with pytest.raises(ExternalServiceError, match="empty score list"):
        await reranker.rerank("query", ["A"])


async def test_lkeap_reranker_raises_on_score_count_mismatch() -> None:
    fake = _FakeLkeapClient(response=_FakeLkeapResponse([0.9]))
    reranker = LKEAPReranker(model_name="m", model_id="lk-5", client=fake)
    with pytest.raises(ExternalServiceError, match="score count mismatch"):
        await reranker.rerank("query", ["A", "B"])


async def test_lkeap_reranker_wraps_sdk_error() -> None:
    fake = _FakeLkeapClient()
    fake.error = RuntimeError("network down")
    reranker = LKEAPReranker(model_name="m", model_id="lk-6", client=fake)
    with pytest.raises(ExternalServiceError, match="LKEAP RunRerank"):
        await reranker.rerank("query", ["A"])


# ── Volcengine ──────────────────────────────────────────────────────


def test_sign_rerank_request_matches_upstream_known_answer() -> None:
    # Expected values generated from the upstream Go IAM signer (service
    # "air", region cn-beijing, fixed X-Date 20260101T000000Z).
    body = (
        '{"datas":[{"query":"what is a banana?",'
        '"content":"A banana is a yellow fruit."}],'
        '"rerank_model":"doubao-seed-rerank",'
        '"rerank_instruction":"Whether the Document answers the Query '
        'or matches the content retrieval intent"}'
    )
    headers = sign_rerank_request(
        access_key="AKID-test",
        secret_key="SK-test-secret",
        region="cn-beijing",
        host="api-knowledgebase.mlp.cn-beijing.volces.com",
        body_bytes=body.encode("utf-8"),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert headers["X-Date"] == "20260101T000000Z"
    assert (
        headers["X-Content-Sha256"]
        == "b814df8c8627b92a981d1e422d8729df1f4c17244b3acd0176f639ff8ddb0db5"
    )
    assert headers["Authorization"] == (
        "HMAC-SHA256 Credential=AKID-test/20260101/cn-beijing/air/request, "
        "SignedHeaders=content-type;host;x-content-sha256;x-date, "
        "Signature=5ec2848f8e21520cdaa0dfe3efbcb33a2cfbc92ac8323fef25e9f66de12ff74a"
    )


async def test_volcengine_reranker_sends_signed_request_and_parses_response() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["host"] = request.headers.get("Host")
        captured["x_date"] = request.headers.get("X-Date")
        captured["x_content_sha256"] = request.headers.get("X-Content-Sha256")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={"code": 0, "message": "success", "data": {"scores": [0.91, 0.27]}},
        )

    client, captured = _capture(handler)
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-1",
            api_key="AKLT-test",
            base_url=_BASE_URL,
            app_secret="secret-test",
        ),
        client=client,
    )
    results = await reranker.rerank("保留对话数据吗", ["会保留", "不会保留"])

    assert captured["path"] == "/v1/api/knowledge/service/rerank"
    assert captured["host"] == "rerank.test"
    assert "AKLT-test" in captured["authorization"]
    assert "secret-test" not in captured["authorization"]
    assert captured["x_date"]
    assert captured["x_content_sha256"] == hashlib.sha256(captured["body"]).hexdigest()

    body = json.loads(captured["body"])
    assert body["rerank_model"] == "doubao-seed-rerank"
    assert "Document" in body["rerank_instruction"]
    assert body["datas"] == [
        {"query": "保留对话数据吗", "content": "会保留"},
        {"query": "保留对话数据吗", "content": "不会保留"},
    ]

    assert [r.index for r in results] == [0, 1]
    assert results[0].document.text == "会保留"
    assert results[0].relevance_score == 0.91
    assert results[1].document.text == "不会保留"


async def test_volcengine_reranker_uses_default_endpoint() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"code": 0, "data": {"scores": [0.5]}})

    client, captured = _capture(handler)
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-2",
            api_key="AKLT-test",
            app_secret="secret-test",
        ),
        client=client,
    )
    await reranker.rerank("query", ["A"])
    assert captured["url"].startswith(
        "https://api-knowledgebase.mlp.cn-beijing.volces.com/api/knowledge/service/rerank"
    )


async def test_volcengine_reranker_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="access key and secret key"):
        await new_volcengine_reranker(
            RerankerConfig(model_name="doubao-seed-rerank", api_key="ark-api-key-only")
        )


async def test_volcengine_reranker_empty_documents_skips_call() -> None:
    client = _json_client(lambda request: httpx.Response(500, text="should not run"))
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-3",
            api_key="AKLT-test",
            app_secret="secret-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    results = await reranker.rerank("query", [])
    assert results == []


async def test_volcengine_reranker_batches_over_limit() -> None:
    request_count = 0
    batch_sizes: list[int] = []

    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        nonlocal request_count
        body = json.loads(request.content)
        request_count += 1
        batch_sizes.append(len(body["datas"]))
        scores = [0.5 for _ in body["datas"]]
        return httpx.Response(200, json={"code": 0, "data": {"scores": scores}})

    client, _ = _capture(handler)
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-4",
            api_key="AKLT-test",
            app_secret="secret-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    total = 50 * 2 + 10
    documents = [f"doc-{i}" for i in range(total)]
    results = await reranker.rerank("query", documents)

    assert len(results) == total
    for i, result in enumerate(results):
        assert result.index == i
        assert result.document.text == f"doc-{i}"
        assert result.relevance_score == 0.5

    assert request_count == 3
    for size in batch_sizes:
        assert size <= 50


async def test_volcengine_reranker_raises_on_api_error_code() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"code": 100004, "message": "quota exceeded"})

    client, _ = _capture(handler)
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-5",
            api_key="AKLT-test",
            app_secret="secret-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    with pytest.raises(ExternalServiceError, match="100004: quota exceeded"):
        await reranker.rerank("query", ["a", "b"])


async def test_volcengine_reranker_raises_on_score_count_mismatch() -> None:
    def handler(request: httpx.Request, captured: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": {"scores": [0.9]}})

    client, _ = _capture(handler)
    reranker = await new_volcengine_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="vc-6",
            api_key="AKLT-test",
            app_secret="secret-test",
            base_url=_BASE_URL,
        ),
        client=client,
    )
    with pytest.raises(ExternalServiceError, match="score count mismatch"):
        await reranker.rerank("query", ["a", "b"])


# ── SSRF gate ───────────────────────────────────────────────────────


@pytest.mark.parametrize("provider", ["aliyun", "zhipu", "jina", "nvidia", "volcengine"])
async def test_dedicated_rerankers_reject_internal_base_url(provider: str) -> None:
    with pytest.raises(ValidationError, match="base URL SSRF check failed"):
        await new_reranker(
            RerankerConfig(
                model_name="m",
                model_id="ssrf-1",
                api_key="sk-test",
                app_secret="secret-test",
                base_url="http://169.254.169.254/latest/meta-data/",
                provider=provider,
            )
        )
