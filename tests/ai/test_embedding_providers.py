"""Tests for the remaining embedding providers.

Covers the eight provider modules that complete the ``new_embedder``
routing table: Aliyun (multimodal DashScope), Azure OpenAI, Gemini, Jina,
NVIDIA, Volcengine, the signed cloud endpoint, and Zhipu. Every
outbound HTTP call is faked through ``httpx.MockTransport`` — no network —
and the SSRF whitelist is pinned to the test hostname so the URL safety
guard is bypassed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from src.ai.embedding import (
    AzureOpenAIEmbedder,
    ConcurrencyEmbedder,
    Config,
    GeminiEmbedder,
    JinaEmbedder,
    NvidiaEmbedder,
    OpenAIEmbedder,
    TaskContext,
    VolcengineEmbedder,
    ZhipuEmbedder,
    new_embedder,
)
from src.ai.embedding.aliyun import new_aliyun_embedder
from src.ai.embedding.azure_openai import new_azure_openai_embedder
from src.ai.embedding.cloud import new_cloud_embedder
from src.ai.embedding.gemini import new_gemini_embedder
from src.ai.embedding.jina import new_jina_embedder
from src.ai.embedding.nvidia import new_nvidia_embedder
from src.ai.embedding.volcengine import new_volcengine_embedder
from src.ai.embedding.zhipu import new_zhipu_embedder
from src.ai.utils import signer as signer_module
from src.common.exception import AIProviderError, ValidationError

_BASE = "https://embedding.test"


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "embedding.test")


def _openai_data_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "data": [
                {"embedding": [float(len(text))], "index": index}
                for index, text in enumerate(payload["input"])
            ]
        },
    )


# ── Aliyun DashScope multimodal ───────────────────────────────────────


def _aliyun_handler(assertions) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == (
            f"{_BASE}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
        assert request.headers["Authorization"] == "Bearer sk-test"
        assertions(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"embedding": [0.1, 0.2], "text_index": 0},
                        {"embedding": [0.3, 0.4], "text_index": 1},
                    ]
                },
                "usage": {"total_tokens": 10},
                "request_id": "req-1",
            },
        )

    return httpx.MockTransport(handler)


async def test_aliyun_batch_embed_wire_format() -> None:
    def assertions(payload: dict) -> None:
        assert payload["model"] == "tongyi-embedding-vision-v1"
        assert payload["input"] == {"contents": [{"text": "hello"}, {"text": "world"}]}
        assert "parameters" not in payload

    embedder = await new_aliyun_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="tongyi-embedding-vision-v1",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=_aliyun_handler(assertions),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
    finally:
        await embedder.aclose()


async def test_aliyun_sends_dimension_when_override_supported() -> None:
    def assertions(payload: dict) -> None:
        assert payload["parameters"] == {"dimension": 768}

    embedder = await new_aliyun_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="tongyi-embedding-vision-v1",
        truncate_prompt_tokens=0,
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=_aliyun_handler(assertions),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_aliyun_orders_by_text_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"embedding": [9.0], "text_index": 1},
                        {"embedding": [8.0], "text_index": 0},
                    ]
                }
            },
        )

    embedder = await new_aliyun_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="tongyi-embedding-vision-v1",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await embedder.batch_embed(TaskContext(), ["a", "b"]) == [
            [8.0],
            [9.0],
        ]
    finally:
        await embedder.aclose()


async def test_aliyun_strips_compatible_mode_path() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"output": {"embeddings": []}})

    embedder = await new_aliyun_embedder(
        api_key="sk-test",
        base_url=f"{_BASE}/compatible-mode/v1",
        model_name="tongyi-embedding-vision-v1",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["url"] == (
            f"{_BASE}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
        )
    finally:
        await embedder.aclose()


async def test_aliyun_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError, match="model name is required"):
        await new_aliyun_embedder(
            api_key="sk-test",
            base_url=_BASE,
            model_name="",
            truncate_prompt_tokens=0,
            dimensions=0,
            model_id="mid",
            pooler=None,
        )


async def test_aliyun_api_error_surfaces_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "InvalidParameter", "message": "bad"})

    embedder = await new_aliyun_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="tongyi-embedding-vision-v1",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match=r"BatchEmbed API error.*400"):
            await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


# ── Azure OpenAI ──────────────────────────────────────────────────────


async def test_azure_batch_embed_wire_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == (
            f"{_BASE}/openai/deployments/my-deployment/embeddings?api-version=2024-10-21"
        )
        assert request.headers["api-key"] == "sk-test"
        payload = json.loads(request.content)
        assert payload["model"] == "my-deployment"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"
        assert "dimensions" not in payload
        assert "truncate_prompt_tokens" not in payload
        return _openai_data_response(request)

    embedder = await new_azure_openai_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="my-deployment",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        api_version="",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[5.0], [5.0]]
    finally:
        await embedder.aclose()


async def test_azure_uses_configured_api_version() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return _openai_data_response(request)

    embedder = await new_azure_openai_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="dep",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        api_version="2025-01-01",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert "api-version=2025-01-01" in captured["url"]
    finally:
        await embedder.aclose()


async def test_azure_sends_dimensions_when_override_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _openai_data_response(request)

    embedder = await new_azure_openai_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="dep",
        truncate_prompt_tokens=0,
        dimensions=768,
        model_id="mid",
        api_version="2024-10-21",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_azure_requires_base_url() -> None:
    with pytest.raises(ValidationError, match="Azure resource endpoint"):
        await new_azure_openai_embedder(
            api_key="sk-test",
            base_url="",
            model_name="dep",
            truncate_prompt_tokens=0,
            dimensions=0,
            model_id="mid",
            api_version="2024-10-21",
            pooler=None,
        )


async def test_azure_requires_model_name() -> None:
    with pytest.raises(ValidationError, match="deployment name"):
        await new_azure_openai_embedder(
            api_key="sk-test",
            base_url=_BASE,
            model_name="",
            truncate_prompt_tokens=0,
            dimensions=0,
            model_id="mid",
            api_version="2024-10-21",
            pooler=None,
        )


# ── Gemini ────────────────────────────────────────────────────────────


async def test_gemini_batch_embed_wire_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/models/text-embedding-004:batchEmbedContents"
        assert request.headers["x-goog-api-key"] == "sk-test"
        payload = json.loads(request.content)
        assert payload == {
            "requests": [
                {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": "hello"}]},
                },
                {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": "world"}]},
                },
            ]
        }
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [0.1, 0.2]},
                    {"values": [0.3, 0.4]},
                ]
            },
        )

    embedder = await new_gemini_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="text-embedding-004",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
    finally:
        await embedder.aclose()


async def test_gemini_sends_output_dimensionality() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["requests"][0]["output_dimensionality"] == 768
        return httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})

    embedder = await new_gemini_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="text-embedding-004",
        truncate_prompt_tokens=0,
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_gemini_normalizes_model_prefix_and_openai_suffix() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["model"] = json.loads(request.content)["requests"][0]["model"]
        return httpx.Response(200, json={"embeddings": [{"values": [0.1]}]})

    embedder = await new_gemini_embedder(
        api_key="sk-test",
        base_url=f"{_BASE}/openai",
        model_name="models/text-embedding-004",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["url"] == f"{_BASE}/models/text-embedding-004:batchEmbedContents"
        assert captured["model"] == "models/text-embedding-004"
    finally:
        await embedder.aclose()


async def test_gemini_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]})

    embedder = await new_gemini_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match="returned 2 embeddings for 1 inputs"):
            await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_gemini_empty_input_returns_empty() -> None:
    embedder = await new_gemini_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
    )
    try:
        assert await embedder.batch_embed(TaskContext(), []) == []
    finally:
        await embedder.aclose()


# ── Jina ──────────────────────────────────────────────────────────────


async def test_jina_batch_embed_wire_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-test"
        payload = json.loads(request.content)
        assert payload == {
            "model": "jina-embeddings-v2",
            "input": ["hello", "world"],
            "truncate": True,
        }
        return _openai_data_response(request)

    embedder = await new_jina_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="jina-embeddings-v2",
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[5.0], [5.0]]
    finally:
        await embedder.aclose()


async def test_jina_sends_dimensions_when_override_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _openai_data_response(request)

    embedder = await new_jina_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="jina-embeddings-v2",
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


# ── NVIDIA ────────────────────────────────────────────────────────────


async def test_nvidia_batch_embed_wire_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-test"
        payload = json.loads(request.content)
        assert payload["model"] == "nvolveqa40k"
        assert payload["input"] == ["hello"]
        assert payload["encoding_format"] == "float"
        assert payload["input_type"] == "passage"
        assert "truncate_prompt_tokens" not in payload
        return _openai_data_response(request)

    embedder = await new_nvidia_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="nvolveqa40k",
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["hello"])
    finally:
        await embedder.aclose()


async def test_nvidia_uses_query_input_type_for_query_context() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["input_type"] = json.loads(request.content)["input_type"]
        return _openai_data_response(request)

    embedder = await new_nvidia_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="nvolveqa40k",
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        ctx = SimpleNamespace(is_background_task=False, embed_query=True)
        await embedder.batch_embed(ctx, ["hello"])
        assert captured["input_type"] == "query"
    finally:
        await embedder.aclose()


async def test_nvidia_sends_dimensions_when_override_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _openai_data_response(request)

    embedder = await new_nvidia_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="nvolveqa40k",
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


# ── Volcengine Ark multimodal ─────────────────────────────────────────


async def test_volcengine_batch_embed_one_request_per_text() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/api/v3/embeddings/multimodal"
        assert request.headers["Authorization"] == "Bearer sk-test"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": {"embedding": [0.5, 0.6]},
                "model": "doubao-embedding",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    embedder = await new_volcengine_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="doubao-embedding",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["a", "b"])
        assert result == [[0.5, 0.6], [0.5, 0.6]]
        assert len(calls) == 2
        assert calls[0] == {
            "model": "doubao-embedding",
            "input": [{"type": "text", "text": "a"}],
        }
        assert calls[1]["input"] == [{"type": "text", "text": "b"}]
    finally:
        await embedder.aclose()


async def test_volcengine_strips_api_v3_suffix() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"data": {"embedding": [0.1]}},
        )

    embedder = await new_volcengine_embedder(
        api_key="sk-test",
        base_url=f"{_BASE}/api/v3",
        model_name="doubao-embedding",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["url"] == f"{_BASE}/api/v3/embeddings/multimodal"
    finally:
        await embedder.aclose()


async def test_volcengine_sends_dimensions_when_override_supported() -> None:
    captured: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": {"embedding": [0.1]}})

    embedder = await new_volcengine_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="doubao-embedding",
        truncate_prompt_tokens=0,
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["body"]["dimensions"] == 768
    finally:
        await embedder.aclose()


# ── The kb (signed) ────────────────────────────────────────────


def _signature_is_valid(request: httpx.Request, app_id: str, api_key: str) -> bool:
    body_json = request.content.decode("utf-8")
    signed = signer_module.sign_request(
        app_id,
        api_key,
        request.headers["X-Request-ID"],
        body_json,
        timestamp=request.headers["X-Timestamp"],
        nonce=request.headers["X-Nonce"],
    )
    return signed["X-Signature"] == request.headers["X-Signature"]


async def test_cloud_batch_embed_signed_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/api/v1/embeddings"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-APPID"] == "app-1"
        assert request.headers["X-API-Key"] == "secret-1"
        assert request.headers["X-Request-ID"]
        assert request.headers["X-Timestamp"]
        assert request.headers["X-Nonce"]
        assert _signature_is_valid(request, "app-1", "secret-1")
        payload = json.loads(request.content)
        assert payload == {"model": "m", "input": ["hello", "world"]}
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            },
        )

    embedder = await new_cloud_embedder(
        Config(
            source="remote",
            base_url=_BASE,
            model_name="m",
            model_id="mid",
            app_id="app-1",
            app_secret="secret-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
    finally:
        await embedder.aclose()


async def test_cloud_uses_remote_model_name() -> None:
    captured: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    embedder = await new_cloud_embedder(
        Config(
            source="remote",
            base_url=_BASE,
            model_name="local-name",
            model_id="mid",
            extra_config={"remote_model_name": "remote-name"},
            app_id="app-1",
            app_secret="secret-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["body"]["model"] == "remote-name"
        assert embedder.get_model_name() == "local-name"
    finally:
        await embedder.aclose()


async def test_cloud_sends_dimensions_when_override() -> None:
    captured: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    embedder = await new_cloud_embedder(
        Config(
            source="remote",
            base_url=_BASE,
            model_name="m",
            model_id="mid",
            dimensions=768,
            supports_dimension_override=True,
            app_id="app-1",
            app_secret="secret-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
        assert captured["body"]["dimensions"] == 768
    finally:
        await embedder.aclose()


async def test_cloud_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="AppID is required"):
        await new_cloud_embedder(
            Config(source="remote", base_url=_BASE, model_name="m", model_id="mid")
        )
    with pytest.raises(ValidationError, match="AppSecret is required"):
        await new_cloud_embedder(
            Config(
                source="remote",
                base_url=_BASE,
                model_name="m",
                model_id="mid",
                app_id="app-1",
            )
        )


async def test_cloud_api_error_surfaces_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    embedder = await new_cloud_embedder(
        Config(
            source="remote",
            base_url=_BASE,
            model_name="m",
            model_id="mid",
            app_id="app-1",
            app_secret="secret-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match=r"status 401"):
            await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


# ── Zhipu ─────────────────────────────────────────────────────────────


async def test_zhipu_batch_embed_wire_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_BASE}/embeddings"
        assert request.headers["Authorization"] == "Bearer sk-test"
        payload = json.loads(request.content)
        assert payload["model"] == "embedding-2"
        assert payload["input"] == ["hello", "world"]
        assert payload["truncate_prompt_tokens"] == 511
        assert "dimensions" not in payload
        return _openai_data_response(request)

    embedder = await new_zhipu_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="embedding-2",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
        assert result == [[5.0], [5.0]]
    finally:
        await embedder.aclose()


async def test_zhipu_sends_dimensions_when_override_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _openai_data_response(request)

    embedder = await new_zhipu_embedder(
        api_key="sk-test",
        base_url=_BASE,
        model_name="embedding-2",
        truncate_prompt_tokens=256,
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


# ── Custom headers + factory integration ─────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model_name", "expected"),
    [
        ("volcengine", "doubao-embedding", VolcengineEmbedder),
        ("jina", "jina-embeddings-v2", JinaEmbedder),
        ("nvidia", "nvolveqa40k", NvidiaEmbedder),
        ("gemini", "text-embedding-004", GeminiEmbedder),
        ("zhipu", "embedding-2", ZhipuEmbedder),
        ("azure_openai", "my-deployment", AzureOpenAIEmbedder),
    ],
)
async def test_factory_applies_custom_headers(
    provider: str,
    model_name: str,
    expected: type,
) -> None:
    custom_headers = {
        "X-Trace": "trace-1",
        "Authorization": "should-be-skipped",
        "x-goog-api-key": "should-be-skipped",
    }
    embedder = await new_embedder(
        Config(
            source="remote",
            provider=provider,
            base_url=_BASE,
            model_name=model_name,
            model_id="mid",
            custom_headers=custom_headers,
        ),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    inner = embedder._inner
    assert isinstance(inner, expected)
    # The factory attaches every user header; reserved names are dropped
    # at send time by the transport (covered by the transport tests).
    assert inner._custom_headers == custom_headers


async def test_factory_routes_aliyun_text_to_openai_compatible() -> None:
    embedder = await new_embedder(
        Config(
            source="remote",
            provider="aliyun",
            base_url=f"{_BASE}/compatible-mode/v1",
            model_name="text-embedding-v3",
            model_id="mid",
        ),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OpenAIEmbedder)
    await embedder._inner.aclose()
