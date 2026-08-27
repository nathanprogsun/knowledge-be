"""Tests for the embedding base, providers, batch pooler, and concurrency.

All outbound HTTP is faked through ``httpx.MockTransport`` — no network.
Embedding base URLs use whitelisted test hostnames so the SSRF guard's
DNS step is bypassed.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import httpx
import pytest

from src.ai.embedding import (
    AliyunEmbedder,
    AzureOpenAIEmbedder,
    CloudEmbedder,
    ConcurrencyEmbedder,
    Config,
    Context,
    Embedder,
    GeminiEmbedder,
    JinaEmbedder,
    LocalLimiter,
    NvidiaEmbedder,
    OpenAIEmbedder,
    TaskContext,
    VolcengineEmbedder,
    ZhipuEmbedder,
    apply_custom_headers,
    config_from_model,
    gate_named_n,
    is_reserved_header,
    new_batch_embedder,
    new_embedder,
    new_ollama_embedder,
    new_openai_embedder,
    set_governor,
    validate_embedding_base_url,
    wrap_embedding_concurrency,
)
from src.ai.embedding import openai as openai_module
from src.ai.embedding.ollama import OllamaEmbedder
from src.ai.embedding.transport import new_embedding_http_client
from src.ai.utils.ollama_service import OllamaEmbedRequest, OllamaService
from src.common.exception import AIProviderError, ExternalServiceError, ValidationError
from src.common.json import JsonValue

_OPENAI_BASE = "https://embedding.test/v1"


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "embedding.test,api.openai.com")


@pytest.fixture(autouse=True)
def _reset_governor() -> Iterator[None]:
    set_governor(None, 0)
    yield
    set_governor(None, 0)


def _json_response(request: httpx.Request) -> httpx.Response:
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


# ── config_from_model ────────────────────────────────────────────────


class _Model:
    def __init__(
        self,
        *,
        id: str,
        name: str,
        source: str,
        parameters: dict[str, JsonValue],
    ) -> None:
        self.id = id
        self.name = name
        self.source = source
        self.parameters = parameters


def test_config_from_model_maps_every_field() -> None:
    model = _Model(
        id="mid",
        name="text-embedding",
        source="remote",
        parameters={
            "base_url": "https://x/v1",
            "api_key": "sk",
            "provider": "openai",
            "max_concurrency": 4,
            "extra_config": {"a": "b"},
            "custom_headers": {"X": "y"},
            "app_id": "aid",
            "app_secret": "sec",
            "embedding_parameters": {
                "dimension": 768,
                "supports_dimension_override": True,
                "truncate_prompt_tokens": 256,
            },
        },
    )
    config = config_from_model(model)
    assert config.source == "remote"
    assert config.base_url == "https://x/v1"
    assert config.api_key == "sk"
    assert config.model_id == "mid"
    assert config.model_name == "text-embedding"
    assert config.dimensions == 768
    assert config.supports_dimension_override is True
    assert config.truncate_prompt_tokens == 256
    assert config.provider == "openai"
    assert config.max_concurrency == 4
    assert config.extra_config == {"a": "b"}
    assert config.custom_headers == {"X": "y"}
    assert config.app_id == "aid"
    assert config.app_secret == "sec"


def test_config_from_model_defaults() -> None:
    model = _Model(id="mid", name="m", source="local", parameters={})
    config = config_from_model(model)
    assert config.base_url == ""
    assert config.dimensions == 0
    assert config.supports_dimension_override is False
    assert config.truncate_prompt_tokens == 0
    assert config.max_concurrency == 0
    assert config.extra_config == {}
    assert config.custom_headers == {}


def test_config_from_model_none_returns_empty() -> None:
    assert config_from_model(None) == Config()


# ── factory routing ──────────────────────────────────────────────────


async def test_factory_routes_local_to_ollama() -> None:
    embedder = await new_embedder(
        Config(source="local", model_name="nomic-embed-text", model_id="mid"),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OllamaEmbedder)
    assert embedder.get_model_name() == "nomic-embed-text"
    assert embedder.get_model_id() == "mid"


async def test_factory_routes_remote_openai() -> None:
    embedder = await new_embedder(
        Config(
            source="remote",
            provider="openai",
            base_url=_OPENAI_BASE,
            model_name="text-embedding-3-small",
            model_id="mid",
        ),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OpenAIEmbedder)
    assert embedder.get_model_name() == "text-embedding-3-small"


async def test_factory_detects_provider_from_base_url() -> None:
    embedder = await new_embedder(
        Config(
            source="remote",
            provider="",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            model_id="mid",
        ),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OpenAIEmbedder)


@pytest.mark.parametrize(
    ("provider", "model_name", "base_url", "expected"),
    [
        # Aliyun text models reuse the OpenAI-compatible client; multimodal
        # models go through the dedicated DashScope embedder.
        ("aliyun", "text-embedding-v1", f"{_OPENAI_BASE}/compatible-mode/v1", OpenAIEmbedder),
        ("aliyun", "tongyi-embedding-vision-v1", _OPENAI_BASE, AliyunEmbedder),
        ("aliyun", "multimodal-embedding-v1", _OPENAI_BASE, AliyunEmbedder),
        ("volcengine", "doubao-embedding", _OPENAI_BASE, VolcengineEmbedder),
        ("jina", "jina-embeddings-v2", _OPENAI_BASE, JinaEmbedder),
        ("azure_openai", "my-deployment", _OPENAI_BASE, AzureOpenAIEmbedder),
        ("nvidia", "nvolveqa40k", _OPENAI_BASE, NvidiaEmbedder),
        ("gemini", "text-embedding-004", _OPENAI_BASE, GeminiEmbedder),
        ("zhipu", "embedding-2", _OPENAI_BASE, ZhipuEmbedder),
    ],
)
async def test_factory_routes_remote_providers(
    provider: str,
    model_name: str,
    base_url: str,
    expected: type[Embedder],
) -> None:
    config = Config(
        source="remote",
        provider=provider,
        base_url=base_url,
        model_name=model_name,
        model_id="mid",
    )
    embedder = await new_embedder(config, None, None)
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, expected)


async def test_factory_routes_remote_cloud() -> None:
    config = Config(
        source="remote",
        provider="cloud",
        base_url=_OPENAI_BASE,
        model_name="m",
        model_id="mid",
        app_id="aid",
        app_secret="sec",
    )
    embedder = await new_embedder(config, None, None)
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, CloudEmbedder)


async def test_factory_rejects_unsupported_source() -> None:
    config = Config(source="bogus", model_name="m")
    with pytest.raises(ValidationError, match="unsupported embedder source"):
        await new_embedder(config, None, None)


async def test_factory_applies_dimension_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _json_response(request)

    mock_transport = httpx.MockTransport(handler)

    def _client_factory(
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> httpx.AsyncClient:
        return new_embedding_http_client(timeout=timeout, transport=mock_transport)

    monkeypatch.setattr(openai_module, "new_embedding_http_client", _client_factory)

    embedder = await new_embedder(
        Config(
            source="remote",
            provider="openai",
            base_url=_OPENAI_BASE,
            model_name="text-embedding-3-small",
            dimensions=768,
            supports_dimension_override=True,
            model_id="mid",
        ),
        pooler=None,
        ollama_service=None,
    )
    assert isinstance(embedder, ConcurrencyEmbedder)
    assert isinstance(embedder._inner, OpenAIEmbedder)
    result = await embedder.batch_embed(TaskContext(), ["x"])
    assert result == [[1.0]]


# ── OpenAI embedder ──────────────────────────────────────────────────


async def test_openai_batch_embed_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{_OPENAI_BASE}/embeddings"
        payload = json.loads(request.content)
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["hello", "world"]
        assert payload["encoding_format"] == "float"
        assert payload["truncate_prompt_tokens"] == 511
        assert "dimensions" not in payload
        assert request.headers["Authorization"] == "Bearer sk-test"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ]
            },
        )

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="text-embedding-3-small",
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


async def test_openai_embed_returns_single_vector() -> None:
    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(_json_response),
    )
    try:
        assert await embedder.embed(TaskContext(), "hello") == [5.0]
    finally:
        await embedder.aclose()


async def test_openai_embed_retries_when_batch_empty() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"data": []})

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match="no embedding returned"):
            await embedder.embed(TaskContext(), "x")
        assert calls["n"] == 3
    finally:
        await embedder.aclose()


async def test_openai_sends_dimensions_when_override_supported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["dimensions"] == 768
        return _json_response(request)

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=768,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    embedder.set_supports_dimension_override(True)
    try:
        result = await embedder.batch_embed(TaskContext(), ["x"])
        assert result == [[1.0]]
    finally:
        await embedder.aclose()


async def test_openai_api_error_surfaces_status_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match=r"EmbedBatch API error.*500"):
            await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_openai_retries_transport_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return _json_response(request)

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        max_retries=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await embedder.embed(TaskContext(), "hello")
        assert result == [5.0]
        assert calls["n"] == 2
    finally:
        await embedder.aclose()


async def test_openai_custom_headers_skip_reserved() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-Trace"] == "trace-1"
        assert request.headers.get("x-api-key") is None
        return _json_response(request)

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        custom_headers={
            "X-Trace": "trace-1",
            "x-api-key": "should-be-skipped",
            "Authorization": "should-be-skipped",
        },
        transport=httpx.MockTransport(handler),
    )
    try:
        await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_openai_malformed_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    embedder = await new_openai_embedder(
        api_key="sk-test",
        base_url=_OPENAI_BASE,
        model_name="m",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(AIProviderError, match="unmarshal response"):
            await embedder.batch_embed(TaskContext(), ["x"])
    finally:
        await embedder.aclose()


async def test_openai_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError, match="model name is required"):
        await new_openai_embedder(
            api_key="sk-test",
            base_url=_OPENAI_BASE,
            model_name="",
            truncate_prompt_tokens=0,
            dimensions=0,
            model_id="mid",
            pooler=None,
        )


# ── Ollama embedder ──────────────────────────────────────────────────


def _ollama_handler() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.3.13"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "nomic-embed-text:latest",
                            "size": 1,
                            "digest": "d",
                            "modified_at": "2024-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        if request.url.path == "/api/embed":
            payload = json.loads(request.content)
            assert payload["model"] == "nomic-embed-text"
            assert payload["truncate"] is True
            assert payload["options"] == {"num_ctx": 511}
            vectors = [[0.1, 0.2], [0.3, 0.4]]
            return httpx.Response(
                200,
                json={"embeddings": vectors[: len(payload["input"])]},
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    return httpx.MockTransport(handler)


def _ollama_service() -> OllamaService:
    return OllamaService(base_url="http://ollama.test", transport=_ollama_handler())


async def test_ollama_batch_embed() -> None:
    service = _ollama_service()
    embedder = await new_ollama_embedder(
        base_url="",
        model_name="",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        ollama_service=service,
    )
    result = await embedder.batch_embed(TaskContext(), ["hello", "world"])
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert embedder.get_model_name() == "nomic-embed-text"


async def test_ollama_embed_returns_single_vector() -> None:
    service = _ollama_service()
    embedder = await new_ollama_embedder(
        base_url="",
        model_name="nomic-embed-text",
        truncate_prompt_tokens=0,
        dimensions=0,
        model_id="mid",
        pooler=None,
        ollama_service=service,
    )
    assert await embedder.embed(TaskContext(), "hello") == [0.1, 0.2]


async def test_ollama_service_embeddings_raises_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = OllamaService(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ExternalServiceError, match="failed to get embedding vectors"):
        await service.embeddings(OllamaEmbedRequest(model="m", input=["x"]))


# ── Batch pooler ─────────────────────────────────────────────────────


class _EchoEmbedder:
    """Returns a distinct embedding per text; records batch sizes."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.batch_sizes: list[int] = []
        self.error_on: str | None = None

    async def embed(self, ctx: Context, text: str) -> list[float]:
        return [float(len(text))]

    async def batch_embed(self, ctx: Context, texts: list[str]) -> list[list[float]]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            self.batch_sizes.append(len(texts))
            if self.error_on is not None and self.error_on in texts:
                raise AIProviderError("boom", code="test.boom")
            return [[float(len(text))] for text in texts]
        finally:
            self.active -= 1

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def get_model_name(self) -> str:
        return "echo"

    def get_dimensions(self) -> int:
        return 0

    def get_model_id(self) -> str:
        return "echo-id"


async def test_batch_pooler_empty_input() -> None:
    pooler = new_batch_embedder()
    result = await pooler.batch_embed_with_pool(TaskContext(), _EchoEmbedder(), [])
    assert result == []


async def test_batch_pooler_preserves_order() -> None:
    pooler = new_batch_embedder(batch_size=2, max_workers=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    result = await pooler.batch_embed_with_pool(TaskContext(), _EchoEmbedder(), texts)
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]


async def test_batch_pooler_bounds_concurrency() -> None:
    pooler = new_batch_embedder(batch_size=1, max_workers=2)
    inner = _EchoEmbedder()
    texts = [f"t{i}" for i in range(10)]
    await pooler.batch_embed_with_pool(TaskContext(), inner, texts)
    assert inner.peak <= 2
    assert len(inner.batch_sizes) == 10


async def test_batch_pooler_propagates_first_error() -> None:
    pooler = new_batch_embedder(batch_size=1, max_workers=2)
    inner = _EchoEmbedder()
    inner.error_on = "t2"
    texts = [f"t{i}" for i in range(10)]
    with pytest.raises(AIProviderError, match="boom"):
        await pooler.batch_embed_with_pool(TaskContext(), inner, texts)


async def test_batch_pooler_rejects_invalid_sizes() -> None:
    with pytest.raises(ValidationError):
        new_batch_embedder(batch_size=0, max_workers=1)
    with pytest.raises(ValidationError):
        new_batch_embedder(batch_size=1, max_workers=0)


# ── Concurrency governor ─────────────────────────────────────────────


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.count = 0

    async def embed(self, ctx: Context, text: str) -> list[float]:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            self.count += 1
            return [0.0]
        finally:
            self.active -= 1

    async def batch_embed(self, ctx: Context, texts: list[str]) -> list[list[float]]:
        return [[0.0]] * len(texts)

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return [[0.0]] * len(texts)

    def get_model_name(self) -> str:
        return "recording"

    def get_dimensions(self) -> int:
        return 0

    def get_model_id(self) -> str:
        return "recording-id"


async def test_concurrency_passthrough_without_governor() -> None:
    inner = _RecordingEmbedder()
    wrapped = ConcurrencyEmbedder(inner, limit=1)
    await asyncio.gather(
        *(wrapped.embed(TaskContext(is_background_task=True), "x") for _ in range(5))
    )
    assert inner.count == 5
    assert inner.peak == 5


async def test_concurrency_gates_background_calls() -> None:
    set_governor(LocalLimiter(), 1)
    inner = _RecordingEmbedder()
    wrapped = ConcurrencyEmbedder(inner, limit=0)
    await asyncio.gather(
        *(wrapped.embed(TaskContext(is_background_task=True), "x") for _ in range(5))
    )
    assert inner.count == 5
    assert inner.peak == 1


async def test_concurrency_passes_interactive_calls() -> None:
    set_governor(LocalLimiter(), 1)
    inner = _RecordingEmbedder()
    wrapped = ConcurrencyEmbedder(inner, limit=0)
    await asyncio.gather(
        *(wrapped.embed(TaskContext(is_background_task=False), "x") for _ in range(5))
    )
    assert inner.count == 5
    assert inner.peak == 5


async def test_gate_returns_noop_for_non_background() -> None:
    set_governor(LocalLimiter(), 1)
    release = await gate_named_n(TaskContext(is_background_task=False), "mid", "m", 0)
    release()


async def test_wrap_embedding_concurrency_returns_wrapper() -> None:
    inner = _RecordingEmbedder()
    wrapped = wrap_embedding_concurrency(inner, 2)
    assert isinstance(wrapped, ConcurrencyEmbedder)
    assert wrapped.get_model_id() == "recording-id"


# ── Transport ────────────────────────────────────────────────────────


async def test_validate_embedding_base_url_empty_allowed() -> None:
    await validate_embedding_base_url("")


async def test_validate_embedding_base_url_rejects_restricted_host() -> None:
    with pytest.raises(ValidationError, match="base URL SSRF check failed"):
        await validate_embedding_base_url("http://localhost:11434/v1")


def test_apply_custom_headers_skips_reserved_and_empty() -> None:
    headers = {"Authorization": "Bearer x"}
    apply_custom_headers(
        headers,
        {
            "X-Trace": "abc",
            "authorization": "bad",
            "Content-Type": "bad",
            "": "empty-name",
        },
    )
    assert headers == {"Authorization": "Bearer x", "X-Trace": "abc"}


def test_apply_custom_headers_none_is_noop() -> None:
    headers = {"a": "b"}
    apply_custom_headers(headers, None)
    assert headers == {"a": "b"}


def test_is_reserved_header() -> None:
    assert is_reserved_header("Authorization") is True
    assert is_reserved_header("content-type") is True
    assert is_reserved_header("x-goog-api-key") is True
    assert is_reserved_header("X-Trace") is False


async def test_embedding_http_client_injects_transport() -> None:
    client = new_embedding_http_client(timeout=5.0, transport=httpx.MockTransport(_json_response))
    response = await client.post(f"{_OPENAI_BASE}/embeddings", json={"model": "m", "input": ["x"]})
    assert response.status_code == 200
    await client.aclose()
