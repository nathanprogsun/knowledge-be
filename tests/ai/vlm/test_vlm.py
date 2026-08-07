"""Tests for the vision-language (VLM) clients.

All HTTP is faked through ``httpx.MockTransport`` — no network calls.
The config mapping, the three backend wire shapes (Ollama, remote
OpenAI-compatible, managed cloud with signed headers) and the
concurrency governor are pinned here.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast

import httpx
import pytest

from src.ai.provider.registry import PROVIDER_AZURE_OPENAI, PROVIDER_WEKNORACLOUD
from src.ai.utils.signer import sign_request
from src.ai.vlm import Config, RemoteAPIVLM, config_from_model, detect_image_mime, new_vlm
from src.ai.vlm.base import (
    MODEL_SOURCE_LOCAL,
    MODEL_SOURCE_REMOTE,
    ModelParametersLike,
)
from src.ai.vlm.concurrency import ConcurrencyVLM, _ModelGate, wrap_vlm_concurrency
from src.ai.vlm.ollama import OllamaVLM, new_ollama_vlm
from src.ai.vlm.remote_api import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    new_remote_api_vlm,
)
from src.ai.vlm.transport import (
    DEFAULT_TIMEOUT_SECONDS,
    validate_vlm_base_url,
    vlm_http_timeout,
)
from src.ai.vlm.weknoracloud import WeKnoraCloudVLM, new_weknoracloud_vlm
from src.common.exception import AIProviderError, ExternalServiceError, ValidationError
from src.common.json import JsonObject, JsonValue

# ── fixtures / helpers ──────────────────────────────────────────────


@pytest.fixture
def ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the SSRF validator accept the ``vlm.test`` test endpoint."""
    monkeypatch.setenv("SSRF_WHITELIST", "vlm.test")


def _header(headers: dict[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    raise AssertionError(f"missing header {name!r}")


def _content_parts(body: JsonObject) -> list[dict[str, JsonValue]]:
    """Project ``body["messages"][0]["content"]`` with type narrowing."""
    messages = body["messages"]
    assert isinstance(messages, list)
    first = messages[0]
    assert isinstance(first, dict)
    content = first["content"]
    assert isinstance(content, list)
    parts: list[dict[str, JsonValue]] = []
    for part in content:
        assert isinstance(part, dict)
        parts.append(part)
    return parts


class _FakeModelParameters:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str | None = None,
        interface_type: str | None = None,
        extra_config: dict[str, str] | None = None,
        custom_headers: dict[str, str] | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.interface_type = interface_type
        self.extra_config = extra_config
        self.custom_headers = custom_headers
        self.max_concurrency = max_concurrency


class _FakeModel:
    parameters: ModelParametersLike

    def __init__(
        self,
        *,
        id: str = "",
        name: str = "",
        source: str = "",
        parameters: _FakeModelParameters,
    ) -> None:
        self.id = id
        self.name = name
        self.source = source
        self.parameters = parameters


class _FakeOllamaService:
    def __init__(self) -> None:
        self.last_request: JsonObject | None = None
        self.error: Exception | None = None
        self.response: JsonObject = {"message": {"content": "hello"}}

    async def chat(self, chat_request: JsonObject) -> JsonObject:
        self.last_request = chat_request
        if self.error is not None:
            raise self.error
        return self.response


class _SlowVLM:
    def __init__(self, *, delay: float = 0.05) -> None:
        self._delay = delay
        self._active = 0
        self.max_active = 0

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        try:
            await asyncio.sleep(self._delay)
        finally:
            self._active -= 1
        return "ok"

    def get_model_name(self) -> str:
        return "slow-model"

    def get_model_id(self) -> str:
        return "slow-model-id"


# ── config_from_model ───────────────────────────────────────────────


def test_config_from_model_remote_defaults_to_openai() -> None:
    model = _FakeModel(
        id="v1",
        name="gpt-4o",
        source=MODEL_SOURCE_REMOTE,
        parameters=_FakeModelParameters(
            base_url="https://api.example.com/v1",
            api_key="sk",
            provider="openai",
            extra_config={"x": "y"},
            custom_headers={"H": "v"},
        ),
    )
    config = config_from_model(model, "app", "secret")
    assert config is not None
    assert config.interface_type == "openai"
    assert config.custom_headers == {"H": "v"}
    assert config.extra == {"x": "y"}
    assert config.app_id == "app"
    assert config.app_secret == "secret"


def test_config_from_model_local_defaults_to_ollama() -> None:
    model = _FakeModel(
        name="qwen2-vl",
        source=MODEL_SOURCE_LOCAL,
        parameters=_FakeModelParameters(),
    )
    config = config_from_model(model, "", "")
    assert config is not None
    assert config.interface_type == "ollama"


def test_config_from_model_respects_explicit_interface() -> None:
    model = _FakeModel(
        name="qwen2-vl",
        source=MODEL_SOURCE_REMOTE,
        parameters=_FakeModelParameters(interface_type="ollama"),
    )
    config = config_from_model(model, "", "")
    assert config is not None
    assert config.interface_type == "ollama"


def test_config_from_model_returns_none_for_none() -> None:
    assert config_from_model(None, "", "") is None


def test_config_from_model_maps_all_fields() -> None:
    model = _FakeModel(
        id="model-7",
        name="vl-model",
        source=MODEL_SOURCE_REMOTE,
        parameters=_FakeModelParameters(
            base_url="https://api.example.com",
            api_key="sk-1",
            provider="openai",
            interface_type="openai",
            extra_config={"a": "b"},
            custom_headers={"X-Tenant": "t1"},
            max_concurrency=4,
        ),
    )
    config = config_from_model(model, "app-1", "secret-1")
    assert config is not None
    assert config.model_id == "model-7"
    assert config.api_key == "sk-1"
    assert config.base_url == "https://api.example.com"
    assert config.model_name == "vl-model"
    assert config.source == MODEL_SOURCE_REMOTE
    assert config.interface_type == "openai"
    assert config.provider == "openai"
    assert config.max_concurrency == 4
    assert config.extra == {"a": "b"}
    assert config.custom_headers == {"X-Tenant": "t1"}
    assert config.app_id == "app-1"
    assert config.app_secret == "secret-1"


# ── transport helpers ───────────────────────────────────────────────


def test_detect_image_mime() -> None:
    assert detect_image_mime(b"\x89PNG\r\n\x1a\npayload") == "image/png"
    assert detect_image_mime(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image/jpeg"
    assert detect_image_mime(b"GIF89a\x01\x00\x01\x00") == "image/gif"
    assert detect_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert detect_image_mime(b"BM\x00\x00\x00\x00") == "image/bmp"
    assert detect_image_mime(b"II*\x00\x08\x00\x00\x00") == "image/tiff"
    assert detect_image_mime(b"not-an-image") == "image/png"


def test_vlm_http_timeout_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLM_HTTP_TIMEOUT_SECONDS", raising=False)
    assert vlm_http_timeout() == DEFAULT_TIMEOUT_SECONDS


def test_vlm_http_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_HTTP_TIMEOUT_SECONDS", "90")
    assert vlm_http_timeout() == 90.0


def test_vlm_http_timeout_ignores_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_HTTP_TIMEOUT_SECONDS", "abc")
    assert vlm_http_timeout() == DEFAULT_TIMEOUT_SECONDS


async def test_validate_vlm_base_url_empty_passes() -> None:
    await validate_vlm_base_url("")


async def test_validate_vlm_base_url_rejects_restricted_ip() -> None:
    with pytest.raises(ValidationError, match="SSRF"):
        await validate_vlm_base_url("http://127.0.0.1:8000")


# ── Ollama backend ──────────────────────────────────────────────────


def test_new_ollama_vlm_requires_service() -> None:
    with pytest.raises(AIProviderError, match="Ollama service is required"):
        new_ollama_vlm(model_name="qwen2-vl", model_id="m1", ollama_service=None)


async def test_ollama_vlm_predict_builds_chat_request() -> None:
    service = _FakeOllamaService()
    vlm = OllamaVLM(model_name="qwen2-vl", model_id="m1", ollama_service=service)
    result = await vlm.predict([b"\x89PNG\r\n\x1a\n"], "what is this?")
    assert result == "hello"
    request = service.last_request
    assert request is not None
    assert request["model"] == "qwen2-vl"
    assert request["stream"] is False
    options = request["options"]
    assert isinstance(options, dict)
    assert options == {"temperature": 0.1}
    messages = request["messages"]
    assert isinstance(messages, list)
    message = messages[0]
    assert isinstance(message, dict)
    assert message["role"] == "user"
    assert message["content"] == "what is this?"
    images = message["images"]
    assert isinstance(images, list)
    assert isinstance(images[0], str)
    assert images[0] == "iVBORw0KGgo="
    assert vlm.get_model_name() == "qwen2-vl"
    assert vlm.get_model_id() == "m1"


async def test_ollama_vlm_predict_wraps_service_error() -> None:
    service = _FakeOllamaService()
    service.error = ExternalServiceError("ollama down", code="ollama.unavailable")
    vlm = OllamaVLM(model_name="qwen2-vl", model_id="m1", ollama_service=service)
    with pytest.raises(AIProviderError, match="Ollama VLM request failed"):
        await vlm.predict([], "x")


# ── remote OpenAI-compatible backend ────────────────────────────────


async def test_remote_api_predict_sends_payload(ssrf_whitelist: None) -> None:
    sent_url: str = ""
    sent_headers: dict[str, str] = {}
    sent_body: JsonObject = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_url, sent_headers, sent_body
        sent_url = str(request.url)
        sent_headers = dict(request.headers)
        sent_body = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "text"}}]})

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test/v1",
        api_key="sk-test",
        provider="openai",
        extra=None,
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    result = await vlm.predict([b"\xff\xd8\xff\xe0jpeg"], "what is this?")
    assert result == "text"
    assert sent_url == "https://vlm.test/v1/chat/completions"
    assert _header(sent_headers, "Authorization") == "Bearer sk-test"
    assert sent_body["model"] == "gpt-4o"
    assert sent_body["max_tokens"] == DEFAULT_MAX_TOKENS
    assert sent_body["temperature"] == DEFAULT_TEMPERATURE
    parts = _content_parts(sent_body)
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["type"] == "image_url"
    image_url = parts[1]["image_url"]
    assert isinstance(image_url, dict)
    assert image_url["detail"] == "auto"
    assert str(image_url["url"]).startswith("data:image/jpeg;base64,")


async def test_remote_api_predict_omits_auth_when_no_key(ssrf_whitelist: None) -> None:
    sent_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_headers
        sent_headers = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="",
        provider="openai",
        extra=None,
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "ok"
    assert all(key.lower() != "authorization" for key in sent_headers)


async def test_remote_api_predict_returns_content(ssrf_whitelist: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        )

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="sk",
        provider="openai",
        extra=None,
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "answer"


async def test_remote_api_predict_raises_on_no_choices(ssrf_whitelist: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="sk",
        provider="openai",
        extra=None,
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderError, match="no choices"):
        await vlm.predict([], "x")


async def test_remote_api_predict_raises_on_http_error(ssrf_whitelist: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="sk",
        provider="openai",
        extra=None,
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderError, match="status 500"):
        await vlm.predict([], "x")


async def test_remote_api_azure_target(ssrf_whitelist: None) -> None:
    sent_url: str = ""
    sent_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_url, sent_headers
        sent_url = str(request.url)
        sent_headers = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = await new_remote_api_vlm(
        model_name="deployment-name",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="azure-key",
        provider=PROVIDER_AZURE_OPENAI,
        extra=cast(JsonObject, {"api_version": "2024-02-15-preview"}),
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "ok"
    expected = (
        "https://vlm.test/openai/deployments/deployment-name/"
        "chat/completions?api-version=2024-02-15-preview"
    )
    assert sent_url == expected
    assert _header(sent_headers, "api-key") == "azure-key"


async def test_remote_api_custom_headers(ssrf_whitelist: None) -> None:
    sent_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_headers
        sent_headers = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="sk",
        provider="openai",
        extra=None,
        custom_headers={"X-Custom": "custom-value"},
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "ok"
    assert _header(sent_headers, "X-Custom") == "custom-value"


async def test_remote_api_temperature_from_extra(ssrf_whitelist: None) -> None:
    sent_body: JsonObject = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_body
        sent_body = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = await new_remote_api_vlm(
        model_name="gpt-4o",
        model_id="m1",
        base_url="https://vlm.test",
        api_key="sk",
        provider="openai",
        extra=cast(JsonObject, {"temperature": "0.7"}),
        custom_headers=None,
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "ok"
    assert sent_body["temperature"] == 0.7


# ── managed-cloud backend ───────────────────────────────────────────


async def test_new_weknoracloud_vlm_requires_app_credentials() -> None:
    with pytest.raises(ValidationError, match="app id is required"):
        await new_weknoracloud_vlm(
            model_name="m",
            model_id="i",
            base_url="https://vlm.test",
            app_id="",
            app_secret="secret",
            extra=None,
        )
    with pytest.raises(ValidationError, match="app secret is required"):
        await new_weknoracloud_vlm(
            model_name="m",
            model_id="i",
            base_url="https://vlm.test",
            app_id="app",
            app_secret="",
            extra=None,
        )


async def test_weknoracloud_predict_sends_signed_request(ssrf_whitelist: None) -> None:
    sent_url: str = ""
    sent_headers: dict[str, str] = {}
    sent_body: str = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_url, sent_headers, sent_body
        sent_url = str(request.url)
        sent_headers = dict(request.headers)
        sent_body = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": [{"message": {"content": "understood"}}]})

    vlm = await new_weknoracloud_vlm(
        model_name="local-name",
        model_id="model-1",
        base_url="https://vlm.test",
        app_id="app-1",
        app_secret="secret-1",
        extra=None,
        transport=httpx.MockTransport(handler),
    )
    result = await vlm.predict([b"\x89PNG\r\n\x1a\nimg"], "describe this")
    assert result == "understood"
    assert sent_url == "https://vlm.test/api/v1/chat/completions"
    assert _header(sent_headers, "Content-Type") == "application/json"
    body = json.loads(sent_body)
    assert body["model"] == "local-name"
    assert body["stream"] is False
    assert body["max_tokens"] == DEFAULT_MAX_TOKENS
    parts = _content_parts(body)
    assert parts[0] == {"type": "text", "text": "describe this"}
    assert parts[1]["type"] == "image_url"
    image_url = parts[1]["image_url"]
    assert isinstance(image_url, dict)
    assert str(image_url["url"]).startswith("data:image/png;base64,")
    request_id = _header(sent_headers, "X-Request-ID")
    timestamp = _header(sent_headers, "X-Timestamp")
    nonce = _header(sent_headers, "X-Nonce")
    expected = sign_request(
        "app-1",
        "secret-1",
        request_id,
        sent_body,
        timestamp=timestamp,
        nonce=nonce,
    )
    assert _header(sent_headers, "X-Signature") == expected["X-Signature"]


async def test_weknoracloud_predict_uses_remote_model_name(ssrf_whitelist: None) -> None:
    sent_body: str = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent_body
        sent_body = request.content.decode("utf-8")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    vlm = await new_weknoracloud_vlm(
        model_name="local-name",
        model_id="model-1",
        base_url="https://vlm.test",
        app_id="app-1",
        app_secret="secret-1",
        extra=cast(JsonObject, {"remote_model_name": "  qwen-vl-plus  "}),
        transport=httpx.MockTransport(handler),
    )
    assert await vlm.predict([], "x") == "ok"
    body = json.loads(sent_body)
    assert body["model"] == "qwen-vl-plus"


async def test_weknoracloud_predict_raises_on_non_200(ssrf_whitelist: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    vlm = await new_weknoracloud_vlm(
        model_name="local-name",
        model_id="model-1",
        base_url="https://vlm.test",
        app_id="app-1",
        app_secret="secret-1",
        extra=None,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AIProviderError, match="status 401"):
        await vlm.predict([], "x")


# ── new_vlm factory routing ─────────────────────────────────────────


async def test_new_vlm_routes_local_to_ollama() -> None:
    service = _FakeOllamaService()
    config = Config(
        source=MODEL_SOURCE_LOCAL,
        model_name="qwen2-vl",
        model_id="m1",
        interface_type="",
    )
    vlm = await new_vlm(config, service)
    assert isinstance(vlm, ConcurrencyVLM)
    assert isinstance(vlm._inner, OllamaVLM)


async def test_new_vlm_routes_to_managed_cloud(ssrf_whitelist: None) -> None:
    config = Config(
        source=MODEL_SOURCE_REMOTE,
        base_url="https://vlm.test",
        model_name="cloud-vlm",
        model_id="m1",
        interface_type="openai",
        provider=PROVIDER_WEKNORACLOUD,
        app_id="app-1",
        app_secret="secret-1",
    )
    vlm = await new_vlm(config, None)
    assert isinstance(vlm, ConcurrencyVLM)
    assert isinstance(vlm._inner, WeKnoraCloudVLM)


async def test_new_vlm_routes_to_remote_api(ssrf_whitelist: None) -> None:
    config = Config(
        source=MODEL_SOURCE_REMOTE,
        base_url="https://vlm.test",
        model_name="gpt-4o",
        model_id="m1",
        interface_type="openai",
        provider="openai",
        api_key="sk-test",
    )
    vlm = await new_vlm(config, None)
    assert isinstance(vlm, ConcurrencyVLM)
    assert isinstance(vlm._inner, RemoteAPIVLM)


# ── concurrency governor ────────────────────────────────────────────


def test_wrap_vlm_concurrency_returns_decorator() -> None:
    wrapped = wrap_vlm_concurrency(_SlowVLM(), 0)
    assert isinstance(wrapped, ConcurrencyVLM)


async def test_concurrency_limits_parallel_calls() -> None:
    inner = _SlowVLM()
    vlm = ConcurrencyVLM(inner=inner, limit=1, gate=_ModelGate())
    results = await asyncio.gather(vlm.predict([], "a"), vlm.predict([], "b"))
    assert list(results) == ["ok", "ok"]
    assert inner.max_active == 1


async def test_concurrency_shares_slot_across_instances() -> None:
    gate = _ModelGate()
    inner = _SlowVLM()
    first = ConcurrencyVLM(inner=inner, limit=1, gate=gate)
    second = ConcurrencyVLM(inner=inner, limit=1, gate=gate)
    await asyncio.gather(first.predict([], "a"), second.predict([], "b"))
    assert inner.max_active == 1


async def test_concurrency_passthrough_when_limit_zero() -> None:
    inner = _SlowVLM()
    vlm = ConcurrencyVLM(inner=inner, limit=0)
    await asyncio.gather(vlm.predict([], "a"), vlm.predict([], "b"))
    assert inner.max_active == 2
