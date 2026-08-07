"""Tests for the async Ollama service client.

All HTTP is faked through ``httpx.MockTransport`` — no network, no Ollama
install. The availability/optional behavior, tag normalization, and the
REST wire shapes (``/api/version``, ``/api/tags``, ``/api/pull``,
``/api/create``, ``/api/show``) are pinned here.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from src.ai.utils.ollama_service import (
    OllamaService,
    normalize_model_tag,
    resolve_ollama_dial_base_url,
)
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject

_BASE_URL = "http://ollama.test"


def _service(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    is_optional: bool | None = None,
) -> OllamaService:
    return OllamaService(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
        is_optional=is_optional,
    )


# ── version ──────────────────────────────────────────────────────────


async def test_get_version_returns_version_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.3.13"})

    service = _service(handler)
    assert await service.get_version() == "0.3.13"


async def test_get_version_raises_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="failed to get Ollama version"):
        await service.get_version()


# ── availability ─────────────────────────────────────────────────────


async def test_start_service_marks_available_on_heartbeat() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        assert request.url.path == "/"
        return httpx.Response(200)

    service = _service(handler)
    await service.start_service()
    assert service.is_available() is True


async def test_start_service_raises_when_unreachable_and_not_optional() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="ollama service unavailable"):
        await service.start_service()
    assert service.is_available() is False


async def test_start_service_swallows_when_optional() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    service = _service(handler, is_optional=True)
    await service.start_service()
    assert service.is_available() is False


async def test_optional_flag_reads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_OPTIONAL", "true")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    service = OllamaService(
        base_url=_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    await service.start_service()
    assert service.is_available() is False


# ── listing ──────────────────────────────────────────────────────────


async def test_list_models_returns_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest", "size": 100},
                    {"name": "qwen:7b", "size": 200},
                ]
            },
        )

    service = _service(handler)
    assert await service.list_models() == ["llama3:latest", "qwen:7b"]


async def test_list_models_detailed_returns_model_info() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "llama3:latest",
                        "size": 123,
                        "digest": "abc123",
                        "modified_at": "2024-05-04T12:00:00Z",
                    }
                ]
            },
        )

    service = _service(handler)
    models = await service.list_models_detailed()
    assert len(models) == 1
    assert models[0].name == "llama3:latest"
    assert models[0].size == 123
    assert models[0].digest == "abc123"
    assert models[0].modified_at is not None


async def test_list_models_detailed_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="failed to get model list"):
        await service.list_models_detailed()


# ── model availability / pull / ensure ───────────────────────────────


async def test_is_model_available_matches_with_latest_tag() -> None:
    tag_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            tag_calls.append(request.url.path)
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        return httpx.Response(200)

    service = _service(handler)
    assert await service.is_model_available("llama3") is True
    assert await service.is_model_available("llama3:latest") is True
    assert await service.is_model_available("qwen") is False
    assert len(tag_calls) == 3


async def test_optional_is_model_available_returns_false_without_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    service = _service(handler, is_optional=True)
    assert await service.is_model_available("llama3") is False


async def test_pull_model_skips_when_already_available() -> None:
    pull_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        if request.url.path == "/api/pull":
            pull_calls.append(request.url.path)
        return httpx.Response(200)

    service = _service(handler)
    await service.pull_model("llama3")
    assert pull_calls == []


async def test_pull_model_streams_progress_and_completes() -> None:
    frames = [
        {"status": "pulling manifest"},
        {"status": "downloading", "total": 100, "completed": 50},
        {"status": "success"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            assert json.loads(request.content) == {"name": "llama3", "stream": True}
            body = "\n".join(json.dumps(frame) for frame in frames)
            return httpx.Response(200, text=body)
        return httpx.Response(200)

    seen: list[tuple[float, str]] = []

    async def on_progress(percent: float, message: str) -> None:
        seen.append((percent, message))

    service = _service(handler)
    await service.pull_model("llama3", on_progress=on_progress)
    assert any(percent > 0 for percent, _ in seen)
    assert any("Pull progress" in message for _, message in seen)


async def test_pull_model_raises_on_error_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            return httpx.Response(200, text='{"error": "model not found"}')
        return httpx.Response(200)

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="ollama pull failed"):
        await service.pull_model("ghost")


async def test_pull_model_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            return httpx.Response(500, text="boom")
        return httpx.Response(200)

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="failed to pull model"):
        await service.pull_model("ghost")


async def test_ensure_model_available_pulls_when_missing() -> None:
    pull_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            pull_calls.append(request.url.path)
            return httpx.Response(200, text='{"status": "success"}')
        return httpx.Response(200)

    service = _service(handler)
    await service.ensure_model_available("llama3")
    assert pull_calls == ["/api/pull"]


async def test_ensure_model_available_skips_when_present() -> None:
    pull_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
        if request.url.path == "/api/pull":
            pull_calls.append(request.url.path)
        return httpx.Response(200)

    service = _service(handler)
    await service.ensure_model_available("llama3")
    assert pull_calls == []


# ── create / show ────────────────────────────────────────────────────


async def test_create_model_sends_expected_payload() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/create":
            sent["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(200)

    service = _service(handler)
    await service.create_model("mario", "FROM llama3")
    assert sent["body"] == {"model": "mario", "template": "FROM llama3"}


async def test_create_model_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/create":
            return httpx.Response(500, text="boom")
        return httpx.Response(200)

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="failed to create model"):
        await service.create_model("mario", "FROM llama3")


async def test_get_model_info_returns_raw_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": "mario"}
        return httpx.Response(
            200,
            json={"modelfile": "FROM llama3", "details": {"family": "llama"}},
        )

    service = _service(handler)
    info = await service.get_model_info("mario")
    assert info["modelfile"] == "FROM llama3"
    assert info["details"] == {"family": "llama"}


async def test_get_model_info_raises_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    service = _service(handler)
    with pytest.raises(ExternalServiceError, match="malformed response"):
        await service.get_model_info("mario")


# ── chat ─────────────────────────────────────────────────────────────


async def test_chat_sends_request_and_returns_response() -> None:
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "qwen2", "message": {"content": "hello"}, "done": True},
        )

    service = _service(handler)
    request: JsonObject = {
        "model": "qwen2",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    response = await service.chat(request)
    assert response == {"model": "qwen2", "message": {"content": "hello"}, "done": True}
    assert sent["body"] == request


async def test_chat_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = _service(handler)
    request: JsonObject = {"model": "qwen2", "stream": False, "messages": []}
    with pytest.raises(ExternalServiceError, match="failed to complete chat request"):
        await service.chat(request)


async def test_chat_raises_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    service = _service(handler)
    request: JsonObject = {"model": "qwen2", "stream": False, "messages": []}
    with pytest.raises(ExternalServiceError, match="malformed response"):
        await service.chat(request)


async def test_chat_stream_yields_frames_until_done() -> None:
    frames = [
        {"model": "qwen2", "message": {"content": "Hel"}, "done": False},
        {"model": "qwen2", "message": {"content": "lo"}, "done": False},
        {"model": "qwen2", "message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["stream"] is True
        lines = ["data: " + json.dumps(frame) for frame in frames]
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n".join(lines) + "\n")

    service = _service(handler)
    request: JsonObject = {"model": "qwen2", "stream": True, "messages": []}
    assert [frame async for frame in service.chat_stream(request)] == frames


async def test_chat_stream_skips_non_data_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='event: message\ndata: {"done": true}\n: comment\n',
        )

    service = _service(handler)
    request: JsonObject = {"model": "qwen2", "stream": True}
    frames = [frame async for frame in service.chat_stream(request)]
    assert frames == [{"done": True}]


async def test_chat_stream_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    service = _service(handler)
    request: JsonObject = {"model": "qwen2", "stream": True}
    with pytest.raises(ExternalServiceError, match="failed to stream chat response"):
        async for _frame in service.chat_stream(request):
            pass


# ── pure helpers ─────────────────────────────────────────────────────


def test_normalize_model_tag() -> None:
    assert normalize_model_tag("llama3") == "llama3:latest"
    assert normalize_model_tag("llama3:7b") == "llama3:7b"


def test_is_valid_model_name() -> None:
    assert OllamaService.is_valid_model_name("llama3") is True
    assert OllamaService.is_valid_model_name("") is False
    assert OllamaService.is_valid_model_name("llama 3") is False


def test_resolve_ollama_dial_base_url_falls_back_to_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert resolve_ollama_dial_base_url() == "http://localhost:11434"


def test_resolve_ollama_dial_base_url_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    assert resolve_ollama_dial_base_url() == "http://ollama:11434"


def test_base_url_property_is_stripped_of_trailing_slash() -> None:
    service = OllamaService(base_url=f"{_BASE_URL}/")
    assert service.base_url == _BASE_URL
