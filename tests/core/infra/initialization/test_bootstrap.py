"""Core tests for the initialization bootstrap.

Ollama is spoken to over its REST API, so every probe is driven through
an ``httpx.MockTransport`` — no network, no Ollama install. The remote
chat probe hits a hostname that ``validate_ssrf_safe_url`` accepts
(``example.com`` resolves publicly); the transport intercepts the request
before it leaves the process.
"""
# Chinese message strings are asserted verbatim against the Go originals.

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest

from src.common.exception import ExternalServiceError, NotFoundError, ValidationError
from src.core.contracts.infra import ModelTestRequest
from src.core.infra.initialization.factory import build_initialization_service
from src.core.infra.initialization.provider_detect import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_DISPLAY_FALLBACK_URL,
    OLLAMA_OPTIONAL_ENV,
    STATUS_COMPLETED,
    STATUS_FAILED,
    DownloadTaskStore,
    OllamaClient,
    classify_connection_error,
    normalize_model_tag,
    resolve_ollama_display_base_url,
)
from src.core.infra.initialization.service.initialization_service import InitializationService

_OLLAMA_BASE = "http://ollama.test:11434"
_REMOTE_BASE = "https://example.com/v1"


@pytest.fixture(autouse=True)
def clean_ollama_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the Ollama env so host config cannot leak into assertions."""
    monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(OLLAMA_OPTIONAL_ENV, raising=False)
    yield


def _service(
    *,
    ollama_handler: httpx.MockTransport | None = None,
    remote_handler: httpx.MockTransport | None = None,
    store: DownloadTaskStore | None = None,
) -> InitializationService:
    """Assemble the service with mock transports for both HTTP clients."""
    ollama_transport = ollama_handler or httpx.MockTransport(lambda _: httpx.Response(200))
    remote_transport = remote_handler or httpx.MockTransport(lambda _: httpx.Response(200))
    service = build_initialization_service(
        ollama_client=OllamaClient(base_url=_OLLAMA_BASE, transport=ollama_transport),
        http_client=httpx.AsyncClient(transport=remote_transport),
    )
    if store is not None:
        # The factory hands out the process-wide store; tests that assert
        # on task state need an isolated one.
        return InitializationService(
            ollama_client=OllamaClient(base_url=_OLLAMA_BASE, transport=ollama_transport),
            task_store=store,
            http_client=httpx.AsyncClient(transport=remote_transport),
        )
    return service


# ── Base-URL resolution ──────────────────────────────────────────────


def test_display_base_url_falls_back_to_docker_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: no OLLAMA_BASE_URL set (autouse fixture cleared it).
    # Act
    resolved = resolve_ollama_display_base_url()
    # Assert: Go's CheckOllamaStatus fallback, not the dial fallback.
    assert resolved == OLLAMA_DISPLAY_FALLBACK_URL


def test_display_base_url_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OLLAMA_BASE_URL_ENV, "http://custom:11434")
    assert resolve_ollama_display_base_url() == "http://custom:11434"


def test_untagged_model_name_resolves_to_latest() -> None:
    assert normalize_model_tag("qwen3") == "qwen3:latest"
    assert normalize_model_tag("qwen3:8b") == "qwen3:8b"


# ── check_ollama_status ──────────────────────────────────────────────


async def test_status_reports_available_with_version() -> None:
    # Arrange
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.5.1"})
        return httpx.Response(200)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    # Act
    status = await service.check_ollama_status()
    # Assert
    assert status.available is True
    assert status.version == "0.5.1"
    assert status.base_url == OLLAMA_DISPLAY_FALLBACK_URL


async def test_status_reports_unavailable_instead_of_raising() -> None:
    # Arrange: heartbeat refuses the connection.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    # Act
    status = await service.check_ollama_status()
    # Assert: Go returns 200 with available=false, never an error status.
    assert status.available is False
    assert status.version is None


async def test_status_version_probe_failure_degrades_to_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(500)
        return httpx.Response(200)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    status = await service.check_ollama_status()
    assert status.available is True
    assert status.version == "unknown"


# ── list_ollama_models ───────────────────────────────────────────────

_TAGS_PAYLOAD = {
    "models": [
        {
            "name": "qwen3:8b",
            "size": 5200000000,
            "digest": "sha256:abc",
            "modified_at": "2026-01-02T03:04:05Z",
        },
        {"name": "nomic-embed-text:latest", "size": 274000000, "digest": "sha256:def"},
    ]
}


async def test_list_models_returns_detailed_entries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_TAGS_PAYLOAD)
        return httpx.Response(200)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    models = await service.list_ollama_models()
    assert [m.name for m in models] == ["qwen3:8b", "nomic-embed-text:latest"]
    assert models[0].size == 5200000000
    assert models[1].modified_at is None


async def test_list_models_raises_when_ollama_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    with pytest.raises(ExternalServiceError):
        await service.list_ollama_models()


async def test_optional_mode_suppresses_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: OLLAMA_OPTIONAL=true — Go's StartService swallows the error.
    monkeypatch.setenv(OLLAMA_OPTIONAL_ENV, "true")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        raise httpx.ConnectError("connection refused", request=request)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    # Act
    models = await service.list_ollama_models()
    # Assert: heartbeat failed but the listing still ran.
    assert models == []
    assert "/api/tags" in calls


# ── check_ollama_models ──────────────────────────────────────────────


async def test_check_models_resolves_latest_tag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_TAGS_PAYLOAD)
        return httpx.Response(200)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    statuses = await service.check_ollama_models(["nomic-embed-text", "qwen3:8b", "absent"])
    assert statuses == {
        "nomic-embed-text": True,
        "qwen3:8b": True,
        "absent": False,
    }


async def test_check_models_marks_all_false_when_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OLLAMA_OPTIONAL_ENV, "true")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    service = _service(ollama_handler=httpx.MockTransport(handler))
    assert await service.check_ollama_models(["a", "b"]) == {"a": False, "b": False}


# ── downloads ────────────────────────────────────────────────────────


async def test_download_reports_already_present_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=_TAGS_PAYLOAD)
        return httpx.Response(200)

    store = DownloadTaskStore()
    service = _service(ollama_handler=httpx.MockTransport(handler), store=store)
    result = await service.download_ollama_model("qwen3:8b")
    assert result.message == "模型已存在"
    assert result.data["status"] == STATUS_COMPLETED
    assert result.data["progress"] == 100.0
    # No task registered for an already-present model.
    assert store.list_all() == []


async def test_download_creates_task_and_completes() -> None:
    pull_frames = [
        json.dumps({"status": "pulling manifest"}),
        json.dumps({"status": "downloading", "total": 100, "completed": 50}),
        json.dumps({"status": "success"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            return httpx.Response(200, text="\n".join(pull_frames))
        return httpx.Response(200)

    store = DownloadTaskStore()
    service = _service(ollama_handler=httpx.MockTransport(handler), store=store)
    # Act
    result = await service.download_ollama_model("qwen3:8b")
    task_id = result.data["taskId"]
    assert isinstance(task_id, str)
    # The pull runs detached; let it finish.
    await store.wait_for_pulls()
    # Assert
    assert result.message == "模型下载任务已创建"
    final = store.get(task_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.progress == 100.0
    assert final.end_time is not None


async def test_download_records_failure_on_pull_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/pull":
            return httpx.Response(500, text="boom")
        return httpx.Response(200)

    store = DownloadTaskStore()
    service = _service(ollama_handler=httpx.MockTransport(handler), store=store)
    result = await service.download_ollama_model("qwen3:8b")
    await store.wait_for_pulls()
    task_id = result.data["taskId"]
    assert isinstance(task_id, str)
    final = store.get(task_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert "下载失败" in final.message


async def test_download_deduplicates_against_in_flight_task() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200)

    store = DownloadTaskStore()
    existing = store.create(task_id="task-1", model_name="qwen3:8b")
    service = _service(ollama_handler=httpx.MockTransport(handler), store=store)
    result = await service.download_ollama_model("qwen3:8b")
    assert result.message == "模型下载任务已存在"
    assert result.data["taskId"] == existing.id
    assert len(store.list_all()) == 1


def test_progress_lookup_rejects_empty_id() -> None:
    service = _service(store=DownloadTaskStore())
    with pytest.raises(ValidationError):
        service.get_download_progress("")


def test_progress_lookup_raises_not_found() -> None:
    service = _service(store=DownloadTaskStore())
    with pytest.raises(NotFoundError):
        service.get_download_progress("missing")


def test_task_store_never_mutates_existing_records() -> None:
    # Arrange
    store = DownloadTaskStore()
    original = store.create(task_id="t1", model_name="m")
    # Act
    updated = store.update_status("t1", status=STATUS_COMPLETED, progress=100.0, message="done")
    # Assert: the original object is untouched (immutability rule).
    assert original.status != STATUS_COMPLETED
    assert updated is not None
    assert updated.status == STATUS_COMPLETED
    assert updated.end_time is not None


def test_list_download_tasks_returns_every_task() -> None:
    store = DownloadTaskStore()
    store.create(task_id="t1", model_name="a")
    store.create(task_id="t2", model_name="b")
    service = _service(store=store)
    assert {t.id for t in service.list_download_tasks()} == {"t1", "t2"}


# ── remote provider detection ────────────────────────────────────────


def test_error_classification_matches_go_hints() -> None:
    assert classify_connection_error("status code: 401") == "认证失败，请检查API Key"
    assert classify_connection_error("403 forbidden") == "权限不足，请检查API Key权限"
    assert classify_connection_error("404 not found") == "API端点不存在，请检查Base URL"
    assert classify_connection_error("context deadline exceeded") == "连接超时，请检查网络连接"
    assert classify_connection_error("connection refused") == "无法连接到服务器，请检查Base URL"
    assert classify_connection_error("weird failure") == "连接失败"


async def test_remote_check_requires_model_and_base_url() -> None:
    service = _service()
    with pytest.raises(ValidationError):
        await service.check_remote_model(ModelTestRequest(modelName="m", baseUrl=None))


async def test_remote_check_rejects_ssrf_target() -> None:
    service = _service()
    with pytest.raises(ValidationError):
        await service.check_remote_model(
            ModelTestRequest(modelName="m", baseUrl="http://localhost:8080/v1")
        )


async def test_remote_check_succeeds_on_200() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": []})

    service = _service(remote_handler=httpx.MockTransport(handler))
    result = await service.check_remote_model(
        ModelTestRequest(
            modelName="gpt-4o-mini",
            baseUrl=_REMOTE_BASE,
            apiKey="sk-test",
            customHeaders={"X-Trace": "1"},
        )
    )
    assert result.available is True
    assert result.message == "连接正常，模型可用"
    # Auth + custom headers are passed through to the provider.
    assert captured[0].headers["authorization"] == "Bearer sk-test"
    assert captured[0].headers["x-trace"] == "1"
    assert captured[0].url.path.endswith("/chat/completions")


async def test_remote_check_treats_400_as_reachable() -> None:
    # Go: a 400 means the endpoint answered and auth passed — only a
    # parameter was rejected — so the model counts as available.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "max_tokens unsupported"})

    service = _service(remote_handler=httpx.MockTransport(handler))
    result = await service.check_remote_model(ModelTestRequest(modelName="m", baseUrl=_REMOTE_BASE))
    assert result.available is True
    assert result.message == "连接正常，模型可用"


async def test_remote_check_surfaces_hint_and_raw_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    service = _service(remote_handler=httpx.MockTransport(handler))
    result = await service.check_remote_model(ModelTestRequest(modelName="m", baseUrl=_REMOTE_BASE))
    assert result.available is False
    # Format is "<hint>：<raw error>" so operators keep the upstream detail.
    assert result.message.startswith("认证失败，请检查API Key：")
    assert "invalid api key" in result.message
