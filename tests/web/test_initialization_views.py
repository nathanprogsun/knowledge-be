"""Web-layer tests for the initialization router.

Per AGENTS.md §9 the router is exercised over ``httpx.AsyncClient``
against the real app, with the service dependency overridden by a real
``InitializationService`` whose two HTTP clients are backed by
``httpx.MockTransport``. That keeps routing, serialization, the role
gates and the exception handler in the loop while never touching the
network.
"""
# Chinese message strings are asserted verbatim against the Go originals.
# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
from src.core.infra.initialization.provider_detect import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_OPTIONAL_ENV,
    STATUS_DOWNLOADING,
    DownloadTaskStore,
    OllamaClient,
)
from src.core.infra.initialization.service.initialization_service import InitializationService
from src.web.api.infra.initialization.router import router as initialization_router
from src.web.deps.infra_initialization import get_initialization_service
from tests.unit.fakes.auth_gates import override_auth_gates

_OLLAMA_BASE = "http://ollama.test:11434"
_REMOTE_BASE = "https://example.com/v1"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def clean_ollama_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(OLLAMA_OPTIONAL_ENV, raising=False)
    yield


@pytest.fixture
def task_store() -> DownloadTaskStore:
    return DownloadTaskStore()


def _ollama_handler(request: httpx.Request) -> httpx.Response:
    """Default fake Ollama: reachable, one model installed."""
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.5.1"})
    if request.url.path == "/api/tags":
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:8b",
                        "size": 5200000000,
                        "digest": "sha256:abc",
                        "modified_at": "2026-01-02T03:04:05Z",
                    }
                ]
            },
        )
    if request.url.path == "/api/pull":
        return httpx.Response(200, text='{"status":"success"}')
    return httpx.Response(200)


def _remote_handler(request: httpx.Request) -> httpx.Response:
    """Default fake provider: everything answers happily."""
    path = request.url.path
    if path.endswith("/chat/completions"):
        return httpx.Response(200, json={"choices": []})
    if path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]})
    if path.endswith("/rerank"):
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})
    if path.endswith("/audio/transcriptions"):
        return httpx.Response(200, json={"text": "hello"})
    return httpx.Response(404, text="unexpected path")


def _build_app(
    *,
    task_store: DownloadTaskStore,
    ollama: Handler = _ollama_handler,
    remote: Handler = _remote_handler,
) -> FastAPI:
    """App with the initialization router mounted and gates neutralised.

    The router is not yet wired into ``create_app`` — mounting it is
    deferred (``lifespan.py`` is off-limits) — so the test mounts it
    explicitly.
    """
    application = create_app()
    application.include_router(initialization_router)
    override_auth_gates(application)

    def _override() -> InitializationService:
        return InitializationService(
            ollama_client=OllamaClient(
                base_url=_OLLAMA_BASE, transport=httpx.MockTransport(ollama)
            ),
            task_store=task_store,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(remote)),
        )

    application.dependency_overrides[get_initialization_service] = _override
    return application


@pytest.fixture
def app(task_store: DownloadTaskStore) -> FastAPI:
    return _build_app(task_store=task_store)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── GET /initialization/ollama/status ────────────────────────────────


async def test_ollama_status_returns_go_envelope(client: AsyncClient) -> None:
    resp = await client.get("/initialization/ollama/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # camelCase alias, matching Go's gin.H key.
    assert body["data"]["baseUrl"] == "http://host.docker.internal:11434"
    assert body["data"]["available"] is True
    assert body["data"]["version"] == "0.5.1"


async def test_ollama_status_reports_unavailable_as_200(
    task_store: DownloadTaskStore,
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _build_app(task_store=task_store, ollama=down)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/initialization/ollama/status")
    assert resp.status_code == 200
    assert resp.json()["data"]["available"] is False


# ── GET /initialization/ollama/models ────────────────────────────────


async def test_list_ollama_models(client: AsyncClient) -> None:
    resp = await client.get("/initialization/ollama/models")
    assert resp.status_code == 200
    models = resp.json()["data"]
    assert models[0]["name"] == "qwen3:8b"
    assert models[0]["size"] == 5200000000


async def test_list_ollama_models_maps_unavailable_to_502(
    task_store: DownloadTaskStore,
) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = _build_app(task_store=task_store, ollama=down)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/initialization/ollama/models")
    # ExternalServiceError -> 502 via the shared exception handler.
    assert resp.status_code == 502


# ── POST /initialization/ollama/models/check ─────────────────────────


async def test_check_ollama_models_returns_per_name_map(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/ollama/models/check",
        json={"models": ["qwen3:8b", "absent"]},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == {"qwen3:8b": True, "absent": False}


# ── POST /initialization/ollama/models/download ──────────────────────


async def test_download_reports_already_present(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/ollama/models/download",
        json={"modelName": "qwen3:8b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "模型已存在"
    assert body["data"]["status"] == "completed"


async def test_download_creates_task(
    client: AsyncClient,
    task_store: DownloadTaskStore,
) -> None:
    resp = await client.post(
        "/initialization/ollama/models/download",
        json={"modelName": "llama3:8b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "模型下载任务已创建"
    assert body["data"]["modelName"] == "llama3:8b"
    assert isinstance(body["data"]["taskId"], str)
    await task_store.wait_for_pulls()


async def test_download_rejects_missing_model_name(client: AsyncClient) -> None:
    resp = await client.post("/initialization/ollama/models/download", json={})
    assert resp.status_code == 422


# ── GET /initialization/ollama/download/progress/{taskId} ────────────


async def test_download_progress_uses_go_json_names(
    client: AsyncClient,
    task_store: DownloadTaskStore,
) -> None:
    task_store.create(task_id="task-1", model_name="qwen3:8b")
    task_store.update_status("task-1", status=STATUS_DOWNLOADING, progress=42.5, message="下载中")
    resp = await client.get("/initialization/ollama/download/progress/task-1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["id"] == "task-1"
    assert data["modelName"] == "qwen3:8b"
    assert data["status"] == "downloading"
    assert data["progress"] == 42.5
    assert "startTime" in data
    assert data["endTime"] is None


async def test_download_progress_unknown_task_is_404(client: AsyncClient) -> None:
    resp = await client.get("/initialization/ollama/download/progress/missing")
    assert resp.status_code == 404


# ── GET /initialization/ollama/download/tasks ────────────────────────


async def test_list_download_tasks(
    client: AsyncClient,
    task_store: DownloadTaskStore,
) -> None:
    task_store.create(task_id="t1", model_name="a")
    task_store.create(task_id="t2", model_name="b")
    resp = await client.get("/initialization/ollama/download/tasks")
    assert resp.status_code == 200
    assert {t["id"] for t in resp.json()["data"]} == {"t1", "t2"}


# ── POST /initialization/remote/check ────────────────────────────────


async def test_remote_check_available(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/remote/check",
        json={"model": "gpt-4o-mini", "baseUrl": _REMOTE_BASE, "apiKey": "sk-x"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"] == {"available": True, "message": "连接正常，模型可用"}


async def test_remote_check_missing_base_url_is_422(client: AsyncClient) -> None:
    resp = await client.post("/initialization/remote/check", json={"model": "m"})
    assert resp.status_code == 422


async def test_remote_check_blocks_ssrf_target(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/remote/check",
        json={"model": "m", "baseUrl": "http://127.0.0.1:8080/v1"},
    )
    assert resp.status_code == 422


# ── POST /initialization/embedding/test ──────────────────────────────


async def test_embedding_test_returns_dimension(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/embedding/test",
        json={"model": "text-embedding-3-small", "baseUrl": _REMOTE_BASE},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["available"] is True
    assert data["dimension"] == 4
    assert data["message"] == "测试成功，向量维度=4"


async def test_embedding_test_rejects_aliyun_multimodal(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/embedding/test",
        json={
            "model": "multimodal-embedding-v1",
            "baseUrl": _REMOTE_BASE,
            "provider": "aliyun",
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["available"] is False
    assert data["dimension"] == 0
    assert "暂不支持" in data["message"]


async def test_embedding_test_failure_is_200_with_zero_dimension(
    task_store: DownloadTaskStore,
) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    app = _build_app(task_store=task_store, remote=failing)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/initialization/embedding/test",
            json={"model": "m", "baseUrl": _REMOTE_BASE},
        )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["available"] is False
    assert data["dimension"] == 0


async def test_embedding_test_sends_dimensions_only_when_override_supported(
    task_store: DownloadTaskStore,
) -> None:
    captured: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 8}]})

    app = _build_app(task_store=task_store, remote=capture)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        await c.post(
            "/initialization/embedding/test",
            json={"model": "m", "baseUrl": _REMOTE_BASE, "dimension": 8},
        )
        await c.post(
            "/initialization/embedding/test",
            json={
                "model": "m",
                "baseUrl": _REMOTE_BASE,
                "dimension": 8,
                "supportsDimensionOverride": True,
            },
        )
    assert b"dimensions" not in captured[0].content
    assert b"dimensions" in captured[1].content


# ── POST /initialization/rerank/check ────────────────────────────────


async def test_rerank_check_counts_results(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/rerank/check",
        json={"model": "bge-reranker", "baseUrl": _REMOTE_BASE},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["available"] is True
    assert data["message"] == "重排功能正常，返回1个结果"


async def test_rerank_check_empty_results_is_unavailable(
    task_store: DownloadTaskStore,
) -> None:
    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    app = _build_app(task_store=task_store, remote=empty)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/initialization/rerank/check",
            json={"model": "m", "baseUrl": _REMOTE_BASE},
        )
    data = resp.json()["data"]
    assert data["available"] is False
    assert data["message"] == "重排接口连接成功，但未返回重排结果"


async def test_rerank_check_requires_base_url(client: AsyncClient) -> None:
    resp = await client.post("/initialization/rerank/check", json={"model": "m"})
    assert resp.status_code == 422


# ── POST /initialization/asr/check ───────────────────────────────────


async def test_asr_check_reports_transcript(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/asr/check",
        json={"model": "whisper-1", "baseUrl": _REMOTE_BASE},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["available"] is True
    assert data["message"] == "ASR连接成功，转写结果: hello"


async def test_asr_check_auth_failure_is_unavailable(
    task_store: DownloadTaskStore,
) -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    app = _build_app(task_store=task_store, remote=unauthorized)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/initialization/asr/check",
            json={"model": "m", "baseUrl": _REMOTE_BASE},
        )
    data = resp.json()["data"]
    assert data["available"] is False
    assert data["message"].startswith("认证失败，请检查API Key：")


async def test_asr_check_non_fatal_error_still_reachable(
    task_store: DownloadTaskStore,
) -> None:
    # Go: anything outside the fatal classes proves the endpoint answered.
    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal decoding failure")

    app = _build_app(task_store=task_store, remote=server_error)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/initialization/asr/check",
            json={"model": "m", "baseUrl": _REMOTE_BASE},
        )
    data = resp.json()["data"]
    assert data["available"] is True
    assert "ASR端点可达" in data["message"]


# ── POST /initialization/multimodal/test ─────────────────────────────


def _image_upload() -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("probe.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "image/png")}


async def test_multimodal_rejects_invalid_storage_type(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/multimodal/test",
        files=_image_upload(),
        data={
            "vlm_model": "qwen-vl",
            "vlm_base_url": _REMOTE_BASE,
            "storage_type": "s3",
        },
    )
    assert resp.status_code == 422


async def test_multimodal_rejects_incomplete_cos_config(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/multimodal/test",
        files=_image_upload(),
        data={
            "vlm_model": "qwen-vl",
            "vlm_base_url": _REMOTE_BASE,
            "storage_type": "cos",
            "cos_secret_id": "id",
        },
    )
    assert resp.status_code == 422


async def test_multimodal_rejects_non_image_upload(client: AsyncClient) -> None:
    resp = await client.post(
        "/initialization/multimodal/test",
        files={"image": ("notes.txt", b"plain text", "text/plain")},
        data={
            "vlm_model": "qwen-vl",
            "vlm_base_url": _REMOTE_BASE,
            "storage_type": "minio",
            "minio_bucket_name": "bucket",
        },
    )
    assert resp.status_code == 422


async def test_multimodal_valid_request_reports_docreader_gap(
    client: AsyncClient,
) -> None:
    # DocReader is a downstream dependency: the upstream handler
    # returns data.success=false with this message when the reader
    # is unset.
    resp = await client.post(
        "/initialization/multimodal/test",
        files=_image_upload(),
        data={
            "vlm_model": "qwen-vl",
            "vlm_base_url": _REMOTE_BASE,
            "storage_type": "minio",
            "minio_bucket_name": "bucket",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["success"] is False
    assert body["data"]["message"] == "DocReader service not configured"
