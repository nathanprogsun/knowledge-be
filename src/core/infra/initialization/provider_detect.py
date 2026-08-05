"""Ollama probing + remote provider connectivity detection.

Mirrors the Ollama helpers inlined in WeKnora's
``internal/handler/initialization.go`` (``CheckOllamaStatus``,
``ListOllamaModels``, ``CheckOllamaModels``, ``DownloadOllamaModel``,
``GetDownloadProgress``, ``ListDownloadTasks``) plus
``internal/models/utils/ollama/ollama.go`` (``StartService``,
``GetVersion``, ``IsModelAvailable``, ``ListModelsDetailed``) and the
chat-connection probe (``checkChatModelConnection`` /
``classifyConnectionError``).

Go talks to Ollama through the official Go SDK; there is no Python
equivalent in the dependency set, so this module speaks the documented
Ollama REST API directly (``GET /api/version``, ``GET /api/tags``,
``POST /api/pull``) over ``httpx``. Request/response field names are the
Ollama wire names, so behaviour is identical.

The operator-facing ``message`` strings are intentionally Chinese: they
are surfaced verbatim by the web layer and must match the Go strings the
frontend already renders. RUF001 (ambiguous full-width punctuation) is
suppressed file-wide for that reason.
"""
# ruff: noqa: RUF001, RUF002

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import lru_cache

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ExternalServiceError
from src.core.contracts.infra import (
    ModelTestRequest,
    OllamaModelInfo,
    OllamaStatusData,
)

# Go's ``CheckOllamaStatus`` displays this fallback when OLLAMA_BASE_URL is
# unset; ``GetOllamaService`` dials ``http://localhost:11434`` instead. Both
# defaults are preserved verbatim rather than unified.
OLLAMA_DISPLAY_FALLBACK_URL = "http://host.docker.internal:11434"
OLLAMA_DIAL_FALLBACK_URL = "http://localhost:11434"

OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
OLLAMA_OPTIONAL_ENV = "OLLAMA_OPTIONAL"

_UNKNOWN_VERSION = "unknown"
_DEFAULT_TIMEOUT_SECONDS = 30.0
# Go caps async pulls at 12h via context.WithTimeout.
PULL_TIMEOUT_SECONDS = 12 * 60 * 60

STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED})
ACTIVE_STATUSES = frozenset({STATUS_PENDING, STATUS_DOWNLOADING})


def resolve_ollama_display_base_url() -> str:
    """Base URL echoed back to the operator by the status endpoint."""
    return os.environ.get(OLLAMA_BASE_URL_ENV) or OLLAMA_DISPLAY_FALLBACK_URL


def resolve_ollama_dial_base_url() -> str:
    """Base URL the HTTP client actually dials."""
    return os.environ.get(OLLAMA_BASE_URL_ENV) or OLLAMA_DIAL_FALLBACK_URL


def is_ollama_optional() -> bool:
    """``OLLAMA_OPTIONAL=true`` downgrades an unreachable Ollama to a warning."""
    return os.environ.get(OLLAMA_OPTIONAL_ENV) == "true"


# ── Download tasks ───────────────────────────────────────────────────


class DownloadTaskInfo(BaseModel):
    """One async model-pull task (Go ``handler.DownloadTask``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    model_name: str
    status: str
    progress: float = 0.0
    message: str = ""
    start_time: datetime
    end_time: datetime | None = Field(default=None)


class DownloadTaskStore:
    """In-memory registry of model-pull tasks.

    Go keeps a package-level ``map[string]*DownloadTask`` guarded by a
    mutex. Here the state is encapsulated in an object so it can be
    injected (and reset in tests) instead of being a module global. Task
    records are replaced, never mutated, so a concurrent reader always
    sees a consistent snapshot without locking.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, DownloadTaskInfo] = {}
        # Strong references to the in-flight pull coroutines: asyncio only
        # holds a weak reference, so an unreferenced task can be collected
        # mid-flight.
        self._pulls: set[asyncio.Task[None]] = set()

    def retain(self, pull: asyncio.Task[None]) -> None:
        """Keep ``pull`` alive until it finishes, then drop the reference."""
        self._pulls.add(pull)
        pull.add_done_callback(self._pulls.discard)

    async def wait_for_pulls(self) -> None:
        """Await every in-flight pull. Used by tests and graceful shutdown."""
        pending = tuple(self._pulls)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def create(self, *, task_id: str, model_name: str) -> DownloadTaskInfo:
        """Register a fresh ``pending`` task and return it."""
        task = DownloadTaskInfo(
            id=task_id,
            model_name=model_name,
            status=STATUS_PENDING,
            progress=0.0,
            message="准备下载",
            start_time=datetime.now(UTC),
        )
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> DownloadTaskInfo | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[DownloadTaskInfo]:
        return list(self._tasks.values())

    def find_active(self, model_name: str) -> DownloadTaskInfo | None:
        """Return an in-flight task for ``model_name``, if any."""
        for task in self._tasks.values():
            if task.model_name == model_name and task.status in ACTIVE_STATUSES:
                return task
        return None

    def update_status(
        self,
        task_id: str,
        *,
        status: str,
        progress: float,
        message: str,
    ) -> DownloadTaskInfo | None:
        """Replace the task record with an updated copy (never mutate)."""
        existing = self._tasks.get(task_id)
        if existing is None:
            return None
        end_time = datetime.now(UTC) if status in _TERMINAL_STATUSES else existing.end_time
        updated = existing.model_copy(
            update={
                "status": status,
                "progress": progress,
                "message": message,
                "end_time": end_time,
            }
        )
        self._tasks[task_id] = updated
        return updated


@lru_cache(maxsize=1)
def get_download_task_store() -> DownloadTaskStore:
    """Process-wide task store, memoized like ``get_settings()``.

    Download progress must outlive the request that started the pull, so
    the store cannot be request-scoped. It is not registered on
    ``LifeSpanService`` yet — promoting it to an explicit APP-scope
    singleton is checkpoint-2 work (the DI registry is off-limits here).
    """
    return DownloadTaskStore()


# ── Ollama REST client ───────────────────────────────────────────────

ProgressCallback = Callable[[float, str], Awaitable[None]]


class OllamaClient:
    """Minimal async client for the Ollama REST API.

    Tests inject an ``httpx.MockTransport`` via ``transport=`` exactly as
    with ``OidcClient``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._base_url = (base_url or resolve_ollama_dial_base_url()).rstrip("/")
        if transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout)
        else:
            self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        return self._base_url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def heartbeat(self) -> None:
        """Go ``client.Heartbeat`` — raise when Ollama is unreachable."""
        try:
            response = await self._client.get(f"{self._base_url}/")
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"ollama service unavailable: {exc}",
                code="ollama.unavailable",
            ) from exc
        if response.status_code >= 400:
            raise ExternalServiceError(
                f"ollama service unavailable: HTTP {response.status_code}",
                code="ollama.unavailable",
            )

    async def get_version(self) -> str:
        """Go ``GetVersion`` — ``unknown`` when the probe fails."""
        try:
            response = await self._client.get(f"{self._base_url}/api/version")
            if response.status_code >= 400:
                return _UNKNOWN_VERSION
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return _UNKNOWN_VERSION
        if not isinstance(payload, dict):
            return _UNKNOWN_VERSION
        version = payload.get("version")
        return version if isinstance(version, str) and version else _UNKNOWN_VERSION

    async def list_models_detailed(self) -> list[OllamaModelInfo]:
        """Go ``ListModelsDetailed`` — ``GET /api/tags``."""
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to get model list: {exc}",
                code="ollama.list_models_failed",
            ) from exc
        if not isinstance(payload, dict):
            return []
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            return []
        return [
            OllamaModelInfo.model_validate(entry) for entry in raw_models if isinstance(entry, dict)
        ]

    async def pull_model(self, model_name: str, on_progress: ProgressCallback) -> None:
        """Go ``pullModelWithProgress`` — streamed ``POST /api/pull``."""
        payload = {"name": model_name, "stream": True}
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json=payload,
                timeout=PULL_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    progress, message = _parse_pull_progress(line)
                    if message:
                        await on_progress(progress, message)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to pull model: {exc}",
                code="ollama.pull_failed",
            ) from exc


def _parse_pull_progress(line: str) -> tuple[float, str]:
    """Map one ``/api/pull`` NDJSON frame onto (percent, message).

    Mirrors the Go progress callback: percent is ``completed/total*100``
    when both counters are present, otherwise the raw ``status`` string is
    surfaced with no percentage.
    """
    stripped = line.strip()
    if not stripped:
        return 0.0, ""
    try:
        frame = json.loads(stripped)
    except json.JSONDecodeError:
        return 0.0, ""
    if not isinstance(frame, dict):
        return 0.0, ""
    status = frame.get("status")
    status_text = status if isinstance(status, str) else ""
    total = frame.get("total")
    completed = frame.get("completed")
    if isinstance(total, int) and isinstance(completed, int) and total > 0 and completed > 0:
        percent = completed / total * 100
        return percent, f"下载中: {percent:.1f}% ({status_text})"
    if status_text:
        return 0.0, status_text
    return 0.0, ""


def normalize_model_tag(model_name: str) -> str:
    """Go ``IsModelAvailable``: an untagged name means ``:latest``."""
    return model_name if ":" in model_name else f"{model_name}:latest"


def build_status_data(*, available: bool, version: str | None, base_url: str) -> OllamaStatusData:
    """Assemble the frozen status contract."""
    return OllamaStatusData(available=available, version=version, baseUrl=base_url)


# ── Remote provider detection ────────────────────────────────────────

_UNAUTHORIZED_HINT = "认证失败，请检查API Key"
_FORBIDDEN_HINT = "权限不足，请检查API Key权限"
_NOT_FOUND_HINT = "API端点不存在，请检查Base URL"
_TIMEOUT_HINT = "连接超时，请检查网络连接"
_UNREACHABLE_HINT = "无法连接到服务器，请检查Base URL"
_GENERIC_HINT = "连接失败"

CHAT_OK_MESSAGE = "连接正常，模型可用"


def classify_connection_error(error_message: str) -> str:
    """Go ``classifyConnectionError`` — short Chinese hint for an error string."""
    lowered = error_message.lower()
    if "401" in lowered or "unauthorized" in lowered:
        return _UNAUTHORIZED_HINT
    if "403" in lowered or "forbidden" in lowered:
        return _FORBIDDEN_HINT
    if "404" in lowered or "not found" in lowered:
        return _NOT_FOUND_HINT
    if "timeout" in lowered or "context deadline exceeded" in lowered:
        return _TIMEOUT_HINT
    if (
        "connection refused" in lowered
        or "no such host" in lowered
        or "dial tcp" in lowered
        or "connect" in lowered
    ):
        return _UNREACHABLE_HINT
    return _GENERIC_HINT


def build_auth_headers(request: ModelTestRequest) -> dict[str, str]:
    """Bearer auth plus the caller's ``customHeaders`` passthrough."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if request.api_key:
        headers["Authorization"] = f"Bearer {request.api_key}"
    if request.custom_headers:
        headers.update(request.custom_headers)
    return headers


async def check_chat_model_connection(
    request: ModelTestRequest,
    *,
    client: httpx.AsyncClient,
) -> tuple[bool, str]:
    """Go ``checkChatModelConnection`` — one minimal chat completion.

    A ``400`` means the endpoint was reached and authenticated and only
    a parameter was rejected, so Go treats it as success; every other
    failure returns ``"<hint>：<raw error>"``.
    """
    base_url = (request.base_url or "").rstrip("/")
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
    }
    try:
        response = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=build_auth_headers(request),
        )
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        return False, f"{classify_connection_error(detail)}：{detail}"
    if response.status_code == 400:
        return True, CHAT_OK_MESSAGE
    if response.status_code >= 400:
        detail = f"status code: {response.status_code}, body: {response.text}"
        return False, f"{classify_connection_error(detail)}：{detail}"
    return True, CHAT_OK_MESSAGE


__all__ = [
    "ACTIVE_STATUSES",
    "CHAT_OK_MESSAGE",
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_DIAL_FALLBACK_URL",
    "OLLAMA_DISPLAY_FALLBACK_URL",
    "OLLAMA_OPTIONAL_ENV",
    "PULL_TIMEOUT_SECONDS",
    "STATUS_COMPLETED",
    "STATUS_DOWNLOADING",
    "STATUS_FAILED",
    "STATUS_PENDING",
    "DownloadTaskInfo",
    "DownloadTaskStore",
    "OllamaClient",
    "ProgressCallback",
    "build_auth_headers",
    "build_status_data",
    "check_chat_model_connection",
    "classify_connection_error",
    "get_download_task_store",
    "is_ollama_optional",
    "normalize_model_tag",
    "resolve_ollama_dial_base_url",
    "resolve_ollama_display_base_url",
]
