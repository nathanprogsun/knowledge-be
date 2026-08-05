"""Initialization service — startup bootstrap + Ollama/remote detection.

WeKnora has no standalone ``initialization_service.go``: the logic lives
inline in ``internal/handler/initialization.go``. This service extracts
the non-HTTP parts of those handlers so the web layer stays declarative:

- ``check_ollama_status``    ← ``CheckOllamaStatus``
- ``list_ollama_models``     ← ``ListOllamaModels``
- ``check_ollama_models``    ← ``CheckOllamaModels``
- ``download_ollama_model``  ← ``DownloadOllamaModel``
- ``get_download_progress``  ← ``GetDownloadProgress``
- ``list_download_tasks``    ← ``ListDownloadTasks``
- ``check_remote_model``     ← ``CheckRemoteModel``

Everything the KB-scoped handlers do (``UpdateKBConfig``,
``InitializeByKB``, ``GetCurrentConfigByKB``) belongs to the knowledge
domain and is deliberately absent.

Operator-facing ``message`` strings are Chinese so they match the Go
strings the frontend renders verbatim; RUF001 is suppressed file-wide.
"""

from __future__ import annotations

import asyncio
import uuid

import httpx

from src.app_logging import logger
from src.common.exception import ExternalServiceError, NotFoundError, ValidationError
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.contracts.infra import (
    ModelTestRequest,
    OllamaModelInfo,
    OllamaStatusData,
)
from src.core.infra.initialization.model_test import (
    ModelProbeResult,
    MultimodalProbeResult,
    MultimodalTestConfig,
)
from src.core.infra.initialization.model_test import check_asr_model as probe_asr_model
from src.core.infra.initialization.model_test import check_rerank_model as probe_rerank_model
from src.core.infra.initialization.model_test import test_embedding_model as probe_embedding_model
from src.core.infra.initialization.model_test import (
    test_multimodal_function as probe_multimodal_function,
)
from src.core.infra.initialization.provider_detect import (
    STATUS_COMPLETED,
    STATUS_DOWNLOADING,
    STATUS_FAILED,
    DownloadTaskInfo,
    DownloadTaskStore,
    OllamaClient,
    build_status_data,
    check_chat_model_connection,
    is_ollama_optional,
    normalize_model_tag,
    resolve_ollama_display_base_url,
)

_MODEL_ALREADY_PRESENT = "模型已存在"
_TASK_ALREADY_EXISTS = "模型下载任务已存在"
_TASK_CREATED = "模型下载任务已创建"
_DOWNLOAD_STARTED = "开始下载模型"
_DOWNLOAD_DONE = "下载完成"
_PROGRESS_COMPLETE = 100.0


class DownloadStartResult(ModelProbeResult):
    """Outcome of a download request (created / already running / present)."""


class InitializationService:
    """First-run bootstrap probes for Ollama and remote model providers.

    Request-scoped per AGENTS.md §3: it binds an ``httpx.AsyncClient``
    supplied by the factory. The ``DownloadTaskStore`` is process-wide
    because pull progress must outlive the request that started it.
    """

    def __init__(
        self,
        *,
        ollama_client: OllamaClient,
        task_store: DownloadTaskStore,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._ollama = ollama_client
        self._tasks = task_store
        self._http = http_client

    # ── Ollama status / models ───────────────────────────────────────

    async def check_ollama_status(self) -> OllamaStatusData:
        """Heartbeat + version probe. Never raises: unavailability is data.

        Go returns ``available: false`` with the error text in the body
        rather than a non-200, so the frontend can render a hint.
        """
        base_url = resolve_ollama_display_base_url()
        try:
            await self._ollama.heartbeat()
        except ExternalServiceError as exc:
            logger.warning("ollama heartbeat failed: {}", exc.message)
            return build_status_data(available=False, version=None, base_url=base_url)
        version = await self._ollama.get_version()
        return build_status_data(available=True, version=version, base_url=base_url)

    async def _ensure_available(self) -> None:
        """Go's ``if !IsAvailable() { StartService() }`` preamble.

        ``OLLAMA_OPTIONAL=true`` downgrades the failure to a warning, as
        in ``OllamaService.StartService``.
        """
        try:
            await self._ollama.heartbeat()
        except ExternalServiceError as exc:
            if is_ollama_optional():
                logger.info("ollama optional mode: continuing without ollama")
                return
            raise ExternalServiceError(
                f"Ollama服务不可用: {exc.message}",
                code="ollama.unavailable",
            ) from exc

    async def list_ollama_models(self) -> list[OllamaModelInfo]:
        """Installed models with size/digest detail (Go ``ListOllamaModels``)."""
        await self._ensure_available()
        return await self._ollama.list_models_detailed()

    async def check_ollama_models(self, model_names: list[str]) -> dict[str, bool]:
        """Per-name availability map (Go ``CheckOllamaModels``).

        A probe failure for one name records ``False`` rather than
        aborting the batch — matching Go's per-model error handling.
        """
        await self._ensure_available()
        try:
            installed = await self._ollama.list_models_detailed()
        except ExternalServiceError as exc:
            logger.warning("ollama model listing failed: {}", exc.message)
            return dict.fromkeys(model_names, False)
        installed_tags = {model.name for model in installed}
        return {name: normalize_model_tag(name) in installed_tags for name in model_names}

    # ── Ollama downloads ─────────────────────────────────────────────

    async def download_ollama_model(self, model_name: str) -> DownloadStartResult:
        """Kick off an async pull, deduplicating against present/in-flight work."""
        await self._ensure_available()

        statuses = await self.check_ollama_models([model_name])
        if statuses.get(model_name, False):
            return DownloadStartResult(
                available=True,
                message=_MODEL_ALREADY_PRESENT,
                data={
                    "modelName": model_name,
                    "status": STATUS_COMPLETED,
                    "progress": _PROGRESS_COMPLETE,
                },
            )

        existing = self._tasks.find_active(model_name)
        if existing is not None:
            return DownloadStartResult(
                available=True,
                message=_TASK_ALREADY_EXISTS,
                data={
                    "taskId": existing.id,
                    "modelName": existing.model_name,
                    "status": existing.status,
                    "progress": existing.progress,
                },
            )

        task = self._tasks.create(task_id=str(uuid.uuid4()), model_name=model_name)
        # Detached: the pull outlives this request (Go spawns a goroutine
        # with a 12h context). The store holds a strong reference so the
        # task is not garbage-collected mid-flight.
        self._tasks.retain(asyncio.create_task(self._download_async(task.id, model_name)))
        return DownloadStartResult(
            available=True,
            message=_TASK_CREATED,
            data={
                "taskId": task.id,
                "modelName": task.model_name,
                "status": task.status,
                "progress": task.progress,
            },
        )

    async def _download_async(self, task_id: str, model_name: str) -> None:
        """Go ``downloadModelAsync`` — drive the pull and record progress."""
        self._tasks.update_status(
            task_id,
            status=STATUS_DOWNLOADING,
            progress=0.0,
            message=_DOWNLOAD_STARTED,
        )

        async def on_progress(progress: float, message: str) -> None:
            self._tasks.update_status(
                task_id,
                status=STATUS_DOWNLOADING,
                progress=progress,
                message=message,
            )

        try:
            await self._ollama.pull_model(model_name, on_progress)
        except ExternalServiceError as exc:
            logger.error("ollama pull failed for task {}: {}", task_id, exc.message)
            self._tasks.update_status(
                task_id,
                status=STATUS_FAILED,
                progress=0.0,
                message=f"下载失败: {exc.message}",
            )
            return
        self._tasks.update_status(
            task_id,
            status=STATUS_COMPLETED,
            progress=_PROGRESS_COMPLETE,
            message=_DOWNLOAD_DONE,
        )

    def get_download_progress(self, task_id: str) -> DownloadTaskInfo:
        """Go ``GetDownloadProgress`` — 400 on empty id, 404 when unknown."""
        if not task_id:
            raise ValidationError("任务ID不能为空", code="initialization.task_id_required")
        task = self._tasks.get(task_id)
        if task is None:
            raise NotFoundError("下载任务不存在", code="initialization.task_not_found")
        return task

    def list_download_tasks(self) -> list[DownloadTaskInfo]:
        """Go ``ListDownloadTasks`` — every known task, no filtering."""
        return self._tasks.list_all()

    # ── Remote provider detection ────────────────────────────────────

    async def check_remote_model(self, request: ModelTestRequest) -> ModelProbeResult:
        """Go ``CheckRemoteModel`` — minimal chat call against a remote endpoint.

        ``modelName`` and ``baseUrl`` are both mandatory and the URL is
        SSRF-validated before any request leaves the process.
        """
        if not request.model or not request.base_url:
            raise ValidationError(
                "模型名称和Base URL不能为空",
                code="initialization.model_and_base_url_required",
            )
        await validate_ssrf_safe_url(request.base_url)
        available, message = await check_chat_model_connection(request, client=self._http)
        return ModelProbeResult(available=available, message=message)

    # ── Model probes ──────────────────────────────────────────────────
    #
    # Thin delegations to ``model_test``: the probe logic is a pure
    # function of (request, client), and routing it through the service
    # keeps the shared outbound client out of the web layer.

    async def test_embedding_model(self, request: ModelTestRequest) -> ModelProbeResult:
        """Go ``TestEmbeddingModel``."""
        return await probe_embedding_model(request, client=self._http)

    async def check_rerank_model(self, request: ModelTestRequest) -> ModelProbeResult:
        """Go ``CheckRerankModel``."""
        return await probe_rerank_model(request, client=self._http)

    async def check_asr_model(self, request: ModelTestRequest) -> ModelProbeResult:
        """Go ``CheckASRModel``."""
        return await probe_asr_model(request, client=self._http)

    async def test_multimodal_function(
        self,
        config: MultimodalTestConfig,
        *,
        content_type: str,
        size: int,
    ) -> MultimodalProbeResult:
        """Go ``TestMultimodalFunction``."""
        return await probe_multimodal_function(config, content_type=content_type, size=size)


__all__ = ["DownloadStartResult", "InitializationService"]
