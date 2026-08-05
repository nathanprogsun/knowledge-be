"""Initialization HTTP endpoints — Ollama detection + model probes.

Maps the non-KB initialization endpoints from
``internal/router/routes_infra.go`` / ``internal/handler/initialization.go``:

- ``GET  /initialization/ollama/status``                    — Viewer
- ``GET  /initialization/ollama/models``                    — Viewer
- ``POST /initialization/ollama/models/check``              — Admin
- ``POST /initialization/ollama/models/download``           — Admin
- ``GET  /initialization/ollama/download/progress/{taskId}`` — Viewer
- ``GET  /initialization/ollama/download/tasks``            — Viewer
- ``POST /initialization/remote/check``                     — Admin
- ``POST /initialization/embedding/test``                   — Admin
- ``POST /initialization/rerank/check``                     — Admin
- ``POST /initialization/asr/check``                        — Admin
- ``POST /initialization/multimodal/test``                  — Admin

The KB-scoped routes (``/initialization/config/{kbId}``,
``/initialization/initialize/{kbId}``) belong to the knowledge domain
and are intentionally absent, as are the three
``/initialization/extract/*`` routes (they need the chat/LLM layer).

Every route carries ``AuthDep`` plus its role gate, matching the
``g.Viewer()`` / ``g.Admin()`` argument on the Go route registration.
Probe failures are reported as ``available: false`` inside a 200 body —
Go never turns an unreachable provider into a 5xx — so only invalid input
produces a 4xx.

Swagger ``description`` strings are Chinese, mirroring the upstream Go
annotations; RUF001 is suppressed for the same reason as in
``src/web/api/system/router.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from src.core.contracts.infra import (
    CheckOllamaModelsRequest,
    DownloadOllamaModelRequest,
    ModelTestRequest,
)
from src.core.infra.initialization.model_test import MultimodalTestConfig
from src.web.api.infra.initialization.views import (
    DownloadProgressEnvelope,
    DownloadStartEnvelope,
    DownloadTaskListEnvelope,
    EmbeddingTestEnvelope,
    ModelCheckEnvelope,
    MultimodalTestEnvelope,
    OllamaModelsCheckEnvelope,
    OllamaModelsEnvelope,
    OllamaStatusEnvelope,
    download_progress_envelope,
    download_start_envelope,
    download_task_list_envelope,
    embedding_test_envelope,
    model_check_envelope,
    multimodal_test_envelope,
    ollama_models_check_envelope,
    ollama_models_envelope,
    ollama_status_envelope,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.infra_initialization import InitializationServiceDep

router = APIRouter(prefix="/initialization", tags=["initialization"])

# Multipart parameter aliases. Declared as ``Annotated`` types rather than
# call-in-default (``x: str = Form(...)``) so the marker objects are
# module-level singletons — the pattern flake8-bugbear's B008 asks for.
ImageUpload = Annotated[UploadFile, File(description="测试图片")]
FormField = Annotated[str, Form()]


# ── Ollama detection ─────────────────────────────────────────────────


@router.get("/ollama/status", response_model=OllamaStatusEnvelope)
async def check_ollama_status(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    init_svc: InitializationServiceDep,
) -> OllamaStatusEnvelope:
    """Report whether the local Ollama service answers, plus its version.

    Never fails: an unreachable Ollama returns ``available: false`` so the
    setup wizard can render a hint instead of an error page.
    """
    return ollama_status_envelope(await init_svc.check_ollama_status())


@router.get("/ollama/models", response_model=OllamaModelsEnvelope)
async def list_ollama_models(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    init_svc: InitializationServiceDep,
) -> OllamaModelsEnvelope:
    """List installed Ollama models with size / digest / modified_at."""
    return ollama_models_envelope(await init_svc.list_ollama_models())


@router.post("/ollama/models/check", response_model=OllamaModelsCheckEnvelope)
async def check_ollama_models(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: CheckOllamaModelsRequest,
    init_svc: InitializationServiceDep,
) -> OllamaModelsCheckEnvelope:
    """Report per-name whether each requested model is installed.

    A name without an explicit tag is resolved against ``:latest``, and a
    probe failure for one name records ``false`` rather than failing the
    batch.
    """
    return ollama_models_check_envelope(await init_svc.check_ollama_models(list(body.models)))


@router.post("/ollama/models/download", response_model=DownloadStartEnvelope)
async def download_ollama_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: DownloadOllamaModelRequest,
    init_svc: InitializationServiceDep,
) -> DownloadStartEnvelope:
    """Start an async model pull, or report an existing one.

    Three outcomes, all 200: the model is already installed, a pull is
    already in flight (returns its ``taskId``), or a new task was created.
    """
    return download_start_envelope(await init_svc.download_ollama_model(body.model_name))


@router.get(
    "/ollama/download/progress/{task_id}",
    response_model=DownloadProgressEnvelope,
)
async def get_download_progress(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    task_id: str,
    init_svc: InitializationServiceDep,
) -> DownloadProgressEnvelope:
    """Progress of one pull task. 404 when the task id is unknown."""
    return download_progress_envelope(init_svc.get_download_progress(task_id))


@router.get("/ollama/download/tasks", response_model=DownloadTaskListEnvelope)
async def list_download_tasks(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    init_svc: InitializationServiceDep,
) -> DownloadTaskListEnvelope:
    """Every known pull task, including finished and failed ones."""
    return download_task_list_envelope(init_svc.list_download_tasks())


# ── Model probes ─────────────────────────────────────────────────────


@router.post("/remote/check", response_model=ModelCheckEnvelope)
async def check_remote_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ModelTestRequest,
    init_svc: InitializationServiceDep,
) -> ModelCheckEnvelope:
    """Probe a remote chat endpoint with one minimal completion.

    ``modelName`` and ``baseUrl`` are mandatory (422 otherwise) and the
    base URL is SSRF-validated before any request leaves the process.
    """
    return model_check_envelope(await init_svc.check_remote_model(body))


@router.post("/embedding/test", response_model=EmbeddingTestEnvelope)
async def test_embedding_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ModelTestRequest,
    init_svc: InitializationServiceDep,
) -> EmbeddingTestEnvelope:
    """Embed a probe string and return the resolved vector dimension.

    Aliyun vision / multimodal embedding models are rejected up front
    with ``available: false`` (unsupported upstream too).
    """
    return embedding_test_envelope(await init_svc.test_embedding_model(body))


@router.post("/rerank/check", response_model=ModelCheckEnvelope)
async def check_rerank_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ModelTestRequest,
    init_svc: InitializationServiceDep,
) -> ModelCheckEnvelope:
    """Rerank a single probe document and report the result count."""
    return model_check_envelope(await init_svc.check_rerank_model(body))


@router.post("/asr/check", response_model=ModelCheckEnvelope)
async def check_asr_model(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: ModelTestRequest,
    init_svc: InitializationServiceDep,
) -> ModelCheckEnvelope:
    """Transcribe a short silent WAV to probe the transcriptions endpoint.

    Only a fatal error class (auth, missing endpoint, unreachable host,
    unknown model) marks the endpoint unavailable — anything else means it
    was reached, which is what the probe is testing.
    """
    return model_check_envelope(await init_svc.check_asr_model(body))


@router.post("/multimodal/test", response_model=MultimodalTestEnvelope)
async def test_multimodal_function(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    init_svc: InitializationServiceDep,
    image: ImageUpload,
    vlm_model: FormField = "",
    vlm_base_url: FormField = "",
    vlm_api_key: FormField = "",
    vlm_interface_type: FormField = "",
    storage_type: FormField = "",
    cos_secret_id: FormField = "",
    cos_secret_key: FormField = "",
    cos_region: FormField = "",
    cos_bucket_name: FormField = "",
    cos_app_id: FormField = "",
    cos_path_prefix: FormField = "",
    minio_bucket_name: FormField = "",
    minio_path_prefix: FormField = "",
) -> MultimodalTestEnvelope:
    """Upload an image and run it through the multimodal read path.

    Multipart, mirroring the Go handler's form binding. The VLM base URL
    is SSRF-validated, and the upload must be an image within
    ``MAX_FILE_SIZE_MB``.
    """
    config = MultimodalTestConfig(
        vlm_model=vlm_model,
        vlm_base_url=vlm_base_url,
        vlm_api_key=vlm_api_key,
        vlm_interface_type=vlm_interface_type,
        storage_type=storage_type,
        cos_secret_id=cos_secret_id,
        cos_secret_key=cos_secret_key,
        cos_region=cos_region,
        cos_bucket_name=cos_bucket_name,
        cos_app_id=cos_app_id,
        cos_path_prefix=cos_path_prefix,
        minio_bucket_name=minio_bucket_name,
        minio_path_prefix=minio_path_prefix,
    )
    content = await image.read()
    result = await init_svc.test_multimodal_function(
        config,
        content_type=image.content_type or "",
        size=len(content),
    )
    return multimodal_test_envelope(result)


__all__ = ["router"]
