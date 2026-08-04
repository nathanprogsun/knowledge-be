"""Wire-shape conversion for the initialization endpoints.

Go returns bare ``gin.H`` maps rather than typed structs, so each
envelope here is a typed reconstruction of one of those maps. Field names
(including the camelCase JSON aliases) come from the ``c.JSON`` literals
in ``internal/handler/initialization.go`` and must not drift.

The download-task envelopes flatten ``ModelProbeResult.data`` into the
response body because Go puts ``taskId`` / ``modelName`` / ``status`` /
``progress`` directly under ``data``, alongside a sibling ``message`` on
the envelope itself.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.infra import OllamaModelInfo, OllamaStatusData
from src.core.infra.initialization.model_test import ModelProbeResult, MultimodalProbeResult
from src.core.infra.initialization.provider_detect import DownloadTaskInfo
from src.core.infra.initialization.service.initialization_service import DownloadStartResult


class OllamaStatusEnvelope(BaseModel):
    """``{"success": true, "data": {"available", "version", "baseUrl"}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OllamaStatusData


class OllamaModelsDataView(BaseModel):
    """``data`` of ``GET /initialization/ollama/models``."""

    model_config = ConfigDict(frozen=True)

    models: list[OllamaModelInfo] = Field(default_factory=list)


class OllamaModelsEnvelope(BaseModel):
    """``{"success": true, "data": {"models": [...]}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OllamaModelsDataView


class OllamaModelsCheckDataView(BaseModel):
    """``data`` of the models-check probe: name → installed."""

    model_config = ConfigDict(frozen=True)

    models: dict[str, bool] = Field(default_factory=dict)


class OllamaModelsCheckEnvelope(BaseModel):
    """``{"success": true, "data": {"models": {"<name>": bool}}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: OllamaModelsCheckDataView


class DownloadStartEnvelope(BaseModel):
    """``{"success", "message", "data": {...}}`` for the download trigger.

    ``data`` carries the camelCase keys Go emits (``taskId``,
    ``modelName``, ``status``, ``progress``) and varies by outcome —
    already-present responses omit ``taskId``, matching Go.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    data: JsonObject = Field(default_factory=dict)


class DownloadTaskView(BaseModel):
    """One task, serialized with Go's ``DownloadTask`` JSON names."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    id: str
    model_name: str = Field(serialization_alias="modelName", alias="modelName")
    status: str
    progress: float
    message: str
    start_time: datetime = Field(serialization_alias="startTime", alias="startTime")
    end_time: datetime | None = Field(default=None, serialization_alias="endTime", alias="endTime")


class DownloadProgressEnvelope(BaseModel):
    """``{"success": true, "data": <task>}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: DownloadTaskView


class DownloadTaskListEnvelope(BaseModel):
    """``{"success": true, "data": [<task>, ...]}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[DownloadTaskView] = Field(default_factory=list)


class ModelCheckDataView(BaseModel):
    """``{"available", "message"}`` — the shared model-probe payload."""

    model_config = ConfigDict(frozen=True)

    available: bool
    message: str


class ModelCheckEnvelope(BaseModel):
    """``{"success": true, "data": {"available", "message"}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: ModelCheckDataView


class EmbeddingTestDataView(ModelCheckDataView):
    """Embedding probe adds the resolved vector ``dimension``."""

    dimension: int = 0


class EmbeddingTestEnvelope(BaseModel):
    """``{"success": true, "data": {"available", "message", "dimension"}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbeddingTestDataView


class MultimodalTestDataView(BaseModel):
    """``data`` of the multimodal probe.

    The inner ``success`` is the probe outcome — distinct from the
    envelope's transport-level ``success``, which is always ``true``.
    ``message`` is only present on failure (Go omits it on success).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str = ""
    caption: str = ""
    ocr: str = ""
    processing_time: int = 0


class MultimodalTestEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` for the multimodal probe."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: MultimodalTestDataView


# ── Conversion helpers ───────────────────────────────────────────────


def ollama_status_envelope(data: OllamaStatusData) -> OllamaStatusEnvelope:
    return OllamaStatusEnvelope(success=True, data=data)


def ollama_models_envelope(models: list[OllamaModelInfo]) -> OllamaModelsEnvelope:
    return OllamaModelsEnvelope(success=True, data=OllamaModelsDataView(models=models))


def ollama_models_check_envelope(statuses: dict[str, bool]) -> OllamaModelsCheckEnvelope:
    return OllamaModelsCheckEnvelope(
        success=True,
        data=OllamaModelsCheckDataView(models=statuses),
    )


def download_start_envelope(result: DownloadStartResult) -> DownloadStartEnvelope:
    return DownloadStartEnvelope(success=True, message=result.message, data=result.data)


def _task_view(task: DownloadTaskInfo) -> DownloadTaskView:
    return DownloadTaskView(
        id=task.id,
        modelName=task.model_name,
        status=task.status,
        progress=task.progress,
        message=task.message,
        startTime=task.start_time,
        endTime=task.end_time,
    )


def download_progress_envelope(task: DownloadTaskInfo) -> DownloadProgressEnvelope:
    return DownloadProgressEnvelope(success=True, data=_task_view(task))


def download_task_list_envelope(tasks: list[DownloadTaskInfo]) -> DownloadTaskListEnvelope:
    return DownloadTaskListEnvelope(success=True, data=[_task_view(t) for t in tasks])


def model_check_envelope(result: ModelProbeResult) -> ModelCheckEnvelope:
    return ModelCheckEnvelope(
        success=True,
        data=ModelCheckDataView(available=result.available, message=result.message),
    )


def embedding_test_envelope(result: ModelProbeResult) -> EmbeddingTestEnvelope:
    raw_dimension = result.data.get("dimension")
    dimension = raw_dimension if isinstance(raw_dimension, int) else 0
    return EmbeddingTestEnvelope(
        success=True,
        data=EmbeddingTestDataView(
            available=result.available,
            message=result.message,
            dimension=dimension,
        ),
    )


def multimodal_test_envelope(result: MultimodalProbeResult) -> MultimodalTestEnvelope:
    return MultimodalTestEnvelope(
        success=True,
        data=MultimodalTestDataView(
            success=result.success,
            message=result.message,
            caption=result.caption,
            ocr=result.ocr,
            processing_time=result.processing_time,
        ),
    )


__all__ = [
    "DownloadProgressEnvelope",
    "DownloadStartEnvelope",
    "DownloadTaskListEnvelope",
    "DownloadTaskView",
    "EmbeddingTestEnvelope",
    "ModelCheckEnvelope",
    "MultimodalTestEnvelope",
    "OllamaModelsCheckEnvelope",
    "OllamaModelsEnvelope",
    "OllamaStatusEnvelope",
    "download_progress_envelope",
    "download_start_envelope",
    "download_task_list_envelope",
    "embedding_test_envelope",
    "model_check_envelope",
    "multimodal_test_envelope",
    "ollama_models_check_envelope",
    "ollama_models_envelope",
    "ollama_status_envelope",
]
