"""Embedding / rerank / ASR / multimodal connectivity probes.

Mirrors the model-test handlers in
``internal/handler/initialization.go``:

- ``test_embedding_model``      ← ``TestEmbeddingModel``
- ``check_rerank_model``        ← ``CheckRerankModel``
- ``check_asr_model``           ← ``CheckASRModel``
- ``test_multimodal_function``  ← ``TestMultimodalFunction``

Go assembles a throwaway ``*types.Model`` and runs it through the same
``ConfigFromModel`` → provider-client path as production. The provider
client layer does not exist in Python yet, so these probes call
the OpenAI-compatible endpoints every provider in WeKnora ultimately
speaks (``POST {baseUrl}/embeddings``, ``POST {baseUrl}/rerank``,
``POST {baseUrl}/audio/transcriptions``, ``POST {baseUrl}/chat/completions``)
with the same request field names. Availability semantics, message
strings, and the "endpoint reached ⇒ available" heuristics match Go.

Operator-facing messages are Chinese to match the strings the frontend
renders; RUF001 is suppressed file-wide.
"""
# ruff: noqa: RUF001

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.contracts.infra import ModelTestRequest
from src.core.infra.initialization.provider_detect import (
    build_auth_headers,
    classify_connection_error,
    resolve_ollama_dial_base_url,
)

_ALIYUN_PROVIDER = "aliyun"
_ALIYUN_MULTIMODAL_MARKERS = ("vision", "multimodal")
_ALIYUN_MULTIMODAL_UNSUPPORTED = (
    "阿里云多模态 Embedding 模型暂不支持，请使用纯文本 Embedding 模型（如 text-embedding-v4）"
)

_EMBEDDING_PROBE_TEXT = "hello"
_RERANK_PROBE_QUERY = "ping"
_RERANK_PROBE_DOCUMENT = "pong"

_ASR_OK = "ASR连接成功"
_ASR_TEST_FILENAME = "asr_test.wav"

# 44-byte RIFF/WAVE header describing a zero-length 8 kHz mono PCM stream —
# the Python counterpart of Go's ``assets.ASRTestWAV``. Silence is enough to
# prove ``/audio/transcriptions`` is reachable and authenticated.
ASR_TEST_WAV: bytes = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)

_MULTIMODAL_STORAGE_COS = "cos"
_MULTIMODAL_STORAGE_MINIO = "minio"
_IMAGE_CONTENT_TYPE_PREFIX = "image/"
_MAX_FILE_SIZE_ENV = "MAX_FILE_SIZE_MB"
_DEFAULT_MAX_FILE_SIZE_MB = 50
_BYTES_PER_MB = 1024 * 1024
_DOCREADER_NOT_CONFIGURED = "DocReader service not configured"

_MODEL_AND_BASE_URL_REQUIRED = "模型名称和Base URL不能为空"


class ModelProbeResult(BaseModel):
    """``{"available", "message"}`` plus optional probe-specific extras.

    Go returns a bare ``gin.H`` per endpoint; the extras (``dimension``
    for embedding, download-task fields) live in ``data`` and are merged
    into the response body by the view layer so the wire shape is
    byte-identical to Go's.
    """

    model_config = ConfigDict(frozen=True)

    available: bool
    message: str
    data: JsonObject = Field(default_factory=dict)


class MultimodalProbeResult(BaseModel):
    """Result of ``POST /initialization/multimodal/test``.

    Go nests ``{"success", "caption", "ocr", "processing_time"}`` inside
    the envelope's ``data`` — note the inner ``success`` is the probe
    outcome, distinct from the envelope's transport-level ``success``.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str = ""
    caption: str = ""
    ocr: str = ""
    processing_time: int = 0


def _is_aliyun_multimodal_embedding(request: ModelTestRequest) -> bool:
    """Go rejects Aliyun vision/multimodal embedding models up front."""
    if (request.provider or "").lower() != _ALIYUN_PROVIDER:
        return False
    lowered = request.model.lower()
    return any(marker in lowered for marker in _ALIYUN_MULTIMODAL_MARKERS)


def _require_model_and_base_url(request: ModelTestRequest, *, code: str) -> str:
    """Validate the two mandatory fields and return the normalized base URL."""
    if not request.model or not request.base_url:
        raise ValidationError(_MODEL_AND_BASE_URL_REQUIRED, code=code)
    return request.base_url.rstrip("/")


def _embedding_payload(request: ModelTestRequest) -> JsonObject:
    """Go ``OpenAIEmbedder``: ``dimensions`` only when override is supported."""
    payload: JsonObject = {
        "model": request.model,
        "input": _EMBEDDING_PROBE_TEXT,
    }
    if request.supports_dimension_override and request.dimension:
        payload["dimensions"] = request.dimension
    return payload


def _extract_embedding_dimension(body: JsonValue) -> int:
    """Read ``data[0].embedding`` length from an OpenAI-shaped response."""
    if not isinstance(body, dict):
        return 0
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return 0
    first = data[0]
    if not isinstance(first, dict):
        return 0
    vector = first.get("embedding")
    if not isinstance(vector, list):
        return 0
    return len(vector)


async def test_embedding_model(
    request: ModelTestRequest,
    *,
    client: httpx.AsyncClient,
) -> ModelProbeResult:
    """Go ``TestEmbeddingModel`` — embed ``"hello"`` and report the dimension.

    Failures are reported as ``available: false`` with ``dimension: 0``
    inside a 200 response (Go never returns a 5xx here); only the two
    guard clauses (SSRF, unsupported Aliyun model) short-circuit.
    """
    if request.base_url:
        await validate_ssrf_safe_url(request.base_url)
    if _is_aliyun_multimodal_embedding(request):
        return ModelProbeResult(
            available=False,
            message=_ALIYUN_MULTIMODAL_UNSUPPORTED,
            data={"dimension": 0},
        )

    base_url = (request.base_url or "").rstrip("/")
    try:
        response = await client.post(
            f"{base_url}/embeddings",
            json=_embedding_payload(request),
            headers=build_auth_headers(request),
        )
    except httpx.HTTPError as exc:
        return ModelProbeResult(
            available=False,
            message=f"调用Embedding失败: {type(exc).__name__}: {exc}",
            data={"dimension": 0},
        )
    if response.status_code >= 400:
        detail = f"status code: {response.status_code}, body: {response.text}"
        return ModelProbeResult(
            available=False,
            message=f"调用Embedding失败: {classify_connection_error(detail)}：{detail}",
            data={"dimension": 0},
        )
    try:
        dimension = _extract_embedding_dimension(response.json())
    except ValueError as exc:
        return ModelProbeResult(
            available=False,
            message=f"调用Embedding失败: 响应解析失败: {exc}",
            data={"dimension": 0},
        )
    return ModelProbeResult(
        available=True,
        message=f"测试成功，向量维度={dimension}",
        data={"dimension": dimension},
    )


def _rerank_result_count(body: JsonValue) -> int:
    """Length of ``results`` in an OpenAI-compatible rerank response."""
    if not isinstance(body, dict):
        return 0
    results = body.get("results")
    return len(results) if isinstance(results, list) else 0


async def check_rerank_model(
    request: ModelTestRequest,
    *,
    client: httpx.AsyncClient,
) -> ModelProbeResult:
    """Go ``CheckRerankModel`` — rerank one document and count the results."""
    base_url = _require_model_and_base_url(
        request, code="initialization.rerank_model_and_base_url_required"
    )
    await validate_ssrf_safe_url(request.base_url or "")

    payload: JsonObject = {
        "model": request.model,
        "query": _RERANK_PROBE_QUERY,
        "documents": [_RERANK_PROBE_DOCUMENT],
    }
    try:
        response = await client.post(
            f"{base_url}/rerank",
            json=payload,
            headers=build_auth_headers(request),
        )
    except httpx.HTTPError as exc:
        return ModelProbeResult(
            available=False,
            message=f"重排测试失败: {type(exc).__name__}: {exc}",
        )
    if response.status_code >= 400:
        detail = f"Http Status: {response.status_code}"
        return ModelProbeResult(available=False, message=f"重排测试失败: {detail}")
    try:
        count = _rerank_result_count(response.json())
    except ValueError as exc:
        return ModelProbeResult(available=False, message=f"重排测试失败: 响应解析失败: {exc}")
    if count > 0:
        return ModelProbeResult(available=True, message=f"重排功能正常，返回{count}个结果")
    return ModelProbeResult(available=False, message="重排接口连接成功，但未返回重排结果")


def _asr_transcript(body: JsonValue) -> str:
    if not isinstance(body, dict):
        return ""
    text = body.get("text")
    return text if isinstance(text, str) else ""


def _asr_failure_message(detail: str) -> str | None:
    """Fatal-error classification from Go ``CheckASRModel``.

    Returns ``None`` when the error is non-fatal — Go then reports the
    endpoint as *reachable* because only a fatal class (auth, missing
    endpoint, unreachable host, unknown model) proves it is not.
    """
    lowered = detail.lower()
    if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return f"认证失败，请检查API Key：{detail}"
    if "404" in lowered or "not found" in lowered:
        if "model" in lowered:
            return f"模型不存在，请检查模型名称：{detail}"
        return f"API端点不存在，请检查Base URL：{detail}"
    if "connection refused" in lowered or "no such host" in lowered or "dial tcp" in lowered:
        return f"无法连接到服务器，请检查Base URL：{detail}"
    if "connecterror" in lowered or "connecttimeout" in lowered:
        return f"无法连接到服务器，请检查Base URL：{detail}"
    return None


async def check_asr_model(
    request: ModelTestRequest,
    *,
    client: httpx.AsyncClient,
) -> ModelProbeResult:
    """Go ``CheckASRModel`` — transcribe a silent WAV to probe the endpoint."""
    base_url = _require_model_and_base_url(
        request, code="initialization.asr_model_and_base_url_required"
    )
    await validate_ssrf_safe_url(request.base_url or "")

    headers = build_auth_headers(request)
    # multipart bodies must not carry a hand-set JSON content type
    headers.pop("Content-Type", None)
    try:
        response = await client.post(
            f"{base_url}/audio/transcriptions",
            files={"file": (_ASR_TEST_FILENAME, ASR_TEST_WAV, "audio/wav")},
            data={"model": request.model},
            headers=headers,
        )
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        failure = _asr_failure_message(detail)
        if failure is not None:
            return ModelProbeResult(available=False, message=failure)
        return ModelProbeResult(available=True, message=f"ASR端点可达（非致命错误: {detail}）")

    if response.status_code >= 400:
        detail = f"status code: {response.status_code}, body: {response.text}"
        failure = _asr_failure_message(detail)
        if failure is not None:
            return ModelProbeResult(available=False, message=failure)
        return ModelProbeResult(available=True, message=f"ASR端点可达（非致命错误: {detail}）")

    try:
        text = _asr_transcript(response.json())
    except ValueError:
        text = ""
    if text:
        return ModelProbeResult(available=True, message=f"ASR连接成功，转写结果: {text}")
    return ModelProbeResult(available=True, message=_ASR_OK)


class MultimodalTestConfig(BaseModel):
    """Form fields of Go's ``testMultimodalForm`` used by the probe.

    The chunking fields (``chunk_size``, ``chunk_overlap``,
    ``separators``) only matter for the DocReader path, which is an
    optional external dependency; they are accepted and validated so the
    wire contract does not change when DocReader lands.
    """

    model_config = ConfigDict(frozen=True)

    vlm_model: str = ""
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_interface_type: str = ""
    storage_type: str = ""
    cos_secret_id: str = ""
    cos_secret_key: str = ""
    cos_region: str = ""
    cos_bucket_name: str = ""
    cos_app_id: str = ""
    cos_path_prefix: str = ""
    minio_bucket_name: str = ""
    minio_path_prefix: str = ""


def validate_multimodal_config(config: MultimodalTestConfig) -> MultimodalTestConfig:
    """Go's ``TestMultimodalFunction`` guard clauses (400 on any failure).

    ``vlm_interface_type == "ollama"`` rewrites the base URL from
    ``OLLAMA_BASE_URL`` exactly as Go does; the returned config is a new
    object (never a mutation of the caller's).
    """
    resolved = config
    if config.vlm_interface_type == "ollama":
        resolved = config.model_copy(
            update={"vlm_base_url": f"{resolve_ollama_dial_base_url()}/v1"}
        )
    storage_type = resolved.storage_type.lower()
    resolved = resolved.model_copy(update={"storage_type": storage_type})

    if not resolved.vlm_model or not resolved.vlm_base_url:
        raise ValidationError(
            "VLM模型名称和Base URL不能为空",
            code="initialization.vlm_model_and_base_url_required",
        )
    if storage_type == _MULTIMODAL_STORAGE_COS:
        if not all(
            (
                resolved.cos_secret_id,
                resolved.cos_secret_key,
                resolved.cos_region,
                resolved.cos_bucket_name,
                resolved.cos_app_id,
            )
        ):
            raise ValidationError(
                "COS配置信息不能为空",
                code="initialization.cos_config_required",
            )
    elif storage_type == _MULTIMODAL_STORAGE_MINIO:
        if not resolved.minio_bucket_name:
            raise ValidationError(
                "MinIO配置信息不能为空",
                code="initialization.minio_config_required",
            )
    else:
        raise ValidationError(
            "无效的存储类型",
            code="initialization.invalid_storage_type",
        )
    return resolved


def get_max_file_size_mb() -> int:
    """``MAX_FILE_SIZE_MB`` env cap (Go ``utils.GetMaxFileSizeMB``).

    Deliberately an env var, not a runtime system setting — mirrors the
    comment on Go's ``utils/filesize.go``.
    """
    raw = os.environ.get(_MAX_FILE_SIZE_ENV, "")
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_FILE_SIZE_MB
    return parsed if parsed > 0 else _DEFAULT_MAX_FILE_SIZE_MB


def validate_image_upload(*, content_type: str, size: int) -> None:
    """Go's upload guards: image content type + ``MAX_FILE_SIZE_MB`` cap."""
    if not content_type.startswith(_IMAGE_CONTENT_TYPE_PREFIX):
        raise ValidationError(
            "只允许上传图片文件",
            code="initialization.image_only",
        )
    max_size_mb = get_max_file_size_mb()
    if size > max_size_mb * _BYTES_PER_MB:
        raise ValidationError(
            f"图片文件大小不能超过{max_size_mb}MB",
            code="initialization.image_too_large",
        )


async def test_multimodal_function(
    config: MultimodalTestConfig,
    *,
    content_type: str,
    size: int,
) -> MultimodalProbeResult:
    """Go ``TestMultimodalFunction`` — validate inputs, then run DocReader.

    DocReader is an optional external dependency (``docreader_addr`` is
    configured but no client exists yet). The upstream handler has the
    identical branch —
    ``if h.documentReader == nil { return ..., "DocReader service not
    configured" }`` — which surfaces as ``data.success = false`` inside a
    200 response. That path is what this implementation takes, so the wire
    behaviour is already correct and turns green once the reader lands.
    """
    resolved = validate_multimodal_config(config)
    await validate_ssrf_safe_url(resolved.vlm_base_url)
    validate_image_upload(content_type=content_type, size=size)
    return MultimodalProbeResult(
        success=False,
        message=_DOCREADER_NOT_CONFIGURED,
        processing_time=0,
    )


__all__ = [
    "ASR_TEST_WAV",
    "ModelProbeResult",
    "MultimodalProbeResult",
    "MultimodalTestConfig",
    "check_asr_model",
    "check_rerank_model",
    "get_max_file_size_mb",
    "test_embedding_model",
    "test_multimodal_function",
    "validate_image_upload",
    "validate_multimodal_config",
]
