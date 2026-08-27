"""Upload-pipeline helpers for the file-import path.

Storage-agnostic helpers shared by the knowledge-from-file orchestration:
extension classification (import allow-list, video / image / audio,
table formats), file-name and file-type normalisation, MD5 content
hashing (the dedup identity), the effective process-config resolution
(multimodal, question generation), the duplicate-file gate, and the
async-task seam (``DocumentProcessPayload`` + ``DocumentTaskDispatcher``).

The task-dispatch seam is deliberately a protocol: the worker layer is
not merged yet, so the orchestrator accepts any object exposing
``dispatch(payload=...)`` and degrades gracefully when none is wired.
Byte persistence itself lives behind ``FileService`` (see
``src.ai.storage.base``); nothing here reads or writes objects.

Field and JSON names mirror the upstream contract exactly.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.common.exception import ValidationError
from src.common.json import BindParams, JsonObject, JsonValue
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

# ── File-type constants (mirror the upstream import allow-list) ─────────

# Returned by :func:`file_type_of` when a name carries no extension.
UNKNOWN_FILE_TYPE = "unknown"

# Extensions accepted by every knowledge import path (direct upload and
# the worker's post-download re-check share one source of truth).
SUPPORTED_IMPORT_EXTENSIONS: frozenset[str] = frozenset(
    {
        "pdf",
        "txt",
        "docx",
        "doc",
        "epub",
        "html",
        "htm",
        "mhtml",
        "md",
        "markdown",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "csv",
        "xlsx",
        "xls",
        "pptx",
        "ppt",
        "json",
        "mp3",
        "wav",
        "m4a",
        "flac",
        "ogg",
    }
)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({"mp4", "mov", "avi", "mkv", "webm", "wmv", "flv"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {"jpg", "jpeg", "png", "gif", "webp", "bmp", "svg", "tiff"}
)

AUDIO_EXTENSIONS: frozenset[str] = frozenset({"mp3", "wav", "m4a", "flac", "ogg"})

# Spreadsheet formats that receive an extra table-summary task after the
# document-process task.
DATA_TABLE_EXTENSIONS: frozenset[str] = frozenset({"csv", "xlsx", "xls"})

# ── Ingestion defaults (mirror the upstream defaults) ───────────────────

# ``channel`` falls back to ``web`` when left blank.
DEFAULT_CHANNEL = "web"

# Default number of questions generated per chunk when the effective
# config does not pin one.
DEFAULT_QUESTION_COUNT = 3

# Metadata key under which per-upload process overrides are persisted.
METADATA_PROCESS_OVERRIDES_KEY = "process_overrides"


# ── Extension / file-type helpers ───────────────────────────────────────


def normalize_file_extension(ext: str) -> str:
    """Lowercase a bare extension, stripping a leading dot and whitespace."""
    return ext.lower().strip().lstrip(".")


def file_type_of(file_name: str) -> str:
    """Return the last dot-suffix of a file name, or ``unknown``."""
    parts = file_name.split(".")
    if len(parts) < 2:
        return UNKNOWN_FILE_TYPE
    return parts[-1]


def is_video_type(file_type: str) -> bool:
    """True for a video container extension."""
    return normalize_file_extension(file_type) in VIDEO_EXTENSIONS


def is_image_type(file_type: str) -> bool:
    """True for an image extension."""
    return normalize_file_extension(file_type) in IMAGE_EXTENSIONS


def is_audio_type(file_type: str) -> bool:
    """True for an audio extension."""
    return normalize_file_extension(file_type) in AUDIO_EXTENSIONS


def is_supported_import_type(file_type: str) -> bool:
    """True when the extension is on the import allow-list."""
    ext = normalize_file_extension(file_type)
    return ext != "" and ext != UNKNOWN_FILE_TYPE and ext in SUPPORTED_IMPORT_EXTENSIONS


def is_valid_file_type(file_name: str) -> bool:
    """True when a file name's extension is importable."""
    return is_supported_import_type(file_type_of(file_name))


def validate_import_file_type(file_type: str) -> str:
    """Normalise and reject an unimportable file type; return it bare.

    Raises ``ValidationError`` for an undeterminable type, a video
    container, or an extension outside the import allow-list — matching
    the shared file-import gate.
    """
    normalized = normalize_file_extension(file_type)
    if normalized == "" or normalized == UNKNOWN_FILE_TYPE:
        raise ValidationError(
            code="knowledge.invalid_file_type",
            message="无法确定文件类型",
        )
    if is_video_type(normalized):
        raise ValidationError(
            code="knowledge.video_not_supported",
            message="暂不支持上传视频文件",
        )
    if not is_supported_import_type(normalized):
        raise ValidationError(
            code="knowledge.unsupported_file_type",
            message=f"不支持的文件类型: {normalized}",
        )
    return normalized


def calculate_file_hash(data: bytes) -> str:
    """MD5 hex digest of uploaded content (the dedup identity)."""
    return hashlib.md5(data).hexdigest()


def validate_file_name(file_name: str) -> str:
    """Return the safe basename of an upload, or raise ``ValidationError``.

    Mirrors the upstream input validation: control characters are
    rejected and the base name is bounded to the storage column width.
    """
    base = os.path.basename(file_name.strip())
    if not base:
        raise ValidationError(
            code="knowledge.empty_file_name",
            message="file name is required",
        )
    if any(ord(ch) < 32 for ch in base):
        raise ValidationError(
            code="knowledge.invalid_file_name",
            message="文件名包含非法字符",
        )
    if len(base) > 255:
        raise ValidationError(
            code="knowledge.invalid_file_name",
            message="文件名包含非法字符",
        )
    return base


def default_channel(channel: str) -> str:
    """Return ``channel``, falling back to ``web`` when blank."""
    return channel if channel else DEFAULT_CHANNEL


# ── Effective process-config resolution ─────────────────────────────────


def _is_true(value: JsonValue) -> bool:
    """True only for an explicit JSON boolean ``true``."""
    return isinstance(value, bool) and value


def _config_enabled(config: JsonObject | None) -> bool:
    """True when a JSON config enables its capability.

    Accepts either an explicit ``enabled`` flag or a configured model id
    / name, matching the upstream "is enabled" semantics for the VLM and
    ASR prerequisite checks.
    """
    if not isinstance(config, dict):
        return False
    if _is_true(config.get("enabled")):
        return True
    return bool(config.get("model_id")) or bool(config.get("model_name"))


@dataclass(frozen=True)
class EffectiveProcessConfig:
    """Merged per-upload processing view used for task dispatch."""

    enable_multimodel: bool
    enable_question_generation: bool
    question_count: int


def resolve_effective_process_config(
    *,
    chunking_config: JsonObject | None,
    question_generation_config: JsonObject | None,
    enable_multimodel: bool | None,
    process_overrides: JsonObject | None,
) -> EffectiveProcessConfig:
    """Merge KB defaults with per-upload overrides.

    Precedence mirrors the upstream resolution: an explicit override
    wins, then the request-level ``enable_multimodel`` flag, then the
    knowledge-base chunking default. Question generation reads the
    override config first and falls back to the KB config; a non-positive
    count defaults to ``DEFAULT_QUESTION_COUNT``.
    """
    overrides = process_overrides if isinstance(process_overrides, dict) else {}

    overridden_multimodel = overrides.get("enable_multimodel")
    if isinstance(overridden_multimodel, bool):
        multimodal = overridden_multimodel
    elif enable_multimodel is not None:
        multimodal = enable_multimodel
    else:
        multimodal = _is_true((chunking_config or {}).get("enable_multimodal"))

    override_question = overrides.get("question_generation_config")
    if isinstance(override_question, dict):
        question_enabled = _is_true(override_question.get("enabled"))
        raw_count = override_question.get("question_count")
    elif isinstance(question_generation_config, dict):
        question_enabled = _is_true(question_generation_config.get("enabled"))
        raw_count = question_generation_config.get("question_count")
    else:
        question_enabled = False
        raw_count = None

    question_count = DEFAULT_QUESTION_COUNT
    if isinstance(raw_count, (int, float)) and raw_count > 0:
        question_count = int(raw_count)

    return EffectiveProcessConfig(
        enable_multimodel=multimodal,
        enable_question_generation=question_enabled,
        question_count=question_count,
    )


def validate_media_prerequisites(
    *,
    file_type: str,
    vlm_config: JsonObject | None,
    asr_config: JsonObject | None,
) -> None:
    """Enforce the model prerequisites for image / audio imports.

    Image imports require a configured vision model; audio imports
    require a configured ASR model. Matches the upstream gate.
    """
    normalized = normalize_file_extension(file_type)
    if is_image_type(normalized) and not _config_enabled(vlm_config):
        raise ValidationError(
            code="knowledge.vlm_required",
            message="上传图片文件需要设置VLM模型",
        )
    if is_audio_type(normalized) and not _config_enabled(asr_config):
        raise ValidationError(
            code="knowledge.asr_required",
            message="上传音频文件需要设置ASR语音识别模型",
        )


# ── Duplicate gate ──────────────────────────────────────────────────────


async def check_file_knowledge_exists(
    repo: KnowledgeRepository,
    *,
    tenant_id: int,
    knowledge_base_id: str,
    file_name: str,
    file_type: str,
    file_size: int,
    file_hash: str,
) -> Document | None:
    """Return the existing row when a file is already imported.

    Mirrors the upstream duplicate gate: a file is a duplicate when a
    non-failed ``file`` row in the same knowledge base shares the content
    hash, compared within the same file type (same content in different
    formats stays importable). When no hash is available the identity
    falls back to ``(file_name, file_size)``. The query narrows by the
    exact columns first; status and type are filtered after the read so
    the raw-SQL repository surface stays untouched.
    """
    columns: BindParams = {
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "type": "file",
    }
    if file_hash:
        columns["file_hash"] = file_hash
    rows = await repo.find_all_by_column_values(columns)
    if file_hash:
        candidates = [row for row in rows if row.parse_status != "failed"]
    else:
        candidates = [
            row
            for row in rows
            if row.parse_status != "failed"
            and row.file_name == file_name
            and row.file_size == file_size
        ]
    if not candidates:
        return None
    if file_hash:
        target_type = normalize_file_extension(file_type)
        for row in candidates:
            if row.file_type is None or normalize_file_extension(row.file_type) == target_type:
                return row
        return None
    return candidates[0]


# ── Async task seam ─────────────────────────────────────────────────────


class DocumentProcessPayload(BaseModel):
    """Payload of the document-process task (wire names mirror upstream)."""

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    file_path: str
    file_name: str
    file_type: str
    enable_multimodel: bool = False
    enable_question_generation: bool = False
    question_count: int = DEFAULT_QUESTION_COUNT
    language: str = ""


@runtime_checkable
class DocumentTaskDispatcher(Protocol):
    """Enqueue a document-process task; returns the task id."""

    async def dispatch(self, *, payload: DocumentProcessPayload) -> str:
        """Persist the task and return its id, or raise on failure."""
        ...


__all__ = [
    "AUDIO_EXTENSIONS",
    "DATA_TABLE_EXTENSIONS",
    "DEFAULT_CHANNEL",
    "DEFAULT_QUESTION_COUNT",
    "IMAGE_EXTENSIONS",
    "METADATA_PROCESS_OVERRIDES_KEY",
    "SUPPORTED_IMPORT_EXTENSIONS",
    "UNKNOWN_FILE_TYPE",
    "VIDEO_EXTENSIONS",
    "DocumentProcessPayload",
    "DocumentTaskDispatcher",
    "EffectiveProcessConfig",
    "calculate_file_hash",
    "check_file_knowledge_exists",
    "default_channel",
    "file_type_of",
    "is_audio_type",
    "is_image_type",
    "is_supported_import_type",
    "is_valid_file_type",
    "is_video_type",
    "normalize_file_extension",
    "resolve_effective_process_config",
    "validate_file_name",
    "validate_import_file_type",
    "validate_media_prerequisites",
]
