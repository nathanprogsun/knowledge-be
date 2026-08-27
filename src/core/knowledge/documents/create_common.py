"""Shared helpers for the knowledge create-variant modules.

Pure helpers (input validation, URL checks, file-type classification,
hashing, manual-knowledge constants) and the small persistence
primitives (row builder, wire projection, duplicate lookup, tag
validation) that ``create_url`` / ``create_passage`` / ``create_manual``
all share. None of these helpers touch the storage layer directly — the
callers inject the repositories.

Deferred seams (neutral wording): the storage-engine configured check,
the storage-quota gate, the async processing-task enqueue, and the
process-config (embedding / VLM / ASR) resolution a real import would
need. Those land with the storage and worker domains.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import urllib.parse
import uuid
from datetime import datetime

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    PARSE_STATUS_FAILED,
)
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.knowledge import Document
from src.db.models.knowledge_tag import KnowledgeTag

# ── Constants ─────────────────────────────────────────────────────────

# Max characters of a manual-knowledge content body.
MANUAL_CONTENT_MAX_LENGTH = 200000

# File extension appended to manual-knowledge file names.
MANUAL_FILE_EXTENSION = ".md"

# Extension returned by :func:`get_file_type` when a name has no suffix.
UNKNOWN_FILE_TYPE = "unknown"

# Maximum URL length accepted by the safe-URL check.
_MAX_URL_LENGTH = 2048

# Enable-status values stamped on freshly created rows.
ENABLE_STATUS_DISABLED = "disabled"
ENABLE_STATUS_ENABLED = "enabled"

# Extensions accepted by every import path (direct upload, file-URL
# download, and the worker's post-download re-check). A single set keeps
# the accepted surface identical across paths.
_SUPPORTED_IMPORT_EXTENSIONS: frozenset[str] = frozenset(
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

# XSS patterns matched against user-supplied text, mirroring the shared
# input-validation contract: a match rejects a filename / passage / title
# and is stripped from manual markdown content.
_XSS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)<script[^>]*>.*?</script>"),
    re.compile(r"(?i)<iframe[^>]*>.*?</iframe>"),
    re.compile(r"(?i)<object[^>]*>.*?</object>"),
    re.compile(r"(?i)<embed[^>]*>.*?</embed>"),
    re.compile(r"(?i)<embed[^>]*>"),
    re.compile(r"(?i)<form[^>]*>.*?</form>"),
    re.compile(r"(?i)<input[^>]*>"),
    re.compile(r"(?i)<button[^>]*>.*?</button>"),
    re.compile(r"(?i)javascript:"),
    re.compile(r"(?i)vbscript:"),
    re.compile(r"(?i)onload\s*="),
    re.compile(r"(?i)onerror\s*="),
    re.compile(r"(?i)onclick\s*="),
    re.compile(r"(?i)onmouseover\s*="),
    re.compile(r"(?i)onfocus\s*="),
    re.compile(r"(?i)onblur\s*="),
)

# URL schemes tolerated by the safe-URL check. Only ``http`` / ``https``
# survive the stricter HTTP check applied at the create boundary; the
# provider schemes are retained for legacy stored content.
_ALLOWED_URL_PROTOCOLS: tuple[str, ...] = (
    "http://",
    "https://",
    "resource://",
    "storage://",
    "local://",
    "minio://",
    "cos://",
    "tos://",
    "s3://",
    "oss://",
    "ks3://",
    "obs://",
)

# ── Boundary validators ──────────────────────────────────────────────


def require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def require_knowledge_base_id(knowledge_base_id: str) -> None:
    """Reject a blank knowledge-base id at the service boundary."""
    if not knowledge_base_id.strip():
        raise ValidationError(
            code="knowledge.kb_required",
            message="knowledge base ID is required",
        )


# ── Input validation ─────────────────────────────────────────────────


def validate_input(value: str) -> tuple[str, bool]:
    """Validate and trim user-supplied text.

    Mirrors the shared input-validation contract: empty input passes,
    ASCII control characters (except tab / LF / CR) fail, and a match on
    any XSS pattern fails. The returned string is the trimmed value.
    Returns ``("", False)`` on failure.
    """
    if value == "":
        return "", True
    for char in value:
        code = ord(char)
        if code < 32 and code not in (9, 10, 13):
            return "", False
    if any(pattern.search(value) for pattern in _XSS_PATTERNS):
        return "", False
    return value.strip(), True


def clean_markdown(input: str) -> str:
    """Strip XSS patterns from manual markdown content."""
    cleaned = input
    for pattern in _XSS_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# ── URL checks ───────────────────────────────────────────────────────


def is_valid_http_url(url: str) -> bool:
    """True only for an ``http://`` / ``https://`` URL."""
    return url.startswith(("http://", "https://"))


def is_safe_url(url: str) -> bool:
    """True when the URL passes the static safety checks.

    Mirrors the shared safe-URL contract: non-empty, at most 2048
    characters, an allowed protocol prefix, and no XSS pattern. This is
    a static check — the SSRF host guard is applied separately by the
    caller's URL guard.
    """
    if not url or len(url) > _MAX_URL_LENGTH:
        return False
    lowered = url.lower()
    if not any(lowered.startswith(protocol) for protocol in _ALLOWED_URL_PROTOCOLS):
        return False
    return not any(pattern.search(url) for pattern in _XSS_PATTERNS)


# ── File-type helpers ────────────────────────────────────────────────


def normalize_file_extension(ext: str) -> str:
    """Lowercase an extension and strip whitespace / a leading dot."""
    return ext.strip().lower().lstrip(".")


def is_supported_import_extension(ext: str) -> bool:
    """Whether a bare extension is importable."""
    normalized = normalize_file_extension(ext)
    if normalized == "" or normalized == UNKNOWN_FILE_TYPE:
        return False
    return normalized in _SUPPORTED_IMPORT_EXTENSIONS


def get_file_type(filename: str) -> str:
    """Return the trailing extension of a filename, else ``unknown``."""
    parts = filename.split(".")
    if len(parts) < 2:
        return UNKNOWN_FILE_TYPE
    return parts[-1]


def is_video_type(file_type: str) -> bool:
    """True when the file type is a video container."""
    return file_type.lower() in ("mp4", "mov", "avi", "mkv", "webm", "wmv", "flv")


def validate_import_file_type(file_type: str) -> None:
    """Reject an un-importable file type at the create boundary.

    Mirrors the shared file-import gate: an undeterminable type, a video
    container, or an unsupported extension each raise
    ``ValidationError`` before the row is persisted.
    """
    normalized = normalize_file_extension(file_type)
    if normalized == "" or normalized == UNKNOWN_FILE_TYPE:
        raise ValidationError(
            code="knowledge.file_type_unknown",
            message="无法确定文件类型",
        )
    if is_video_type(normalized):
        raise ValidationError(
            code="knowledge.video_unsupported",
            message="暂不支持上传视频文件",
        )
    if not is_supported_import_extension(normalized):
        raise ValidationError(
            code="knowledge.file_type_unsupported",
            message=f"不支持的文件类型: {normalized}",
        )


def is_file_url(raw_url: str, file_name: str, file_type: str) -> bool:
    """Whether a URL should be treated as a direct file download.

    Priority: the URL path carries a known import extension first; then
    fall back to the caller-supplied ``file_name`` / ``file_type`` hints.
    """
    try:
        parsed = urllib.parse.urlparse(raw_url)
    except ValueError:
        parsed = None
    if parsed is not None:
        ext = normalize_file_extension(posixpath.splitext(parsed.path)[1])
        if ext and is_supported_import_extension(ext):
            return True
    return bool(file_name) or bool(file_type)


def extract_file_name_from_url(raw_url: str) -> str:
    """Return the final path segment of a URL, else ``""``."""
    try:
        parsed = urllib.parse.urlparse(raw_url)
    except ValueError:
        return ""
    base = posixpath.basename(parsed.path)
    if base in ("", ".", "/"):
        return ""
    return base


def calculate_str(value: str) -> str:
    """Return the MD5 hex digest of a URL string."""
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def default_channel(channel: str | None) -> str:
    """Fall back to the web ingestion channel when none is supplied."""
    return channel or CHANNEL_WEB


def ensure_manual_file_name(title: str, now: datetime) -> str:
    """Build the ``.md`` file name for a manual-knowledge entry.

    A title already ending in ``.md`` is kept; otherwise the extension is
    appended. An empty title falls back to a timestamped name.
    """
    if not title:
        return f"manual-{now:%Y%m%d-%H%M%S}{MANUAL_FILE_EXTENSION}"
    trimmed = title.strip()
    if trimmed.lower().endswith(MANUAL_FILE_EXTENSION):
        return trimmed
    return trimmed + MANUAL_FILE_EXTENSION


# ── Row builder / projection ─────────────────────────────────────────


def build_document_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    type: str,
    title: str,
    source: str,
    channel: str,
    parse_status: str,
    enable_status: str,
    embedding_model_id: str | None,
    now: datetime,
    description: str | None = None,
    file_name: str | None = None,
    file_type: str | None = None,
    file_size: int | None = None,
    file_hash: str | None = None,
    file_path: str | None = None,
    storage_size: int = 0,
    metadata: JsonObject | None = None,
) -> Document:
    """Build a new ``documents`` row with create-path defaults.

    The id is a fresh UUID and the custom-metadata map is stamped empty
    so the persisted JSONB value is never ``NULL`` — the same defaults
    the generic document service applies.
    """
    return Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type=type,
        title=title,
        description=description,
        source=source,
        channel=channel,
        parse_status=parse_status,
        pending_subtasks_count=0,
        summary_status="none",
        enable_status=enable_status,
        embedding_model_id=embedding_model_id,
        file_name=file_name,
        file_type=file_type,
        file_size=file_size,
        file_hash=file_hash,
        file_path=file_path,
        storage_size=storage_size,
        metadata=metadata,
        custom_metadata={},
        last_faq_import_result=None,
        created_at=now,
        updated_at=now,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


def to_knowledge(row: Document) -> Knowledge:
    """Project a persisted ``documents`` row onto the wire shape.

    Storage-only columns (``pending_subtasks_count``, ``custom_metadata``,
    ``last_faq_import_result``) and the relation fields are absent from
    the wire projection, matching the generic document service.
    """
    return Knowledge(
        id=row.id,
        tenant_id=row.tenant_id,
        knowledge_base_id=row.knowledge_base_id,
        type=row.type,
        title=row.title,
        description=row.description,
        source=row.source,
        channel=row.channel,
        summary_status=row.summary_status,
        parse_status=row.parse_status,
        enable_status=row.enable_status,
        embedding_model_id=row.embedding_model_id,
        file_name=row.file_name,
        file_type=row.file_type,
        file_size=row.file_size,
        file_hash=row.file_hash,
        file_path=row.file_path,
        storage_size=row.storage_size,
        metadata=row.metadata,
        created_at=row.created_at,
        updated_at=row.updated_at,
        processed_at=row.processed_at,
        error_message=row.error_message,
        deleted_at=row.deleted_at,
    )


# ── Duplicate handling ───────────────────────────────────────────────


async def find_duplicate_document(
    *,
    tenant_id: int,
    kb_id: str,
    doc_type: str,
    knowledge_repo: KnowledgeRepository,
    source: str | None = None,
    file_hash: str | None = None,
) -> Document | None:
    """Return a matching live row of ``doc_type`` in the knowledge base.

    Mirrors the upstream existence check: failed rows are excluded, and
    a hash match is preferred over a source match. The check scans the
    repository's list read because the merged repository owns no
    dedicated existence query yet.
    """
    rows = await knowledge_repo.list_by_knowledge_base(tenant_id, kb_id)
    for row in rows:
        if row.type != doc_type or row.parse_status == PARSE_STATUS_FAILED:
            continue
        if file_hash is not None and row.file_hash == file_hash:
            return row
        if source is not None and row.source == source:
            return row
    return None


async def refresh_duplicate_timestamps(
    *,
    knowledge_repo: KnowledgeRepository,
    knowledge_id: str,
    now: datetime,
) -> Document | None:
    """Refresh the timestamps of a duplicate row.

    Mirrors the upstream duplicate path, which re-stamps the existing
    row's creation / update time before reporting the collision.
    ``created_at`` is written through the column-scoped update (the
    full-row update treats it as immutable).
    """
    return await knowledge_repo.update_columns(
        knowledge_id,
        {"created_at": now, "updated_at": now},
    )


# ── Tag validation ───────────────────────────────────────────────────


def _unique_tag_ids(tag_ids: list[str]) -> list[str]:
    """Return the non-empty tag ids, deduplicated in input order."""
    unique: list[str] = []
    seen: set[str] = set()
    for tag_id in tag_ids:
        if tag_id and tag_id not in seen:
            seen.add(tag_id)
            unique.append(tag_id)
    return unique


async def validate_knowledge_tags(
    *,
    tenant_id: int,
    kb_id: str,
    tag_ids: list[str],
    tag_repo: TagRepository,
) -> list[str]:
    """Validate every tag id exists and belongs to the knowledge base.

    Returns the deduplicated, non-empty tag ids for the caller to attach
    after the row is persisted. Mirrors the upstream tag validation:
    an unknown id or a tag owned by another knowledge base is rejected.
    """
    unique = _unique_tag_ids(tag_ids)
    if not unique:
        return []
    tags = await tag_repo.get_by_ids(tenant_id, unique)
    found: dict[str, KnowledgeTag] = {tag.id: tag for tag in tags}
    for tag_id in unique:
        tag = found.get(tag_id)
        if tag is None:
            raise ValidationError(
                code="knowledge.tag_not_found",
                message=f"标签 {tag_id} 不存在",
            )
        if tag.knowledge_base_id != kb_id:
            raise ValidationError(
                code="knowledge.tag_not_in_kb",
                message="标签不属于当前知识库",
            )
    return unique


__all__ = [
    "ENABLE_STATUS_DISABLED",
    "ENABLE_STATUS_ENABLED",
    "MANUAL_CONTENT_MAX_LENGTH",
    "MANUAL_FILE_EXTENSION",
    "UNKNOWN_FILE_TYPE",
    "build_document_row",
    "calculate_str",
    "clean_markdown",
    "default_channel",
    "ensure_manual_file_name",
    "extract_file_name_from_url",
    "find_duplicate_document",
    "get_file_type",
    "is_file_url",
    "is_safe_url",
    "is_supported_import_extension",
    "is_valid_http_url",
    "is_video_type",
    "normalize_file_extension",
    "refresh_duplicate_timestamps",
    "require_knowledge_base_id",
    "require_tenant_id",
    "to_knowledge",
    "validate_import_file_type",
    "validate_input",
    "validate_knowledge_tags",
]
