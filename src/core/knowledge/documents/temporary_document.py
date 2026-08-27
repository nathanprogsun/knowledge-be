"""Temporary-document domain: session-scoped, expiring chat attachments.

A temporary document records a chat attachment upload and, once parsed,
the extracted text plus its chunks. Rows are scoped to a ``session_id``,
expire after a fixed TTL, and are swept by ``cleanup_expired``.

Scope of this module
--------------------

- Domain value types mirroring the wire contract: the stored chunk shape,
  extracted-image shape, async task payload, and per-upload processing
  options.
- Lifecycle orchestration over the repository: ``create`` (upload
  metadata leg), ``get`` / ``list`` / ``delete`` (soft-delete leg) and
  ``cleanup_expired`` (expiry sweep).
- Pure content-selection helpers used by prompt assembly: query-term
  extraction, budgeted chunk selection, image-reference decoding, and the
  text/chunking transform that the async parse worker applies.

Deferred responsibilities (neutral comments mark each seam):

- **Byte persistence.** ``create`` records the metadata row with the
  caller-supplied storage reference; reading/writing the actual file
  lives behind a storage service that is not part of this foundation.
- **Async parse.** The pre-parse->promote worker needs a document parser,
  an image resolver, and a task queue; until those land, the lifecycle
  transitions in the repository (``mark_processing`` / ``mark_ready`` /
  ``mark_failed``) are the promotion API and the content-side transform
  is ``analyze_content``.
- **Dynamic parser support.** Extension acceptance starts from the static
  allow-list; consulting installed parser engines for extra extensions is
  deferred with the parser layer.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.knowledge.documents.chunker import (
    STRATEGY_AUTO,
    approx_token_count,
    default_config,
    detect_language,
    split,
)
from src.db.dao.temporary_document_repository import TemporaryDocumentRepository
from src.db.models.temporary_document import (
    MAX_TEMPORARY_ATTACHMENTS_PER_MESSAGE,
    TEMPORARY_DOCUMENT_STATUS_UPLOADED,
    TemporaryDocument,
)

# ── TTL / budgets / caps ───────────────────────────────────────────────

# Default expiry window for a temporary document.
TEMPORARY_DOCUMENT_DEFAULT_TTL: timedelta = timedelta(hours=24)

# Content fits inline (no chunk selection) when its token count is at or
# below this and fits the prompt budget.
TEMPORARY_DOCUMENT_INLINE_TOKENS = 12000
# Per-message prompt budget for attachment content.
TEMPORARY_DOCUMENT_PROMPT_BUDGET = 12000
# Maximum number of chunks a single attachment may contribute to a prompt.
TEMPORARY_DOCUMENT_MAX_PROMPT_PARTS = 16

# Chunking parameters for the async parse transform.
TEMPORARY_DOCUMENT_CHUNK_SIZE = 1600
TEMPORARY_DOCUMENT_CHUNK_OVERLAP = 160

# Extracted-text threshold (in runes, ignoring image markdown) below which
# a document is treated as image-only / scanned and eligible for OCR.
TEMPORARY_DOCUMENT_LOW_TEXT_RUNES = 200
# Maximum number of page images a scanned document may send for OCR.
TEMPORARY_DOCUMENT_IMAGE_OCR_MAX_PAGES = 8

# Upload size cap. Defaults to 50 MB; ``MAX_FILE_SIZE_MB`` overrides.
_MAX_FILE_SIZE_ENV = "MAX_FILE_SIZE_MB"
_DEFAULT_MAX_FILE_SIZE_MB = 50
_BYTES_PER_MB = 1024 * 1024

# ── Extension allow-lists ──────────────────────────────────────────────

# File extensions accepted for temporary uploads (lowercase, leading dot).
TEMPORARY_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".docx",
        ".doc",
        ".pdf",
        ".ppt",
        ".pptx",
        ".epub",
        ".mhtml",
        ".xlsx",
        ".xls",
        ".md",
        ".markdown",
        ".txt",
        ".csv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".log",
        ".html",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac",
    }
)

# Text extensions parsed inline (no external parser required).
TEMPORARY_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"}
)

# Image extensions (vision-capable attachments expose their image directly).
TEMPORARY_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
)

# Matches markdown image references so text-yield estimation ignores
# image-only content (e.g. scanned PDFs).
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Han-run bigrams (CJK query terms), mirroring the Han script detection.
_HAN_IDEOGRAPH_RE = re.compile(r"[\u4e00-\u9fff]")

_VISUAL_QUERY_MARKERS: tuple[str, ...] = (
    "图",
    "表格",
    "截图",
    "页面",
    "排版",
    "chart",
    "figure",
    "diagram",
    "image",
    "layout",
)

# Batch size for the expiry sweep (bounded loops, Go uses the same 100).
_EXPIRY_SWEEP_BATCH = 100


# ── Domain value types ────────────────────────────────────────────────


class TemporaryDocumentChunk(BaseModel):
    """One parsed chunk of a temporary document (wire shape)."""

    model_config = ConfigDict(frozen=True)

    seq: int
    content: str
    context_header: str = ""
    start: int
    end: int
    token_count: int


class TemporaryDocumentImage(BaseModel):
    """An extracted image reference of a temporary document."""

    model_config = ConfigDict(frozen=True)

    original_ref: str = ""
    url: str
    mime_type: str = ""


class TemporaryDocumentTaskPayload(BaseModel):
    """Payload for the async parse task of a temporary document."""

    model_config = ConfigDict(frozen=True)

    tenant_id: int
    document_id: str


class TemporaryDocumentCreateOptions(BaseModel):
    """Per-upload processing options persisted in ``processing_options``."""

    model_config = ConfigDict(frozen=True)

    # Verified agent source workspace used to resolve parser/model
    # dependencies; the document itself remains owned by ``tenant_id``.
    resource_tenant_id: int = 0
    asr_model_id: str = ""
    parser_engine: str = ""
    # Enables image understanding (caption/OCR) during async parse.
    vlm_model_id: str = ""
    image_understanding: bool = False
    # 0 uses the global OCR page cap.
    ocr_max_pages: int = 0


# Alias for the selection helpers' return shape.
ContentSelection: TypeAlias = tuple[str, int, int]


# ── Upload-side helpers ────────────────────────────────────────────────


def temporary_document_ttl() -> timedelta:
    """Default expiry window for a temporary document."""
    return TEMPORARY_DOCUMENT_DEFAULT_TTL


def max_upload_bytes() -> int:
    """Upload size cap in bytes (``MAX_FILE_SIZE_MB`` env, default 50 MB)."""
    raw = os.environ.get(_MAX_FILE_SIZE_ENV, "")
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_FILE_SIZE_MB * _BYTES_PER_MB
    return parsed * _BYTES_PER_MB if parsed > 0 else _DEFAULT_MAX_FILE_SIZE_MB * _BYTES_PER_MB


def supported_extension(ext: str) -> bool:
    """Return True when ``ext`` (lowercase, leading dot) is accepted.

    The caller lowercases the extension before it reaches this check; the
    helper lowercases again so it is safe against any casing. Static
    allow-list only; consulting installed parser engines for additional
    extensions is deferred with the parser layer.
    """
    return ext.lower() in TEMPORARY_DOCUMENT_EXTENSIONS


def is_text_extension(ext: str) -> bool:
    """Return True for extensions parsed inline as plain text."""
    return ext in TEMPORARY_TEXT_EXTENSIONS


def is_image_extension(ext: str) -> bool:
    """Return True for image extensions (vision-capable attachments)."""
    return ext in TEMPORARY_IMAGE_EXTENSIONS


def _validate_file_name(file_name: str) -> str:
    """Return a safe basename for an upload, or raise ``ValidationError``."""
    base = os.path.basename(file_name.strip())
    if not base:
        raise ValidationError(
            code="temporary_document.file_name_required",
            message="file name is required",
        )
    if any(ord(ch) < 32 for ch in base):
        raise ValidationError(
            code="temporary_document.unsafe_file_name",
            message="invalid characters in file name",
        )
    return base


# ── Content / prompt-side helpers ─────────────────────────────────────


def approx_text_content_runes(md: str) -> int:
    """Runes of real text in markdown, ignoring image references."""
    stripped = _MARKDOWN_IMAGE_RE.sub("", md)
    return len(stripped.strip())


def query_terms(query: str) -> list[str]:
    """Tokenise a query for chunk scoring (words + Han bigrams).

    Mirrors the query-term extraction: whitespace/punctuation-separated
    fields of at least two runes, plus every consecutive Han rune bigram.
    """
    q = query.lower().strip()
    seen: set[str] = set()
    terms: list[str] = []
    for field in re.split(r"[\W_]+", q):
        if len(field) < 2:
            continue
        if field not in seen:
            seen.add(field)
            terms.append(field)
    for i in range(len(q) - 1):
        pair = q[i : i + 2]
        if (
            _HAN_IDEOGRAPH_RE.fullmatch(pair[0])
            and _HAN_IDEOGRAPH_RE.fullmatch(pair[1])
            and pair not in seen
        ):
            seen.add(pair)
            terms.append(pair)
    return terms


def is_visual_document_query(query: str) -> bool:
    """Return True when the query is visual and should surface images."""
    lower = query.lower()
    return any(marker in lower for marker in _VISUAL_QUERY_MARKERS)


def image_refs_of(raw: list[JsonObject] | None) -> list[TemporaryDocumentImage]:
    """Leniently decode stored image references (skips malformed entries)."""
    if not raw:
        return []
    images: list[TemporaryDocumentImage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        original_ref = item.get("original_ref")
        mime_type = item.get("mime_type")
        images.append(
            TemporaryDocumentImage(
                url=url,
                original_ref=original_ref if isinstance(original_ref, str) else "",
                mime_type=mime_type if isinstance(mime_type, str) else "",
            )
        )
    return images


def analyze_content(content: str) -> tuple[list[TemporaryDocumentChunk], int]:
    """Chunk parsed text and estimate its total token count.

    This is the content-side transform of the async parse step: pure
    (no storage or IO). ``SplitterConfig`` is frozen, so the strategy /
    size / overlap are applied by rebuilding the defaults with
    :func:`dataclasses.replace`.
    """
    lang = detect_language(content)
    cfg = replace(
        default_config(),
        strategy=STRATEGY_AUTO,
        chunk_size=TEMPORARY_DOCUMENT_CHUNK_SIZE,
        chunk_overlap=TEMPORARY_DOCUMENT_CHUNK_OVERLAP,
    )
    parts = split(content, cfg)
    chunks = [
        TemporaryDocumentChunk(
            seq=part.seq,
            content=part.content,
            context_header=part.context_header,
            start=part.start,
            end=part.end,
            token_count=approx_token_count(part.embedding_content(), lang),
        )
        for part in parts
    ]
    return chunks, approx_token_count(content, lang)


def _decode_chunks(document: TemporaryDocument) -> list[TemporaryDocumentChunk]:
    """Decode the stored chunk list, skipping malformed entries."""
    chunks: list[TemporaryDocumentChunk] = []
    for item in document.chunks:
        try:
            chunks.append(TemporaryDocumentChunk.model_validate(item))
        except PydanticValidationError:
            continue
    return chunks


def _chunk_score(chunk: TemporaryDocumentChunk, terms: list[str]) -> int:
    """Term-frequency score of a chunk against the query terms."""
    text = (chunk.context_header + "\n" + chunk.content).lower()
    score = 0
    for term in terms:
        score += text.count(term) * (1 + len(term) // 2)
    return score


def select_content_with_budget(
    document: TemporaryDocument,
    query: str,
    budget: int,
) -> ContentSelection:
    """Select attachment content for a prompt under a token budget.

    Returns ``(content, selected, total)``: the assembled text, the number
    of chunks selected, and the total number of chunks. When the document
    is short enough to inline, the full extracted content is returned.
    Otherwise chunks are ranked by query-term frequency, greedily selected
    up to the budget, and re-assembled in document order.
    """
    chunks = _decode_chunks(document)
    if budget <= 0:
        budget = TEMPORARY_DOCUMENT_PROMPT_BUDGET
    if not chunks or (
        document.token_count <= TEMPORARY_DOCUMENT_INLINE_TOKENS and document.token_count <= budget
    ):
        return document.content or "", len(chunks), len(chunks)

    terms = query_terms(query)
    ranked = sorted(
        ((chunk, _chunk_score(chunk, terms)) for chunk in chunks),
        key=lambda pair: (-pair[1], pair[0].seq),
    )
    selected: list[TemporaryDocumentChunk] = []
    tokens = 0
    for candidate, _score in ranked:
        if len(selected) >= TEMPORARY_DOCUMENT_MAX_PROMPT_PARTS:
            break
        if tokens > 0 and tokens + candidate.token_count > budget:
            continue
        selected.append(candidate)
        tokens += candidate.token_count
    selected.sort(key=lambda chunk: chunk.seq)

    parts: list[str] = []
    for part in selected:
        body = part.content.strip()
        if part.context_header:
            body = f"{part.context_header}\n\n{body}"
        parts.append(body)
    return "\n\n---\n\n".join(parts), len(selected), len(chunks)


# ── Service ────────────────────────────────────────────────────────────


class TemporaryDocumentService:
    """Temporary-document lifecycle orchestration over the repository.

    Request-scoped: the repository holds the per-request ``AsyncSession``.
    The upload metadata leg (``create``) is implemented here; byte
    persistence, parser dispatch and task scheduling are deferred with
    the storage / parser / worker layers.
    """

    def __init__(self, *, repo: TemporaryDocumentRepository) -> None:
        self._repo = repo

    async def create(
        self,
        *,
        tenant_id: int,
        session_id: str,
        resource_ref: str,
        file_name: str,
        mime_type: str,
        file_size: int,
        options: TemporaryDocumentCreateOptions | None = None,
    ) -> TemporaryDocument:
        """Validate and record an upload's metadata row (``uploaded``).

        Byte persistence is out of scope here: the caller supplies the
        storage reference from the file-persistence step. Persisting the
        row triggers no parse yet — scheduling the async parse is deferred
        with the worker layer.
        """
        if tenant_id <= 0 or not session_id.strip():
            raise ValidationError(
                code="temporary_document.invalid_scope",
                message="invalid attachment scope",
            )
        safe_name = _validate_file_name(file_name)
        ext = os.path.splitext(safe_name)[1].lower()
        if not supported_extension(ext):
            raise ValidationError(
                code="temporary_document.unsupported_file_type",
                message=f"unsupported file type: {ext}",
            )
        cap = max_upload_bytes()
        if file_size <= 0 or file_size > cap:
            raise ValidationError(
                code="temporary_document.invalid_file_size",
                message=(f"file size must be between 1 byte and {cap // _BYTES_PER_MB}MB"),
            )

        now = datetime.now(UTC)
        opts = options or TemporaryDocumentCreateOptions()
        row = TemporaryDocument(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            session_id=session_id,
            resource_ref=resource_ref,
            file_name=safe_name,
            file_type=ext,
            mime_type=mime_type.strip(),
            file_size=file_size,
            status=TEMPORARY_DOCUMENT_STATUS_UPLOADED,
            expires_at=now + temporary_document_ttl(),
            processing_options=opts.model_dump(exclude_defaults=True),
            created_at=now,
            updated_at=now,
        )
        return await self._repo.create(row)

    async def get(
        self,
        *,
        tenant_id: int,
        session_id: str,
        document_id: str,
    ) -> TemporaryDocument | None:
        """Return the live document scoped to the session, or ``None``."""
        return await self._repo.get_scoped(
            tenant_id=tenant_id,
            session_id=session_id,
            document_id=document_id,
        )

    async def list(
        self,
        *,
        tenant_id: int,
        session_id: str,
    ) -> list[TemporaryDocument]:
        """Return every live document of the session, oldest first."""
        return await self._repo.list_scoped(
            tenant_id=tenant_id,
            session_id=session_id,
        )

    async def delete(
        self,
        *,
        tenant_id: int,
        session_id: str,
        document_id: str,
    ) -> bool:
        """Soft-delete the session-scoped document.

        Returns whether a live row was removed. Removing the underlying
        stored file (source + extracted images) is deferred with the
        file-persistence layer.
        """
        return await self._repo.delete_scoped(
            tenant_id=tenant_id,
            session_id=session_id,
            document_id=document_id,
            now=datetime.now(UTC),
        )

    async def cleanup_expired(self) -> int:
        """Sweep every expired document; return the number removed.

        Loops over bounded batches of the most-stale rows (ordered by
        ``expires_at``) so a large backlog does not run in one unbounded
        query.
        """
        removed = 0
        while True:
            documents = await self._repo.list_expired(
                before=datetime.now(UTC),
                limit=_EXPIRY_SWEEP_BATCH,
            )
            if not documents:
                return removed
            for document in documents:
                deleted = await self._repo.delete_scoped(
                    tenant_id=document.tenant_id,
                    session_id=document.session_id,
                    document_id=document.id,
                    now=datetime.now(UTC),
                )
                if deleted:
                    removed += 1
            if len(documents) < _EXPIRY_SWEEP_BATCH:
                return removed


__all__ = [
    "MAX_TEMPORARY_ATTACHMENTS_PER_MESSAGE",
    "TEMPORARY_DOCUMENT_DEFAULT_TTL",
    "TEMPORARY_DOCUMENT_EXTENSIONS",
    "TEMPORARY_DOCUMENT_IMAGE_OCR_MAX_PAGES",
    "TEMPORARY_DOCUMENT_INLINE_TOKENS",
    "TEMPORARY_DOCUMENT_LOW_TEXT_RUNES",
    "TEMPORARY_DOCUMENT_MAX_PROMPT_PARTS",
    "TEMPORARY_DOCUMENT_PROMPT_BUDGET",
    "TEMPORARY_IMAGE_EXTENSIONS",
    "TEMPORARY_TEXT_EXTENSIONS",
    "TemporaryDocumentChunk",
    "TemporaryDocumentCreateOptions",
    "TemporaryDocumentImage",
    "TemporaryDocumentService",
    "TemporaryDocumentTaskPayload",
    "analyze_content",
    "approx_text_content_runes",
    "image_refs_of",
    "is_image_extension",
    "is_text_extension",
    "is_visual_document_query",
    "max_upload_bytes",
    "query_terms",
    "select_content_with_budget",
    "supported_extension",
    "temporary_document_ttl",
]
