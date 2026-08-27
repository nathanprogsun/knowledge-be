"""Knowledge summary generation — standalone module.

``process_summary`` rebuilds a knowledge item's description (and, when the
knowledge base indexes into a retrieval store, its ``summary`` chunk) from
its text chunks using an injected chat client. It mirrors the upstream
summary semantics:

- text chunks are re-ordered (parser ``start_at`` offsets while no chunk
  has been manually edited, ``chunk_index`` afterwards) and concatenated
  to reconstruct the document;
- long content is sampled (head 60% / middle 20% / tail 20%) to fit the
  model input window;
- content below the real-text threshold is rejected without calling the
  model (the scanned-PDF guard), and the row is marked ``failed``;
- user-authored custom metadata is folded in as trusted document context
  (internal ingestion metadata stays excluded);
- the result is only published when the source chunks / metadata did not
  change while the model was running (stale-refresh guard raises
  ``ConflictError`` without touching ``summary_status``);
- when the knowledge base needs an embedding model, the summary is
  persisted as a ``summary`` chunk and re-indexed through the injected
  syncer hook.

The chat client is an injected seam (``src.ai.llm.Chat``); the module
never calls a provider directly. Repository and knowledge-base
dependencies are injected per call so the web layer composes them on the
request session.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from src.ai.llm import Chat, ChatOptions, Message
from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.types import CHUNK_TYPE_SUMMARY
from src.core.knowledge.documents.types import (
    SUMMARY_STATUS_COMPLETED,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_PROCESSING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

# Default model input window (characters) and output cap (tokens).
DEFAULT_MAX_INPUT_CHARS = 1024 * 24
DEFAULT_MAX_TOKENS = 2048

# Minimum number of real (non-image-markup) characters required before the
# model is called; documents below this are treated as empty and rejected.
MIN_TEXT_CONTENT_RUNES = 10

# Summary-chunk content prefix.
_SUMMARY_CHUNK_PREFIX = "# Summary\n"

# Sampling temperature for the summary call (deliberately low).
_SUMMARY_TEMPERATURE = 0.3

# First-chunk fallback cap when the model call fails for non-content
# reasons (LLM / IO error), mirroring the upstream ``first_chunk`` fallback.
_FALLBACK_SUMMARY_CHARS = 500

# Marker used when long content is sampled, so the model knows text was cut.
_OMIT_MARKER = "\n\n[...content omitted...]\n\n"

# Markdown image references ``![alt](path)`` — pure visual placeholders.
_MD_IMAGE_REF: re.Pattern[str] = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# ``<image_original>...</image_original>`` blocks wrap a redundant verbatim
# image link; the whole block is removed.
_IMAGE_ORIGINAL_BLOCK: re.Pattern[str] = re.compile(
    r"<image_original\b[^>]*>.*?</image_original>",
    re.IGNORECASE | re.DOTALL,
)

# Self-closing / attribute-only HTML ``<img>`` tags.
_HTML_IMG_TAG: re.Pattern[str] = re.compile(r"<img\b[^>]*/?>", re.IGNORECASE)

# Wrapper-style ``<image>`` / ``<image_caption>`` / ``<image_ocr>`` tags
# (opening or closing). Only the tags are stripped; OCR / caption text
# between the open and close tags is preserved.
_IMAGE_WRAPPER_TAG: re.Pattern[str] = re.compile(r"</?image[a-z_]*\b[^>]*/?>", re.IGNORECASE)

# Locale -> human-readable language name used to fill the ``{{language}}``
# prompt placeholder (the model should answer in the user's language).
_LANGUAGE_NAMES: dict[str, str] = {
    "zh-CN": "Chinese (Simplified)",
    "zh": "Chinese (Simplified)",
    "zh-Hans": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "zh-HK": "Chinese (Traditional)",
    "zh-Hant": "Chinese (Traditional)",
    "en-US": "English",
    "en": "English",
    "en-GB": "English",
    "ko-KR": "Korean",
    "ko": "Korean",
    "ja-JP": "Japanese",
    "ja": "Japanese",
    "ru-RU": "Russian",
    "ru": "Russian",
    "fr-FR": "French",
    "fr": "French",
    "de-DE": "German",
    "de": "German",
    "es-ES": "Spanish",
    "es": "Spanish",
    "pt-BR": "Portuguese",
    "pt": "Portuguese",
}

DEFAULT_LANGUAGE = "zh-CN"


class SummaryChunkSyncer(Protocol):
    """Re-index hook for a freshly written summary chunk."""

    async def sync_summary_chunk(self, *, tenant_id: int, chunk: Chunk) -> None:
        """Index ``chunk``'s summary content in the retrieval store."""


@dataclass(frozen=True)
class SummaryResult:
    """Outcome of one summary run."""

    knowledge: Knowledge
    summary: str
    summary_chunk_id: str | None = None


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_knowledge_id(id: str) -> None:
    """Reject a blank knowledge id at the service boundary."""
    if not id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="knowledge ID is required",
        )


def _to_knowledge(row: Document) -> Knowledge:
    """Project a persisted ``documents`` row onto the wire shape."""
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


def strip_image_markup(content: str) -> str:
    """Remove image markup from ``content``, keeping OCR / caption text.

    ``<image_ocr>`` / ``<image_caption>`` blocks keep their inner text:
    a naive "strip the whole block" would discard exactly the text a
    scanned-PDF VLM pipeline produced.
    """
    stripped = _IMAGE_ORIGINAL_BLOCK.sub("", content)
    stripped = _MD_IMAGE_REF.sub("", stripped)
    stripped = _HTML_IMG_TAG.sub("", stripped)
    return _IMAGE_WRAPPER_TAG.sub("", stripped)


def real_text_rune_count(content: str) -> int:
    """Return the character count of ``content`` after image markup is stripped."""
    return len(strip_image_markup(content).strip())


def custom_metadata_text(custom_metadata: JsonObject | None) -> str:
    """Render user-authored custom metadata as stable prompt context.

    Keys are sorted and only non-blank scalar values contribute; internal
    ingestion metadata is never passed to the model.
    """
    if not custom_metadata:
        return ""
    lines: list[str] = []
    for key in sorted(custom_metadata):
        value = custom_metadata[key]
        if value is None:
            continue
        text = str(value).strip()
        if key.strip() and text:
            lines.append(f"{key.strip()}: {text}")
    return "\n".join(lines)


def sample_long_content(content: str, max_chars: int) -> str:
    """Sample ``content`` to fit within ``max_chars``.

    Short content returns as-is. Long content keeps the head (60%), the
    tail (20%), and an evenly-placed middle block (20%) joined by an
    omission marker so the model knows content was skipped. If the window
    is too small to hold two markers plus a readable middle block, the
    content is truncated instead.
    """
    if len(content) <= max_chars:
        return content
    omit = len(_OMIT_MARKER)
    usable = max_chars - 2 * omit
    if usable < 100:
        return content[:max_chars]
    head_len = usable * 60 // 100
    tail_len = usable * 20 // 100
    mid_len = usable - head_len - tail_len
    head = content[:head_len]
    tail = content[len(content) - tail_len :]
    mid_start = len(content) // 2 - mid_len // 2
    if mid_start < head_len:
        mid_start = head_len
    mid_end = mid_start + mid_len
    if mid_end > len(content) - tail_len:
        mid_end = len(content) - tail_len
        mid_start = mid_end - mid_len
        if mid_start < head_len:
            mid_start = head_len
    middle = content[mid_start:mid_end]
    return head + _OMIT_MARKER + middle + _OMIT_MARKER + tail


def render_placeholders(template: str, values: dict[str, str]) -> str:
    """Replace ``{{key}}`` occurrences in ``template`` with ``values[key]``.

    Unknown placeholders are left untouched. Built-in auto-values
    (``current_time`` / ``current_week`` / ``yesterday``) are filled only
    when the caller did not supply them.
    """
    if template == "":
        return ""
    now = datetime.now(UTC)
    merged: dict[str, str] = {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_week": now.strftime("%A"),
        "yesterday": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    merged.update(values)
    result = template
    for key, value in merged.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def language_name_for(locale: str) -> str:
    """Map a locale code to a human-readable language name for prompts."""
    return _LANGUAGE_NAMES.get(locale, locale)


def needs_embedding_model(kb: KnowledgeBaseInfo) -> bool:
    """Return whether any enabled indexing pipeline needs an embedding model.

    The default indexing strategy enables vector and keyword search, so an
    absent ``indexing_strategy`` reports ``True``.
    """
    strategy = kb.indexing_strategy
    if not strategy:
        return True
    return bool(strategy.get("vector_enabled", True)) or bool(strategy.get("keyword_enabled", True))


def _metadata_version(custom_metadata: JsonObject | None) -> str:
    """Return a stable serialization of custom metadata for staleness checks."""
    if not custom_metadata:
        return ""
    return json.dumps(custom_metadata, sort_keys=True, ensure_ascii=False)


def _sort_chunks_for_summary(chunks: list[Chunk]) -> list[Chunk]:
    """Order chunks for document reconstruction.

    Parser ``start_at`` offsets stay authoritative until any chunk has been
    manually edited (``content_revision > 0``); afterwards ``chunk_index``
    is the safer reading order.
    """
    edited = any(chunk.content_revision > 0 for chunk in chunks)
    if edited:
        return sorted(chunks, key=lambda c: (c.chunk_index, c.id))
    return sorted(chunks, key=lambda c: c.start_at)


def _reconstruct_content(chunks: list[Chunk]) -> str:
    """Concatenate chunk bodies into a reconstructed document.

    Parser offsets describe the immutable source; once a replacement has
    changed length they can no longer be applied to the effective content,
    so edited documents concatenate their current enabled, non-blank
    chunks instead of truncating at stale offsets.
    """
    edited = any(chunk.content_revision > 0 for chunk in chunks)
    if edited:
        parts = [c.content for c in chunks if c.is_enabled and c.content.strip() != ""]
        return "\n\n".join(parts)
    content = ""
    for chunk in chunks:
        if chunk.start_at <= len(content):
            content = content[: chunk.start_at] + chunk.content
        else:
            content += chunk.content
    return content


async def _mark_summary_status(
    *,
    knowledge_repo: KnowledgeRepository,
    row: Document,
    status: str,
    now: datetime,
    description: str | None = None,
) -> Document:
    """Persist the summary status (and optional description) on the row."""
    updates: BindParams = {"summary_status": status, "updated_at": now}
    if description is not None:
        updates["description"] = description
    return await knowledge_repo.update(row.model_copy(update=updates))


async def _source_changed(
    *,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    tenant_id: int,
    knowledge_id: str,
    metadata_version: str,
    source_chunks: list[Chunk],
) -> bool:
    """Report whether chunk bodies or metadata changed during the run.

    Database lookup failures are re-raised so a transient read error is
    not mistaken for stale work that can be silently discarded.
    """
    latest = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if latest is None or _metadata_version(latest.custom_metadata) != metadata_version:
        return True
    for source in source_chunks:
        latest_chunk = await chunk_repo.get_by_id(tenant_id, source.id)
        if (
            latest_chunk.content_revision != source.content_revision
            or latest_chunk.is_enabled != source.is_enabled
        ):
            return True
    return False


async def _upsert_summary_chunk(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    all_chunks: list[Chunk],
    first_chunk: Chunk,
    summary: str,
    chunk_repo: ChunkRepository,
    summary_chunk_syncer: SummaryChunkSyncer | None,
) -> str:
    """Create or refresh the ``summary`` chunk, returning its id.

    The chunk carries only the model's summary text (prefixed ``# Summary``);
    file names are deliberately omitted so retrieved context never
    re-introduces the scanned-file hallucination vector.
    """
    now = datetime.now(UTC)
    content = f"{_SUMMARY_CHUNK_PREFIX}{summary}"
    existing = next((c for c in all_chunks if c.chunk_type == CHUNK_TYPE_SUMMARY), None)
    if existing is not None:
        refreshed = existing.model_copy(
            update={
                "content": content,
                "source_content": content,
                "is_enabled": True,
                "updated_at": now,
            }
        )
        stored = await chunk_repo.update(refreshed)
    else:
        max_index = max((c.chunk_index for c in all_chunks), default=0)
        summary_chunk = Chunk(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            content=content,
            chunk_index=max_index + 1,
            is_enabled=True,
            start_at=0,
            end_at=0,
            pre_chunk_id=None,
            next_chunk_id=None,
            chunk_type=CHUNK_TYPE_SUMMARY,
            parent_chunk_id=first_chunk.id,
            image_info=None,
            relation_chunks=None,
            indirect_relation_chunks=None,
            metadata=None,
            tag_id=None,
            status=1,
            content_hash=None,
            flags=1,
            seq_id=0,
            source_content=content,
            content_revision=0,
            index_status="ready",
            last_editor_id="",
            context_header="",
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        stored = await chunk_repo.create(summary_chunk)
    if summary_chunk_syncer is not None:
        await summary_chunk_syncer.sync_summary_chunk(tenant_id=tenant_id, chunk=stored)
    return stored.id


async def _summarize(
    chat: Chat,
    prompt: str,
    content: str,
    max_tokens: int,
    language: str,
) -> str:
    """Run the summary chat call and return the model's output text.

    The summary system prompt is the injected ``prompt`` (a blank prompt
    yields an empty system message, mirroring the upstream default).
    """
    summary_prompt = render_placeholders(prompt, {"language": language_name_for(language)})
    response = await chat.chat(
        [
            Message(role="system", content=summary_prompt),
            Message(role="user", content=content),
        ],
        ChatOptions(temperature=_SUMMARY_TEMPERATURE, max_tokens=max_tokens, thinking=False),
    )
    return response.content


async def process_summary(
    *,
    tenant_id: int,
    knowledge_id: str,
    chat: Chat,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    kb_service: KBService,
    prompt: str,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    language: str = DEFAULT_LANGUAGE,
    summary_chunk_syncer: SummaryChunkSyncer | None = None,
) -> SummaryResult:
    """Generate (or regenerate) the summary description for a knowledge item.

    The knowledge base must have a summary model configured. The row is
    marked ``processing`` first and ``completed`` (with the summary as its
    description) once the model output is published. Content below the
    real-text threshold marks the row ``failed`` and raises
    ``ValidationError``; a concurrent edit of the source chunks / metadata
    while the model ran raises ``ConflictError`` and leaves
    ``summary_status`` untouched so a newer run can finish.

    Raises ``NotFoundError`` for an absent document, and
    ``ValidationError`` for a blank scope, an unconfigured summary model,
    or an empty source.
    """
    _require_tenant_id(tenant_id)
    _require_knowledge_id(knowledge_id)
    row = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if row is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=row.knowledge_base_id)
    if not kb.summary_model_id:
        raise ValidationError(
            code="knowledge.summary_model_not_configured",
            message="summary model is not configured",
        )

    text_chunks = await chunk_repo.list_by_knowledge_id(tenant_id, knowledge_id)
    enabled_text = [chunk for chunk in text_chunks if chunk.is_enabled]
    if not enabled_text:
        raise ValidationError(
            code="knowledge.summary_no_text_chunks",
            message="no enabled text chunks to summarize",
        )
    all_chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": knowledge_id}
    )

    now = datetime.now(UTC)
    await _mark_summary_status(
        knowledge_repo=knowledge_repo,
        row=row,
        status=SUMMARY_STATUS_PROCESSING,
        now=now,
    )
    metadata_version = _metadata_version(row.custom_metadata)

    content = sample_long_content(
        _reconstruct_content(_sort_chunks_for_summary(enabled_text)), max_input_chars
    )
    if real_text_rune_count(content) < MIN_TEXT_CONTENT_RUNES:
        # Scanned PDF with no OCR / caption: do not call the model (it
        # would hallucinate from the file name alone). Mark failed with an
        # empty description and raise.
        await _mark_summary_status(
            knowledge_repo=knowledge_repo,
            row=row,
            status=SUMMARY_STATUS_FAILED,
            now=datetime.now(UTC),
            description="",
        )
        raise ValidationError(
            code="knowledge.summary_insufficient_content",
            message="insufficient text content for summary generation",
        )

    content_with_metadata = content
    custom = custom_metadata_text(row.custom_metadata)
    if custom:
        content_with_metadata = f"Document metadata:\n{custom}\n\nDocument content:\n{content}"
    content_with_metadata = sample_long_content(content_with_metadata, max_input_chars)

    try:
        summary = await _summarize(chat, prompt, content_with_metadata, max_tokens, language)
    except Exception:
        # LLM / IO failure: fall back to the first chunk's opening text
        # rather than failing the row, mirroring the upstream fallback.
        summary = enabled_text[0].content[:_FALLBACK_SUMMARY_CHARS]

    if await _source_changed(
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        metadata_version=metadata_version,
        source_chunks=enabled_text,
    ):
        raise ConflictError(
            code="knowledge.summary_superseded",
            message=(
                "summary source changed while generating; discard this "
                "result and leave summary_status untouched"
            ),
        )

    updated = await _mark_summary_status(
        knowledge_repo=knowledge_repo,
        row=row,
        status=SUMMARY_STATUS_COMPLETED,
        now=datetime.now(UTC),
        description=summary,
    )

    summary_chunk_id: str | None = None
    if summary.strip() and needs_embedding_model(kb):
        summary_chunk_id = await _upsert_summary_chunk(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=row.knowledge_base_id,
            all_chunks=all_chunks,
            first_chunk=enabled_text[0],
            summary=summary,
            chunk_repo=chunk_repo,
            summary_chunk_syncer=summary_chunk_syncer,
        )
    return SummaryResult(
        knowledge=_to_knowledge(updated),
        summary=summary,
        summary_chunk_id=summary_chunk_id,
    )


__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_MAX_INPUT_CHARS",
    "DEFAULT_MAX_TOKENS",
    "MIN_TEXT_CONTENT_RUNES",
    "SummaryChunkSyncer",
    "SummaryResult",
    "custom_metadata_text",
    "language_name_for",
    "needs_embedding_model",
    "process_summary",
    "real_text_rune_count",
    "render_placeholders",
    "sample_long_content",
    "strip_image_markup",
]
