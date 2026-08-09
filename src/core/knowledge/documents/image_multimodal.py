"""Image multimodal processing — VLM caption/OCR and child-chunk indexing.

Processes one image reference from a parent chunk through the vision
model: the OCR prompt extracts body text (sanitised into Markdown), the
caption prompt produces a short description, and the results are
persisted as ``image_ocr`` / ``image_caption`` child chunks and indexed
through the retrieval composite. Orphaned or user-aborted work is
dropped before the model is touched, and a per-knowledge pending gate is
decremented once per image so the parent knowledge cannot strand in
``processing``.

The VLM client is an injected seam (``src.ai.vlm.VLM``); this module
never calls a provider directly. Image bytes come from an injected file
service for ``provider://`` URLs or an HTTP(S) downloader; indexing
flows through the ``IndexEngine`` seam (satisfied by the composite
retrieve engine). Captions are always generated — the OCR pass alone is
optional.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import ClassVar, Protocol, runtime_checkable

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.types import IndexInfo, SourceType
from src.ai.vlm.base import VLM
from src.app_logging import logger
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_DEFAULT,
    CHUNK_STATUS_INDEXED,
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
)
from src.core.knowledge.documents.chunk_extract import append_custom_prompt_instructions
from src.core.knowledge.documents.image_update import ImageInfo
from src.core.knowledge.documents.index_pipeline import IndexEngine
from src.core.knowledge.documents.process_document import kb_needs_embedding
from src.core.knowledge.documents.summary import DEFAULT_LANGUAGE, language_name_for
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_DELETING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

# ── OCR prompt templates ──────────────────────────────────────────────

VLM_OCR_PROMPT = (
    "<system_prompt>\n"
    "You are an OCR assistant. Your task is to extract all body text content "
    "from this document image and output in pure Markdown format.\n"
    "</system_prompt>\n\n"
    "<instructions>\n"
    "1. Ignore headers and footers.\n"
    "2. Use Markdown table syntax for tables.\n"
    "3. Use LaTeX format for formulas (wrapped with $ or $$).\n"
    "4. Organize content in the original reading order.\n"
    "5. Output ONLY the extracted text content. Do NOT include any HTML tags, "
    "reasoning, or unrelated comments.\n"
    "6. If there is absolutely no recognizable text content in the image, "
    "reply ONLY with: No text content.\n"
    "</instructions>"
)

VLM_OCR_SCANNED_PDF_PROMPT = (
    "<system_prompt>\n"
    "You are an OCR and document layout extraction assistant. The input image "
    "is a page from a scanned PDF document.\n"
    "Your task is to carefully extract all text and layout structure from the "
    "image, and output the result in pure Markdown format.\n"
    "</system_prompt>\n\n"
    "<instructions>\n"
    "1. Ignore headers, footers, and page numbers.\n"
    "2. Preserve the original document's paragraph and hierarchical structure "
    "as much as possible.\n"
    "3. If there are tables, use Markdown table syntax to represent them.\n"
    "4. If there are mathematical formulas, use LaTeX format wrapped in $ or $$.\n"
    "5. Output ONLY the extracted text content. Do NOT include any HTML tags, "
    "reasoning, or unrelated comments.\n"
    "6. If there is absolutely no recognizable text content in the image, "
    "reply ONLY with: No text content.\n"
    "</instructions>"
)

#: Image source marker that selects the scanned-PDF OCR prompt.
IMAGE_SOURCE_SCANNED_PDF = "scanned_pdf"


# ── OCR output sanitising ─────────────────────────────────────────────

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_CODE_BLOCK_PATTERN = re.compile(r"(?s)^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$")
_HTML_DOC_PATTERN = re.compile(
    r"(?i)^\s*(<!DOCTYPE|<html|<body|<div|<p[\s>]|<table|<h[1-6][\s>])"
)
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")

#: Replies a vision model produces when the image has no recognizable text.
_KNOWN_EMPTY_REPLIES: frozenset[str] = frozenset(
    {
        "无文字内容",
        "无法识别",
        "no text",
        "no text content",
        "no content",
        "empty",
        "图片中没有文字",
        "图片中没有可识别的文字",
    }
)


def strip_markdown_code_block(text: str) -> str:
    """Remove a markdown code-fence wrapper some models add around output."""
    match = _CODE_BLOCK_PATTERN.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def looks_like_html(text: str) -> bool:
    """True when the text appears to be an HTML document or carries heavy tags."""
    if _HTML_DOC_PATTERN.match(text):
        return True
    tags = _HTML_TAG_PATTERN.findall(text)
    if not tags:
        return False
    tag_chars = sum(len(tag) for tag in tags)
    return tag_chars / len(text) > 0.3


def is_known_empty_reply(text: str) -> bool:
    """True when the text matches a known "no content" vision-model reply.

    Trailing punctuation is stripped before comparison so that responses
    like ``No text content.`` still match ``no text content``.
    """
    lower = text.strip().lower()
    lower = lower.rstrip(".!?。！？")  # noqa: RUF001
    return lower in _KNOWN_EMPTY_REPLIES


def sanitize_ocr_text(raw: str) -> str:
    """Clean vision-model OCR output: strip HTML wrappers, convert, drop noise.

    Byte length matches the reference behaviour for the "HTML with almost
    no text" gate so CJK replies whose stripped text is short but
    meaningful survive the check.
    """
    text = raw.strip()
    if text == "":
        return ""
    text = strip_markdown_code_block(text)

    # If stripping HTML tags leaves almost no text, the response is useless
    # (e.g. an image placeholder skeleton).
    plain_text = _HTML_TAG_PATTERN.sub("", text).strip()
    if len(plain_text.encode("utf-8")) < 10 and _HTML_TAG_PATTERN.search(text):
        return ""

    if looks_like_html(text):
        text = ocr_html_to_markdown(text).strip()
        if text == "":
            return ""

    if is_known_empty_reply(text):
        return ""

    text = _MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


class _HTMLToMarkdown(HTMLParser):
    """Convert an HTML fragment to Markdown (headings, tables, lists, ...).

    A focused subset of the OCR HTML-to-Markdown conversion for the block
    elements vision-model output produces. Unrecognised markup falls
    through to its text content.
    """

    _HEADING_LEVELS: ClassVar[dict[str, int]] = {
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "h4": 4,
        "h5": 5,
        "h6": 6,
    }
    _INLINE_MARKERS: ClassVar[dict[str, str]] = {
        "strong": "**",
        "b": "**",
        "em": "*",
        "i": "*",
        "code": "`",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._block_open = False
        self._list_depth = 0
        self._ordered: list[bool] = []
        self._counters: list[int] = []
        self._table_rows: list[list[str]] = []
        self._table_row: list[str] = []
        self._table_cell: list[str] = []
        self._in_table_cell = False
        self._in_pre = False
        self._pre_buffer: list[str] = []
        self._link_stack: list[str] = []

    def result(self) -> str:
        return "".join(self._out)

    # ── output helpers ──────────────────────────────────────────────

    def _flush_block(self) -> None:
        if self._block_open and not self._in_table_cell and not self._in_pre:
            self._out.append("\n\n")
            self._block_open = False

    def _emit(self, text: str) -> None:
        if self._in_table_cell:
            self._table_cell.append(text)
        elif self._in_pre:
            self._pre_buffer.append(text)
        else:
            self._out.append(text)

    def _emit_block(self, text: str) -> None:
        if self._in_table_cell or self._in_pre:
            self._emit(text)
            return
        self._flush_block()
        self._out.append(text)
        self._block_open = True

    # ── parser callbacks ────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._HEADING_LEVELS:
            self._flush_block()
            self._emit("#" * self._HEADING_LEVELS[tag] + " ")
        elif tag in ("p", "div", "section", "article", "header", "footer"):
            self._flush_block()
            self._block_open = True
        elif tag == "br":
            self._emit("\n")
        elif tag == "hr":
            self._emit_block("---")
        elif tag in self._INLINE_MARKERS:
            if not (tag == "code" and self._in_pre):
                self._emit(self._INLINE_MARKERS[tag])
        elif tag == "a":
            self._link_stack.append(dict(attrs).get("href") or "")
            self._emit("[")
        elif tag == "img":
            attributes = dict(attrs)
            self._emit(f"![{attributes.get('alt') or ''}]({attributes.get('src') or ''})")
        elif tag == "ul":
            self._list_depth += 1
            self._ordered.append(False)
            self._counters.append(0)
            self._flush_block()
        elif tag == "ol":
            self._list_depth += 1
            self._ordered.append(True)
            self._counters.append(1)
            self._flush_block()
        elif tag == "li":
            self._flush_block()
            indent = "    " * (self._list_depth - 1)
            if self._ordered and self._ordered[-1]:
                number = self._counters[-1]
                self._counters[-1] = number + 1
                self._emit(f"{indent}{number}. ")
            else:
                self._emit(f"{indent}- ")
            self._block_open = True
        elif tag == "blockquote":
            self._emit("> ")
            self._block_open = True
        elif tag == "pre":
            self._flush_block()
            self._in_pre = True
            self._pre_buffer = []
            self._out.append("```\n")
        elif tag == "table":
            self._table_rows = []
        elif tag == "tr":
            self._table_row = []
        elif tag in ("td", "th"):
            self._in_table_cell = True
            self._table_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._HEADING_LEVELS:
            self._flush_block()
            self._block_open = True
        elif tag in ("p", "div", "section", "article", "header", "footer"):
            self._flush_block()
        elif tag in self._INLINE_MARKERS:
            if not (tag == "code" and self._in_pre):
                self._emit(self._INLINE_MARKERS[tag])
        elif tag == "a":
            href = self._link_stack.pop() if self._link_stack else ""
            if href:
                self._emit(f"]({href})")
            else:
                self._emit("]")
        elif tag == "li":
            self._emit("\n")
            self._block_open = False
        elif tag in ("ul", "ol"):
            if self._list_depth > 0:
                self._list_depth -= 1
            if self._ordered:
                self._ordered.pop()
            if self._counters:
                self._counters.pop()
            self._flush_block()
            self._block_open = True
        elif tag == "blockquote":
            self._flush_block()
        elif tag == "pre":
            self._in_pre = False
            self._out.append("".join(self._pre_buffer).rstrip())
            self._out.append("\n```")
            self._block_open = True
        elif tag in ("td", "th"):
            self._in_table_cell = False
            self._table_row.append("".join(self._table_cell).strip())
        elif tag == "tr":
            self._table_rows.append(self._table_row)
        elif tag == "table":
            self._emit_block(self._render_table())

    def handle_data(self, data: str) -> None:
        self._emit(data)

    def _render_table(self) -> str:
        if not self._table_rows:
            return ""
        rows = [[cell.replace("|", "\\|") for cell in row] for row in self._table_rows]
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        lines = ["| " + " | ".join(padded[0]) + " |"]
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
        return "\n".join(lines)


def ocr_html_to_markdown(content: str) -> str:
    """Convert HTML to Markdown, falling back to the original on failure."""
    converter = _HTMLToMarkdown()
    try:
        converter.feed(content)
        converter.close()
    except Exception:
        return content
    return converter.result()


# ── VLM configuration ─────────────────────────────────────────────────


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class VLMConfig:
    """Effective vision-model configuration (new-style and legacy).

    ``description_language`` and ``custom_instructions`` are carried into
    the caption / OCR prompts; the legacy ``model_name`` + ``base_url``
    pair supports pre-existing inline configurations.
    """

    enabled: bool = False
    model_id: str = ""
    description_language: str = ""
    custom_instructions: str = ""
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    interface_type: str = ""

    def is_enabled(self) -> bool:
        """True for a new-style (``enabled`` + model id) or legacy config."""
        if self.enabled and self.model_id != "":
            return True
        return self.model_name != "" and self.base_url != ""


def vlm_config_from_json(config: JsonObject | None) -> VLMConfig:
    """Decode a ``vlm_config`` JSON blob onto the effective config shape."""
    raw = config if isinstance(config, dict) else {}
    return VLMConfig(
        enabled=isinstance(raw.get("enabled"), bool) and bool(raw["enabled"]),
        model_id=_as_str(raw.get("model_id")),
        description_language=_as_str(raw.get("description_language")),
        custom_instructions=_as_str(raw.get("custom_instructions")),
        model_name=_as_str(raw.get("model_name")),
        base_url=_as_str(raw.get("base_url")),
        api_key=_as_str(raw.get("api_key")),
        interface_type=_as_str(raw.get("interface_type")),
    )


def _process_overrides(row: Document | None) -> JsonObject | None:
    if row is None:
        return None
    overrides = (row.metadata or {}).get("process_overrides")
    return overrides if isinstance(overrides, dict) else None


def resolve_vlm_config(kb: KnowledgeBaseInfo, row: Document | None) -> VLMConfig:
    """Merge the KB's vision config with a per-upload override.

    An override replaces the config wholesale except for
    ``description_language`` and ``custom_instructions``, which fall back
    to the KB values when the override leaves them blank.
    """
    base = vlm_config_from_json(kb.vlm_config)
    raw = (_process_overrides(row) or {}).get("vlm_config")
    if not isinstance(raw, dict):
        return base
    override = vlm_config_from_json(raw)
    return VLMConfig(
        enabled=override.enabled,
        model_id=override.model_id,
        description_language=override.description_language or base.description_language,
        custom_instructions=override.custom_instructions or base.custom_instructions,
        model_name=override.model_name,
        base_url=override.base_url,
        api_key=override.api_key,
        interface_type=override.interface_type,
    )


def resolve_caption_language(cfg: VLMConfig, locale: str = "") -> str:
    """Resolve the human-readable caption language for the effective config.

    The config's ``description_language`` wins; otherwise the payload
    locale is mapped to a language name; an absent locale falls back to
    the process default.
    """
    language = cfg.description_language.strip()
    if language != "":
        return language
    locale = locale.strip()
    if locale != "":
        return language_name_for(locale)
    return language_name_for(DEFAULT_LANGUAGE)


def build_ocr_prompt(image_source_type: str, custom_instructions: str = "") -> str:
    """Build the OCR prompt for an image, appending business instructions."""
    prompt = (
        VLM_OCR_SCANNED_PDF_PROMPT
        if image_source_type == IMAGE_SOURCE_SCANNED_PDF
        else VLM_OCR_PROMPT
    )
    return append_custom_prompt_instructions(prompt, custom_instructions, "image_ocr")


def build_vlm_caption_prompt(language: str, custom_instructions: str = "") -> str:
    """Build the caption (description) prompt for an image."""
    prompt = (
        "Provide a brief and concise description of the main content of the "
        f"image in {language}."
    )
    return append_custom_prompt_instructions(prompt, custom_instructions, "image_description")


# ── Image reference parsing ───────────────────────────────────────────

#: Provider schemes carried by stored object paths (``provider://...``).
_PROVIDER_SCHEMES: tuple[str, ...] = (
    "local",
    "minio",
    "cos",
    "tos",
    "s3",
    "oss",
    "ks3",
    "obs",
    "dummy",
)
_STORAGE_BACKEND_SCHEME = "storage://"
_RESOURCE_SCHEME = "resource://"
_RESOURCE_HANDLE_LENGTH = 22


def parse_storage_backend_path(path: str) -> tuple[str, str, bool]:
    """Split a ``storage://<backend>/<provider://...>`` path.

    Returns ``(backend_id, provider_path, ok)``.
    """
    if not path.startswith(_STORAGE_BACKEND_SCHEME):
        return "", "", False
    rest = path[len(_STORAGE_BACKEND_SCHEME) :]
    backend_id, sep, provider_path = rest.partition("/")
    if sep == "" or backend_id == "" or provider_path == "":
        return "", "", False
    return backend_id, provider_path, True


def parse_provider_scheme(file_path: str) -> str:
    """Return the ``provider`` of a ``provider://...`` path, or ``""``."""
    _backend_id, inner, ok = parse_storage_backend_path(file_path)
    if ok:
        file_path = inner
    for provider in _PROVIDER_SCHEMES:
        if file_path.startswith(provider + "://"):
            return provider
    return ""


def parse_resource_path(value: str) -> tuple[str, bool]:
    """Return the handle of a ``resource://`` reference, or ``("", False)``."""
    value = value.strip()
    if not value.startswith(_RESOURCE_SCHEME):
        return "", False
    handle = value[len(_RESOURCE_SCHEME) :]
    if len(handle) != _RESOURCE_HANDLE_LENGTH:
        return "", False
    for char in handle:
        if not (char.isascii() and (char.isalnum() or char in "_-")):
            return "", False
    return handle, True


def is_resource_reference(value: str) -> bool:
    """True when ``value`` is a valid ``resource://`` reference."""
    return parse_resource_path(value)[1]


# ── Payload / outcome ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImageMultimodalPayload:
    """One image to run through OCR/caption and index as child chunks."""

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    chunk_id: str
    image_url: str
    image_local_path: str = ""
    enable_ocr: bool = False
    enable_caption: bool = False
    language: str = ""
    image_source_type: str = ""


@dataclass(frozen=True, slots=True)
class ImageMultimodalOutcome:
    """Result of one image multimodal pass."""

    ocr_text: str = ""
    caption: str = ""
    image_bytes: int = 0
    chunks_created: int = 0
    indexed: bool = False
    skipped: str = ""
    read_error: str = ""
    ocr_error: str = ""
    caption_error: str = ""
    vlm_model_id: str = ""
    ocr_chars: int = 0
    caption_chars: int = 0
    ocr_skipped: str = ""


# ── Injectable seams ──────────────────────────────────────────────────


class ImageReadError(Exception):
    """Raised when an image cannot be read; treated as a skip, not a failure."""


@runtime_checkable
class VLMModelResolver(Protocol):
    """Build a VLM client for an effective config (model-based or legacy)."""

    async def resolve(self, *, config: VLMConfig) -> VLM | None:
        """Return a usable VLM client, or ``None`` when unavailable."""
        ...


@runtime_checkable
class ImageBytesReader(Protocol):
    """Read a stored image object through its storage backend."""

    async def get_file(self, *, url: str) -> bytes | None:
        """Return the object bytes, or ``None`` when not found."""
        ...


@runtime_checkable
class ImageUrlDownloader(Protocol):
    """SSRF-safe HTTP(S) image downloader."""

    async def download(self, *, url: str) -> bytes:
        """Download the image bytes from an HTTP(S) URL."""
        ...


@runtime_checkable
class EmbeddingResolver(Protocol):
    """Resolve the embedder for a knowledge base's embedding model."""

    async def resolve_embedder(self, *, embedding_model_id: str) -> Embedder | None:
        """Return the embedder, or ``None`` when unavailable."""
        ...


@runtime_checkable
class IndexEngineResolver(Protocol):
    """Resolve the retrieval index engine for a knowledge base."""

    async def resolve_engine(
        self, *, tenant_id: int, vector_store_id: str | None
    ) -> IndexEngine | None:
        """Return the index engine, or ``None`` when unavailable."""
        ...


@runtime_checkable
class MultimodalFinalizer(Protocol):
    """Count one finished image toward the parent knowledge's pending gate."""

    async def finalize(
        self, *, tenant_id: int, knowledge_id: str, knowledge_base_id: str
    ) -> None:
        """Decrement the pending-image counter; enqueue post-process when drained."""
        ...


# ── Child chunk construction ──────────────────────────────────────────


def build_multimodal_chunks(
    *,
    payload: ImageMultimodalPayload,
    image_info: ImageInfo,
    now: datetime,
) -> list[Chunk]:
    """Build ``image_ocr`` / ``image_caption`` child chunks for ``image_info``.

    The image record JSON is attached to every child chunk so the image
    relationship survives re-parses. Empty OCR / caption text produces no
    corresponding chunk.
    """
    image_info_json = json.dumps([image_info.model_dump()], ensure_ascii=False)
    chunks: list[Chunk] = []
    if image_info.ocr_text != "":
        chunks.append(
            _new_child_chunk(
                payload,
                CHUNK_TYPE_IMAGE_OCR,
                image_info.ocr_text,
                image_info_json,
                now,
            )
        )
    if image_info.caption != "":
        chunks.append(
            _new_child_chunk(
                payload,
                CHUNK_TYPE_IMAGE_CAPTION,
                image_info.caption,
                image_info_json,
                now,
            )
        )
    return chunks


def _new_child_chunk(
    payload: ImageMultimodalPayload,
    chunk_type: str,
    content: str,
    image_info_json: str,
    now: datetime,
) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        tenant_id=payload.tenant_id,
        knowledge_base_id=payload.knowledge_base_id,
        knowledge_id=payload.knowledge_id,
        content=content,
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=0,
        chunk_type=chunk_type,
        parent_chunk_id=payload.chunk_id,
        image_info=image_info_json,
        status=CHUNK_STATUS_DEFAULT,
        flags=CHUNK_FLAG_RECOMMENDED,
        created_at=now,
        updated_at=now,
    )


def _build_index_info(chunk: Chunk) -> IndexInfo:
    """Build the retrieval index entry for one multimodal child chunk."""
    return IndexInfo(
        content=chunk.content,
        source_id=chunk.id,
        source_type=SourceType.CHUNK,
        chunk_id=chunk.id,
        knowledge_id=chunk.knowledge_id,
        knowledge_base_id=chunk.knowledge_base_id,
    )


def _read_local_file(path: str) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


# ── Service ───────────────────────────────────────────────────────────


class ImageMultimodalService:
    """Processes one image into OCR/caption child chunks and indexes them.

    Dependencies are injected seams: the chunk service and knowledge-base
    service are concrete request-scoped services, while the VLM, image
    readers, index resolvers and finalizer are protocols a worker layer
    wires per request.
    """

    def __init__(
        self,
        *,
        chunk_service: ChunkService,
        kb_service: KBService,
        knowledge_repo: KnowledgeRepository | None = None,
        vlm_resolver: VLMModelResolver | None = None,
        file_reader: ImageBytesReader | None = None,
        url_downloader: ImageUrlDownloader | None = None,
        embedding_resolver: EmbeddingResolver | None = None,
        index_engine_resolver: IndexEngineResolver | None = None,
        finalizer: MultimodalFinalizer | None = None,
    ) -> None:
        self._chunk_service = chunk_service
        self._kb_service = kb_service
        self._knowledge_repo = knowledge_repo
        self._vlm_resolver = vlm_resolver
        self._file_reader = file_reader
        self._url_downloader = url_downloader
        self._embedding_resolver = embedding_resolver
        self._index_engine_resolver = index_engine_resolver
        self._finalizer = finalizer

    async def process_image(
        self, *, ctx: Context, payload: ImageMultimodalPayload
    ) -> ImageMultimodalOutcome:
        """Run OCR/caption on ``payload`` and persist/index the child chunks.

        Orphaned or aborted work is dropped before the model is touched;
        the pending gate is finalised once per attempt regardless of the
        inner outcome.
        """
        if await self.should_drop_orphaned(payload=payload):
            logger.info(
                "[ImageMultimodal] Dropping task chunk={} knowledge={} kb={} image={}",
                payload.chunk_id,
                payload.knowledge_id,
                payload.knowledge_base_id,
                payload.image_url,
            )
            await self._finalize(payload=payload)
            return ImageMultimodalOutcome(skipped="orphaned")
        try:
            return await self._process(ctx=ctx, payload=payload)
        finally:
            await self._finalize(payload=payload)

    async def should_drop_orphaned(self, *, payload: ImageMultimodalPayload) -> bool:
        """True when the task should exit without retrying.

        True for cancelled / deleting knowledge, or when the parent
        knowledge / knowledge-base row no longer exists (queue entries
        survived a delete).
        """
        if payload.knowledge_id != "" and self._knowledge_repo is not None:
            row = await self._knowledge_repo.get_by_id_only(payload.knowledge_id)
            if row is None:
                return True
            if row.parse_status in (PARSE_STATUS_CANCELLED, PARSE_STATUS_DELETING):
                return True
        if payload.knowledge_base_id != "" and self._kb_service is not None:
            try:
                kb = await self._kb_service.get_knowledge_base_by_id_only(
                    knowledge_base_id=payload.knowledge_base_id
                )
            except NotFoundError:
                return True
            if kb is None:
                return True
        return False

    async def _process(
        self, *, ctx: Context, payload: ImageMultimodalPayload
    ) -> ImageMultimodalOutcome:
        kb = await self._kb_service.get_knowledge_base_by_id_only(
            knowledge_base_id=payload.knowledge_base_id
        )
        row = await self._load_knowledge(payload.knowledge_id)
        config = resolve_vlm_config(kb, row)
        if not config.is_enabled():
            raise ValidationError(
                code="image_multimodal.vlm_disabled",
                message=f"VLM is not enabled for knowledge base {payload.knowledge_base_id}",
            )
        vlm = await self._resolve_vlm(config)
        outcome = ImageMultimodalOutcome(vlm_model_id=config.model_id or "legacy_inline")

        try:
            image_bytes = await self._read_image_bytes(payload)
        except ImageReadError as exc:
            logger.warning(
                "[ImageMultimodal] Skip unreadable image {}: {}", payload.image_url, exc
            )
            return replace(outcome, skipped="unreadable_image", read_error=str(exc))
        outcome = replace(outcome, image_bytes=len(image_bytes))

        image_info = ImageInfo(url=payload.image_url, original_url=payload.image_url)

        if payload.enable_ocr:
            outcome, image_info = await self._run_ocr(
                payload, config, vlm, image_bytes, image_info, outcome
            )

        caption_prompt = build_vlm_caption_prompt(
            resolve_caption_language(config, payload.language),
            config.custom_instructions,
        )
        try:
            caption = await vlm.predict([image_bytes], caption_prompt)
        except Exception as exc:
            logger.warning(
                "[ImageMultimodal] Caption failed for {}: {}", payload.image_url, exc
            )
            outcome = replace(outcome, caption_error=str(exc))
        else:
            if caption != "":
                image_info = image_info.model_copy(update={"caption": caption})
                outcome = replace(outcome, caption=caption, caption_chars=len(caption))

        chunks = build_multimodal_chunks(
            payload=payload, image_info=image_info, now=datetime.now(UTC)
        )
        outcome = replace(outcome, chunks_created=len(chunks))
        if not chunks:
            return replace(outcome, skipped="no_extracted_content")

        await self._chunk_service.create_chunks(chunks=chunks)
        indexed = await self._index_chunks(ctx=ctx, kb=kb, payload=payload, chunks=chunks)
        return replace(outcome, indexed=indexed)

    async def _run_ocr(
        self,
        payload: ImageMultimodalPayload,
        config: VLMConfig,
        vlm: VLM,
        image_bytes: bytes,
        image_info: ImageInfo,
        outcome: ImageMultimodalOutcome,
    ) -> tuple[ImageMultimodalOutcome, ImageInfo]:
        ocr_prompt = build_ocr_prompt(payload.image_source_type, config.custom_instructions)
        try:
            ocr_text = sanitize_ocr_text(await vlm.predict([image_bytes], ocr_prompt))
        except Exception as exc:
            logger.warning(
                "[ImageMultimodal] OCR failed for {}: {}", payload.image_url, exc
            )
            return replace(outcome, ocr_error=str(exc)), image_info
        if ocr_text == "":
            logger.warning(
                "[ImageMultimodal] OCR returned empty/invalid content for {}, discarded",
                payload.image_url,
            )
            return replace(outcome, ocr_skipped="empty_or_invalid"), image_info
        updated = image_info.model_copy(update={"ocr_text": ocr_text})
        return replace(outcome, ocr_text=ocr_text, ocr_chars=len(ocr_text)), updated

    async def _load_knowledge(self, knowledge_id: str) -> Document | None:
        if knowledge_id == "" or self._knowledge_repo is None:
            return None
        try:
            return await self._knowledge_repo.get_by_id_only(knowledge_id)
        except Exception as exc:
            logger.warning(
                "[ImageMultimodal] Failed to load knowledge {} for overrides: {}",
                knowledge_id,
                exc,
            )
            return None

    async def _resolve_vlm(self, config: VLMConfig) -> VLM:
        if self._vlm_resolver is None:
            raise ValidationError(
                code="image_multimodal.vlm_unavailable",
                message="no VLM resolver configured",
            )
        vlm = await self._vlm_resolver.resolve(config=config)
        if vlm is None:
            raise ValidationError(
                code="image_multimodal.vlm_unavailable",
                message="VLM model unavailable",
            )
        return vlm

    async def _read_image_bytes(self, payload: ImageMultimodalPayload) -> bytes:
        """Read the image bytes for a payload.

        ``provider://`` URLs (and ``resource://`` references) are read
        through the injected file service and are never handed to the
        HTTP downloader; a legacy local path is tried before falling back
        to the URL.
        """
        url = payload.image_url
        if is_resource_reference(url) or parse_provider_scheme(url) != "":
            if self._file_reader is None:
                raise ImageReadError(f"no file service available for {url}")
            data = await self._file_reader.get_file(url=url)
            if data is None:
                raise ImageReadError(f"file service get {url}: not found")
            return data

        if payload.image_local_path != "":
            data = _read_local_file(payload.image_local_path)
            if data is not None:
                return data
            logger.warning(
                "[ImageMultimodal] Local file {} not available, falling back to URL",
                payload.image_local_path,
            )

        if self._url_downloader is None:
            raise ImageReadError(f"no downloader available for {url}")
        try:
            return await self._url_downloader.download(url=url)
        except Exception as exc:
            raise ImageReadError(f"download {url}: {exc}") from exc

    async def _index_chunks(
        self,
        *,
        ctx: Context,
        kb: KnowledgeBaseInfo,
        payload: ImageMultimodalPayload,
        chunks: list[Chunk],
    ) -> bool:
        """Index the multimodal chunks; marks them indexed when successful."""
        if not kb_needs_embedding(kb.indexing_strategy):
            logger.info(
                "[ImageMultimodal] Vector/keyword indexing disabled for KB {}, "
                "skipping index for {} multimodal chunks",
                kb.id,
                len(chunks),
            )
            await self._mark_chunks_indexed(chunks)
            return True

        embedder = await self._resolve_embedder(kb)
        if embedder is None:
            return False
        engine = await self._resolve_engine(payload, kb)
        if engine is None:
            return False

        index_info_list = [_build_index_info(chunk) for chunk in chunks]
        try:
            await engine.batch_index(ctx, embedder, index_info_list)
        except Exception as exc:
            logger.error("[ImageMultimodal] Failed to index multimodal chunks: {}", exc)
            return False

        await self._mark_chunks_indexed(chunks)
        return True

    async def _resolve_embedder(self, kb: KnowledgeBaseInfo) -> Embedder | None:
        if self._embedding_resolver is None:
            return None
        embedder = await self._embedding_resolver.resolve_embedder(
            embedding_model_id=kb.embedding_model_id
        )
        if embedder is None:
            logger.warning(
                "[ImageMultimodal] Failed to get embedding model for indexing"
            )
        return embedder

    async def _resolve_engine(
        self, payload: ImageMultimodalPayload, kb: KnowledgeBaseInfo
    ) -> IndexEngine | None:
        if self._index_engine_resolver is None:
            return None
        engine = await self._index_engine_resolver.resolve_engine(
            tenant_id=payload.tenant_id,
            vector_store_id=kb.vector_store_id,
        )
        if engine is None:
            logger.warning("[ImageMultimodal] Failed to init retrieve engine for indexing")
        return engine

    async def _mark_chunks_indexed(self, chunks: list[Chunk]) -> None:
        """Mark persisted chunks as indexed, re-fetching from the DB first."""
        for chunk in chunks:
            try:
                stored = await self._chunk_service.get_chunk_by_id_only(id=chunk.id)
            except NotFoundError:
                continue
            await self._chunk_service.update_chunk(
                chunk=stored.model_copy(update={"status": CHUNK_STATUS_INDEXED})
            )

    async def _finalize(self, *, payload: ImageMultimodalPayload) -> None:
        if self._finalizer is None:
            return
        try:
            await self._finalizer.finalize(
                tenant_id=payload.tenant_id,
                knowledge_id=payload.knowledge_id,
                knowledge_base_id=payload.knowledge_base_id,
            )
        except Exception as exc:
            logger.warning(
                "[ImageMultimodal] Finalize failed for knowledge {}: {}",
                payload.knowledge_id,
                exc,
            )


__all__ = [
    "IMAGE_SOURCE_SCANNED_PDF",
    "VLM_OCR_PROMPT",
    "VLM_OCR_SCANNED_PDF_PROMPT",
    "EmbeddingResolver",
    "ImageBytesReader",
    "ImageMultimodalOutcome",
    "ImageMultimodalPayload",
    "ImageMultimodalService",
    "ImageReadError",
    "ImageUrlDownloader",
    "IndexEngineResolver",
    "MultimodalFinalizer",
    "VLMConfig",
    "VLMModelResolver",
    "build_multimodal_chunks",
    "build_ocr_prompt",
    "build_vlm_caption_prompt",
    "is_known_empty_reply",
    "is_resource_reference",
    "looks_like_html",
    "ocr_html_to_markdown",
    "parse_provider_scheme",
    "parse_resource_path",
    "parse_storage_backend_path",
    "resolve_caption_language",
    "resolve_vlm_config",
    "sanitize_ocr_text",
    "strip_markdown_code_block",
    "vlm_config_from_json",
]
