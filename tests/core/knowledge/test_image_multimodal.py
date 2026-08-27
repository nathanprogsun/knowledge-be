"""Unit and integration tests for the image multimodal service.

Unit tests drive the standalone module with stateful repository mocks
(closure-captured storage, the same pattern used across the core service
tests), a scripted VLM seam, and a real composite retrieve engine built
over a fake engine service so the index fan-out is exercised without any
vector store. The pure sanitizer / prompt / config helpers mirror the
reference OCR sanitisation cases.

Integration tests run against the real applied schema and skip when the
database is unreachable. ``chunks`` carries an INTEGER (32-bit)
``tenant_id`` column, so those tests use an int32-safe tenant id (a local
counter) instead of ``make_test_tenant_id``'s BIGINT range.
"""

from __future__ import annotations

# Chinese test data uses fullwidth punctuation.
import itertools
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from src.ai.embedding import Context, Embedder, TaskContext
from src.ai.retrieval.composite import CompositeRetrieveEngine, new_composite_retrieve_engine
from src.ai.retrieval.types import (
    IndexInfo,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.ai.vlm.base import VLM
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_STATUS_DEFAULT,
    CHUNK_STATUS_INDEXED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.image_multimodal import (
    IMAGE_SOURCE_SCANNED_PDF,
    VLM_OCR_PROMPT,
    VLM_OCR_SCANNED_PDF_PROMPT,
    ImageBytesReader,
    ImageMultimodalOutcome,
    ImageMultimodalPayload,
    ImageMultimodalService,
    ImageUrlDownloader,
    MultimodalFinalizer,
    VLMConfig,
    build_multimodal_chunks,
    build_ocr_prompt,
    build_vlm_caption_prompt,
    is_known_empty_reply,
    is_resource_reference,
    looks_like_html,
    ocr_html_to_markdown,
    parse_provider_scheme,
    parse_resource_path,
    parse_storage_backend_path,
    resolve_caption_language,
    resolve_vlm_config,
    sanitize_ocr_text,
    strip_markdown_code_block,
    vlm_config_from_json,
)
from src.core.knowledge.documents.index_pipeline import IndexEngine
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_NONE,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids from
# this counter so they stay inside the range.
_INT32_TENANT_BASE = 5_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id that fits the ``chunks.tenant_id`` INTEGER column."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _ctx() -> Context:
    return TaskContext()


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _payload(
    *,
    tenant_id: int = 1,
    knowledge_id: str = "k-1",
    knowledge_base_id: str = "kb-1",
    chunk_id: str = "c-1",
    image_url: str = "local://images/img.png",
    **overrides: object,
) -> ImageMultimodalPayload:
    """Build a payload with the given field overrides."""
    return ImageMultimodalPayload(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        image_url=image_url,
        **overrides,  # type: ignore[arg-type]
    )


def _kb(
    *,
    tenant_id: int = 1,
    id: str = "kb-1",
    vlm_config: JsonObject | None = None,
    indexing_strategy: JsonObject | None = None,
    embedding_model_id: str = "embed-1",
    vector_store_id: str | None = None,
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo.model_validate(
        {
            "id": id,
            "name": "mm-kb",
            "type": "document",
            "tenant_id": tenant_id,
            "vlm_config": vlm_config,
            "embedding_model_id": embedding_model_id,
            "vector_store_id": vector_store_id,
            "indexing_strategy": indexing_strategy,
            "created_at": _NOW,
            "updated_at": _NOW,
        }
    )


def _doc_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    id: str = "k-1",
    parse_status: str = PARSE_STATUS_PROCESSING,
    metadata: JsonObject | None = None,
) -> Document:
    return Document.model_validate(
        {
            "id": id,
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "type": "file",
            "title": "mm-fixture",
            "description": None,
            "source": "fixture.png",
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": SUMMARY_STATUS_NONE,
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": "fixture.png",
            "file_type": "png",
            "file_size": 1024,
            "file_hash": None,
            "file_path": "obj/fixture.png",
            "storage_size": 0,
            "metadata": metadata,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
        }
    )


def _parent_chunk(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    id: str = "c-1",
) -> Chunk:
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content="![image](local://images/img.png)",
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=20,
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type=CHUNK_TYPE_TEXT,
        parent_chunk_id=None,
        image_info=None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata=None,
        tag_id=None,
        status=CHUNK_STATUS_STORED,
        content_hash=None,
        flags=1,
        seq_id=0,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        context_header="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


# ── Fake seams ─────────────────────────────────────────────────────────


class _FakeVLM:
    """Scripted VLM seam: records calls and returns canned OCR/caption."""

    def __init__(
        self,
        *,
        ocr_text: str = "",
        caption: str = "",
        ocr_error: Exception | None = None,
        caption_error: Exception | None = None,
    ) -> None:
        self.ocr_text = ocr_text
        self.caption = caption
        self.ocr_error = ocr_error
        self.caption_error = caption_error
        self.prompts: list[str] = []
        self.images: list[list[bytes]] = []

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        self.prompts.append(prompt)
        self.images.append(img_bytes)
        if "You are an OCR assistant" in prompt:
            if self.ocr_error is not None:
                raise self.ocr_error
            return self.ocr_text
        if self.caption_error is not None:
            raise self.caption_error
        return self.caption

    def get_model_name(self) -> str:
        return "fake-vlm"

    def get_model_id(self) -> str:
        return "fake-vlm-id"


class _FakeVLMResolver:
    """VLM resolver double returning a canned model per config."""

    def __init__(self, vlm: VLM | None) -> None:
        self.vlm = vlm
        self.configs: list[VLMConfig] = []

    async def resolve(self, *, config: VLMConfig) -> VLM | None:
        self.configs.append(config)
        return self.vlm


class _FakeChunkService:
    """Chunk-service double with closure-captured storage."""

    def __init__(self) -> None:
        self.rows: dict[str, Chunk] = {}
        self.created: list[Chunk] = []
        self.updated: list[Chunk] = []

    def seed(self, row: Chunk) -> None:
        self.rows[row.id] = row

    async def create_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            self.rows[chunk.id] = chunk
            self.created.append(chunk)
        return chunks

    async def get_chunk_by_id_only(self, *, id: str) -> Chunk:
        row = self.rows.get(id)
        if row is None:
            raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")
        return row

    async def update_chunk(self, *, chunk: Chunk) -> Chunk:
        self.rows[chunk.id] = chunk
        self.updated.append(chunk)
        return chunk


class _FakeKnowledgeRepo:
    """Knowledge-repository double returning a canned row."""

    def __init__(self, row: Document | None = None, missing: bool = False) -> None:
        self.row = row
        self.missing = missing
        self.requests: list[str] = []

    async def get_by_id_only(self, id: str) -> Document | None:
        self.requests.append(id)
        if self.missing:
            return None
        return self.row


class _FakeKBService:
    """Knowledge-base service double returning a canned knowledge base."""

    def __init__(self, kb: KnowledgeBaseInfo | None = None) -> None:
        self.kb = kb
        self.missing = False
        self.requests: list[str] = []

    async def get_knowledge_base_by_id_only(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        self.requests.append(knowledge_base_id)
        if self.missing or self.kb is None or self.kb.id != knowledge_base_id:
            raise NotFoundError(
                code="kb.not_found",
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return self.kb


class _FakeFileReader:
    """Image-bytes reader double for ``provider://`` reads."""

    def __init__(self, data: bytes | None = None, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.urls: list[str] = []

    async def get_file(self, *, url: str) -> bytes | None:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.data


class _FakeDownloader:
    """HTTP downloader double for ``http(s)://`` reads."""

    def __init__(self, data: bytes = b"", error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.urls: list[str] = []

    async def download(self, *, url: str) -> bytes:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.data


class _FakeEmbedder:
    """Embedding double satisfying the ``Embedder`` protocol."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def get_model_name(self) -> str:
        return "fake-embed"

    def get_model_id(self) -> str:
        return "embed-model"

    def get_dimensions(self) -> int:
        return self.dimensions

    async def embed(self, ctx: Context, text: str) -> list[float]:
        return [0.0] * self.dimensions

    async def batch_embed(self, ctx: Context, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return await self.batch_embed(ctx, texts)


class _FakeEngineService:
    """Retrieve-engine double recording every fan-out call."""

    def __init__(
        self,
        *,
        engine_type: RetrieverEngineType = RetrieverEngineType.SQLITE,
        support: list[RetrieverType] | None = None,
        batch_error: Exception | None = None,
    ) -> None:
        self._engine_type = engine_type
        self._support = support or [
            RetrieverType.KEYWORDS,
            RetrieverType.VECTOR,
        ]
        self.batch_error = batch_error
        self.indexed: list[IndexInfo] = []
        self.batch_calls: list[tuple[int, tuple[RetrieverType, ...]]] = []

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._support)

    async def retrieve(self, _ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        return []

    async def index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info: IndexInfo,
        _retriever_types: list[RetrieverType],
    ) -> None:
        self.indexed.append(index_info)

    async def batch_index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> None:
        if self.batch_error is not None:
            raise self.batch_error
        self.indexed.extend(index_info_list)
        self.batch_calls.append((len(index_info_list), tuple(retriever_types)))

    def estimate_storage_size(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        _retriever_types: list[RetrieverType],
    ) -> int:
        return len(index_info_list)

    async def delete_by_chunk_id_list(
        self,
        _ctx: Context,
        index_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass

    async def delete_by_source_id_list(
        self,
        _ctx: Context,
        source_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass

    async def delete_by_knowledge_id_list(
        self,
        _ctx: Context,
        _knowledge_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass

    async def batch_update_chunk_enabled_status(
        self,
        _ctx: Context,
        _chunk_status_map: Mapping[str, bool],
    ) -> None:
        pass

    async def batch_update_chunk_tag_id(
        self,
        _ctx: Context,
        _chunk_tag_map: Mapping[str, str],
    ) -> None:
        pass

    async def copy_indices(
        self,
        _ctx: Context,
        _source_knowledge_base_id: str,
        _source_to_target_kb_id_map: Mapping[str, str],
        _source_to_target_chunk_id_map: Mapping[str, str],
        _target_knowledge_base_id: str,
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass


class _FakeRegistry:
    """Registry double serving one engine service per engine type."""

    def __init__(self, service: _FakeEngineService) -> None:
        self._service = service

    def register(self, service: _FakeEngineService) -> None:
        pass

    def get_retrieve_engine_service(self, engine_type: RetrieverEngineType) -> _FakeEngineService:
        return self._service

    def register_with_store_id(self, store_id: str, svc: _FakeEngineService) -> None:
        pass

    def get_by_store_id(self, store_id: str) -> _FakeEngineService:
        return self._service

    def unregister_by_store_id(self, store_id: str) -> None:
        pass

    async def get_or_load_by_store_id(
        self, ctx: Context, tenant_id: int, store_id: str
    ) -> _FakeEngineService:
        return self._service


class _FakeEmbeddingResolver:
    """Embedding-resolver double returning a canned embedder."""

    def __init__(self, embedder: Embedder | None) -> None:
        self.embedder = embedder

    async def resolve_embedder(self, *, embedding_model_id: str) -> Embedder | None:
        return self.embedder


class _FakeIndexEngineResolver:
    """Index-engine resolver double returning a canned engine."""

    def __init__(self, engine: IndexEngine | None) -> None:
        self.engine = engine
        self.requests: list[tuple[int, str | None]] = []

    async def resolve_engine(
        self, *, tenant_id: int, vector_store_id: str | None
    ) -> IndexEngine | None:
        self.requests.append((tenant_id, vector_store_id))
        return self.engine


class _FakeFinalizer:
    """Multimodal-finalizer double recording every finalize call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str]] = []

    async def finalize(self, *, tenant_id: int, knowledge_id: str, knowledge_base_id: str) -> None:
        self.calls.append((tenant_id, knowledge_id, knowledge_base_id))


def _summary_engine(service: _FakeEngineService) -> CompositeRetrieveEngine:
    return new_composite_retrieve_engine(
        _FakeRegistry(service),
        [
            RetrieverEngineParams(
                retriever_engine_type=RetrieverEngineType.SQLITE,
                retriever_type=RetrieverType.KEYWORDS,
            ),
            RetrieverEngineParams(
                retriever_engine_type=RetrieverEngineType.SQLITE,
                retriever_type=RetrieverType.VECTOR,
            ),
        ],
    )


def _service(
    *,
    kb: KnowledgeBaseInfo,
    vlm: VLM | None = None,
    knowledge: Document | None = None,
    knowledge_missing: bool = False,
    chunk_service: _FakeChunkService | None = None,
    file_reader: ImageBytesReader | None = None,
    downloader: ImageUrlDownloader | None = None,
    embedder: Embedder | None = None,
    engine: IndexEngine | None = None,
    finalizer: MultimodalFinalizer | None = None,
) -> ImageMultimodalService:
    """Build a service wired with the given fakes."""
    if knowledge is None and not knowledge_missing:
        knowledge = _doc_row(
            tenant_id=kb.tenant_id,
            knowledge_base_id=kb.id,
            id="k-1",
        )
    return ImageMultimodalService(
        chunk_service=chunk_service or _FakeChunkService(),
        kb_service=_FakeKBService(kb),
        knowledge_repo=_FakeKnowledgeRepo(knowledge, missing=knowledge_missing),
        vlm_resolver=_FakeVLMResolver(vlm),
        file_reader=file_reader,
        url_downloader=downloader,
        embedding_resolver=_FakeEmbeddingResolver(embedder),
        index_engine_resolver=_FakeIndexEngineResolver(engine),
        finalizer=finalizer,
    )


def _run(
    service: ImageMultimodalService,
    payload: ImageMultimodalPayload | None = None,
    ctx: Context | None = None,
) -> ImageMultimodalOutcome:
    import asyncio

    return asyncio.run(service.process_image(ctx=ctx or _ctx(), payload=payload or _payload()))


# ── OCR sanitizer ─────────────────────────────────────────────────────


def test_sanitize_ocr_text_empty_and_whitespace() -> None:
    assert sanitize_ocr_text("") == ""
    assert sanitize_ocr_text("   \n\t  ") == ""


def test_sanitize_ocr_text_drops_html_skeleton() -> None:
    assert sanitize_ocr_text('<html><body><div class="image"><img/></div></body></html>') == ""
    assert sanitize_ocr_text("<html><body>  \n  </body></html>") == ""


def test_sanitize_ocr_text_passes_markdown_through() -> None:
    source = (
        "# 标题\n\n这是一段正文，包含一些内容。\n\n| 列1 | 列2 |\n| --- | --- |\n| 数据1 | 数据2 |"
    )
    assert sanitize_ocr_text(source) == source


def test_sanitize_ocr_text_strips_code_block_wrappers() -> None:
    assert (
        sanitize_ocr_text("```markdown\n# 文档标题\n\n正文内容在这里。\n```")
        == "# 文档标题\n\n正文内容在这里。"
    )


def test_sanitize_ocr_text_converts_html_code_block() -> None:
    assert sanitize_ocr_text("```html\n<p>这是一段内容</p>\n```") == "这是一段内容"


def test_sanitize_ocr_text_converts_html_document() -> None:
    source = (
        "<html><body><h1>标题</h1>"
        "<p>这是一段很长的正文内容，用来测试 HTML 到 Markdown 的转换。</p></body></html>"
    )
    assert (
        sanitize_ocr_text(source)
        == "# 标题\n\n这是一段很长的正文内容，用来测试 HTML 到 Markdown 的转换。"
    )


def test_sanitize_ocr_text_converts_table_html() -> None:
    source = (
        "<div><h2>报告摘要</h2>"
        "<p>本季度营收同比增长 15%，净利润达到 2.3 亿元。</p>"
        "<table><tr><th>指标</th><th>数值</th></tr>"
        "<tr><td>营收</td><td>10亿</td></tr></table></div>"
    )
    result = sanitize_ocr_text(source)
    assert result != ""
    assert result != source
    assert "## 报告摘要" in result
    assert "| 指标 | 数值 |" in result
    assert "| 营收 | 10亿 |" in result


def test_sanitize_ocr_text_known_empty_replies() -> None:
    assert sanitize_ocr_text("无文字内容") == ""
    assert sanitize_ocr_text("No text") == ""
    assert sanitize_ocr_text("图片中没有文字") == ""
    assert sanitize_ocr_text("No text content.") == ""


def test_sanitize_ocr_text_leaves_plain_text_with_minor_html() -> None:
    source = "这是一段正常文本，价格 <100 元。"
    assert sanitize_ocr_text(source) == source


def test_sanitize_ocr_text_collapses_blank_lines() -> None:
    assert sanitize_ocr_text("段落一\n\n\n\n\n段落二") == "段落一\n\n段落二"


def test_strip_markdown_code_block() -> None:
    assert strip_markdown_code_block("just normal text") == "just normal text"
    assert (
        strip_markdown_code_block("```markdown\n# Title\nContent here\n```")
        == "# Title\nContent here"
    )
    assert strip_markdown_code_block("```html\n<p>hello</p>\n```") == "<p>hello</p>"
    assert strip_markdown_code_block("```\nsome text\n```") == "some text"


def test_looks_like_html() -> None:
    assert looks_like_html("<html><body><p>text</p></body></html>") is True
    assert looks_like_html("<!DOCTYPE html><html><body></body></html>") is True
    assert looks_like_html("<body><p>content</p></body>") is True
    assert looks_like_html("# Title\n\nSome paragraph text") is False
    assert looks_like_html("This is mostly text with a <b>bold</b> word.") is False
    assert (
        looks_like_html("<div><p><span>x</span></p></div><div><p><span>y</span></p></div>") is True
    )


def test_is_known_empty_reply() -> None:
    assert is_known_empty_reply("无文字内容") is True
    assert is_known_empty_reply("无法识别") is True
    assert is_known_empty_reply("no text") is True
    assert is_known_empty_reply("No Text") is True
    assert is_known_empty_reply("NO CONTENT") is True
    assert is_known_empty_reply("empty") is True
    assert is_known_empty_reply("这是正常内容") is False
    assert is_known_empty_reply("") is False


def test_ocr_html_to_markdown_handles_plain_and_does_not_raise() -> None:
    assert ocr_html_to_markdown("<p>hello</p>").strip() == "hello"
    # Unbalanced markup must not raise; text still survives conversion.
    assert "bold" in ocr_html_to_markdown("<b>bold")


# ── Prompt building ───────────────────────────────────────────────────


def test_build_ocr_prompt_default_vs_scanned_pdf() -> None:
    assert build_ocr_prompt("", "") == VLM_OCR_PROMPT
    assert build_ocr_prompt(IMAGE_SOURCE_SCANNED_PDF, "") == VLM_OCR_SCANNED_PDF_PROMPT


def test_build_ocr_prompt_appends_instructions() -> None:
    prompt = build_ocr_prompt("", "focus on tables")
    assert prompt.startswith(VLM_OCR_PROMPT)
    assert "<image_ocr_business_instructions>" in prompt
    assert "focus on tables" in prompt


def test_build_vlm_caption_prompt_uses_language_and_instructions() -> None:
    prompt = build_vlm_caption_prompt("English", "Focus on alarm codes.")
    assert "in English" in prompt
    assert "Focus on alarm codes." in prompt
    assert "<image_description_business_instructions>" in prompt


def test_build_vlm_caption_prompt_without_instructions() -> None:
    prompt = build_vlm_caption_prompt("Chinese (Simplified)")
    assert "in Chinese (Simplified)" in prompt
    assert "<image_description_business_instructions>" not in prompt


def test_resolve_caption_language_precedence() -> None:
    config = VLMConfig(description_language="English")
    assert resolve_caption_language(config, "ko-KR") == "English"
    assert resolve_caption_language(VLMConfig(), "ko-KR") == "Korean"
    assert resolve_caption_language(VLMConfig(), "") == "Chinese (Simplified)"


# ── VLM config resolution ─────────────────────────────────────────────


def test_vlm_config_from_json_and_is_enabled() -> None:
    config = vlm_config_from_json(
        {
            "enabled": True,
            "model_id": "vlm-1",
            "description_language": "English",
            "custom_instructions": "focus",
        }
    )
    assert config.is_enabled() is True
    assert config.description_language == "English"
    assert config.custom_instructions == "focus"


def test_vlm_config_legacy_is_enabled() -> None:
    assert VLMConfig(model_name="qwen", base_url="http://localhost:11434").is_enabled() is True
    assert VLMConfig(enabled=True).is_enabled() is False
    assert VLMConfig(model_name="qwen").is_enabled() is False
    assert VLMConfig().is_enabled() is False


def test_resolve_vlm_config_uses_kb_default() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    config = resolve_vlm_config(kb, None)
    assert config.model_id == "vlm-1"
    assert config.is_enabled() is True


def test_resolve_vlm_config_override_falls_back_on_blank_fields() -> None:
    kb = _kb(
        vlm_config={
            "enabled": True,
            "model_id": "vlm-1",
            "description_language": "English",
            "custom_instructions": "base guidance",
        }
    )
    row = _doc_row(
        tenant_id=1,
        knowledge_base_id="kb-1",
        metadata={"process_overrides": {"vlm_config": {"enabled": True, "model_id": "vlm-2"}}},
    )
    config = resolve_vlm_config(kb, row)
    assert config.model_id == "vlm-2"
    assert config.description_language == "English"
    assert config.custom_instructions == "base guidance"
    assert config.is_enabled() is True


def test_resolve_vlm_config_override_replaces_fields() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    row = _doc_row(
        tenant_id=1,
        knowledge_base_id="kb-1",
        metadata={
            "process_overrides": {
                "vlm_config": {"enabled": True, "model_id": "vlm-2", "model_name": "inline"}
            }
        },
    )
    config = resolve_vlm_config(kb, row)
    assert config.model_id == "vlm-2"
    assert config.model_name == "inline"


# ── Image reference parsing ───────────────────────────────────────────


def test_parse_provider_scheme() -> None:
    assert parse_provider_scheme("local://images/img.png") == "local"
    assert parse_provider_scheme("minio://bucket/img.png") == "minio"
    assert parse_provider_scheme("https://example.com/img.png") == ""
    assert parse_provider_scheme("storage://b-1/minio://bucket/img.png") == "minio"


def test_parse_storage_backend_path() -> None:
    backend_id, path, ok = parse_storage_backend_path("storage://b-1/local://x/y.png")
    assert ok is True
    assert backend_id == "b-1"
    assert path == "local://x/y.png"
    assert parse_storage_backend_path("local://x/y.png") == ("", "", False)


def test_parse_resource_path() -> None:
    handle = "a" * 22
    assert parse_resource_path(f"resource://{handle}") == (handle, True)
    assert is_resource_reference(f"resource://{handle}") is True
    assert parse_resource_path("resource://too-short") == ("", False)
    assert parse_resource_path("local://x.png") == ("", False)


# ── Child chunk construction ──────────────────────────────────────────


def test_build_multimodal_chunks_creates_ocr_and_caption() -> None:
    from src.core.knowledge.documents.image_update import ImageInfo

    image_info = ImageInfo(
        url="local://images/img.png",
        original_url="local://images/img.png",
        caption="a chart",
        ocr_text="row data",
    )
    chunks = build_multimodal_chunks(
        payload=_payload(),
        image_info=image_info,
        now=_NOW,
    )
    assert [c.chunk_type for c in chunks] == [CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION]
    assert all(c.parent_chunk_id == "c-1" for c in chunks)
    assert all(c.is_enabled for c in chunks)
    stored = json.loads(chunks[0].image_info or "[]")
    assert stored[0]["url"] == "local://images/img.png"
    assert stored[0]["ocr_text"] == "row data"
    assert stored[0]["caption"] == "a chart"


def test_build_multimodal_chunks_skips_empty_sections() -> None:
    from src.core.knowledge.documents.image_update import ImageInfo

    only_caption = ImageInfo(url="u", original_url="u", caption="desc")
    assert [
        c.chunk_type
        for c in build_multimodal_chunks(payload=_payload(), image_info=only_caption, now=_NOW)
    ] == [CHUNK_TYPE_IMAGE_CAPTION]
    empty = ImageInfo(url="u", original_url="u")
    assert build_multimodal_chunks(payload=_payload(), image_info=empty, now=_NOW) == []


# ── Service: orphan drop ──────────────────────────────────────────────


def test_should_drop_orphaned_missing_knowledge() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(kb=kb, knowledge_missing=True)
    assert _run(svc).skipped == "orphaned"


def test_should_drop_orphaned_cancelled_or_deleting_knowledge() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    cancelled = _doc_row(tenant_id=1, knowledge_base_id="kb-1", parse_status=PARSE_STATUS_CANCELLED)
    deleting = _doc_row(tenant_id=1, knowledge_base_id="kb-1", parse_status=PARSE_STATUS_DELETING)
    assert _run(_service(kb=kb, knowledge=cancelled)).skipped == "orphaned"
    assert _run(_service(kb=kb, knowledge=deleting)).skipped == "orphaned"


def test_should_drop_orphaned_missing_knowledge_base() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(kb=kb, knowledge=_doc_row(tenant_id=1, knowledge_base_id="kb-1"))
    svc._kb_service.missing = True
    assert _run(svc).skipped == "orphaned"


def test_orphan_drop_still_finalizes() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    finalizer = _FakeFinalizer()
    svc = _service(kb=kb, knowledge_missing=True, finalizer=finalizer)
    outcome = _run(svc)
    assert outcome.skipped == "orphaned"
    assert finalizer.calls == [(1, "k-1", "kb-1")]


# ── Service: happy path ───────────────────────────────────────────────


def test_process_image_creates_ocr_and_caption_chunks() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    chunk_service = _FakeChunkService()
    finalizer = _FakeFinalizer()
    svc = _service(
        kb=kb,
        knowledge=_doc_row(tenant_id=1, knowledge_base_id="kb-1"),
        vlm=_FakeVLM(ocr_text="# Title\n\ndetails", caption="a chart"),
        chunk_service=chunk_service,
        file_reader=_FakeFileReader(b"image-bytes"),
        finalizer=finalizer,
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.ocr_text == "# Title\n\ndetails"
    assert outcome.caption == "a chart"
    assert outcome.image_bytes == len(b"image-bytes")
    assert outcome.chunks_created == 2
    assert outcome.skipped == ""
    assert len(chunk_service.created) == 2
    types = {c.chunk_type for c in chunk_service.created}
    assert types == {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}
    assert all(c.parent_chunk_id == "c-1" for c in chunk_service.created)
    assert finalizer.calls == [(1, "k-1", "kb-1")]


def test_process_image_scanned_pdf_uses_scanned_pdf_prompt() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    vlm = _FakeVLM(ocr_text="page text", caption="c")
    svc = _service(
        kb=kb,
        vlm=vlm,
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(
        svc,
        _payload(image_source_type=IMAGE_SOURCE_SCANNED_PDF, enable_ocr=True),
    )
    assert outcome.chunks_created == 2
    assert vlm.prompts[0] == VLM_OCR_SCANNED_PDF_PROMPT
    assert "scanned PDF" in vlm.prompts[0]


def test_process_image_skips_ocr_when_disabled() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    vlm = _FakeVLM(ocr_text="should be ignored", caption="a chart")
    svc = _service(kb=kb, vlm=vlm, file_reader=_FakeFileReader(b"bytes"))
    outcome = _run(svc, _payload(enable_ocr=False))
    assert outcome.caption == "a chart"
    assert outcome.ocr_text == ""
    assert outcome.chunks_created == 1
    assert all(prompt.startswith("Provide a brief") for prompt in vlm.prompts)


def test_process_image_ocr_empty_discards_ocr_chunk() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_text="No text content.", caption="a chart"),
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.ocr_text == ""
    assert outcome.ocr_skipped == "empty_or_invalid"
    assert outcome.caption == "a chart"
    assert outcome.chunks_created == 1


def test_process_image_caption_empty_keeps_ocr_chunk() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_text="body text", caption=""),
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.ocr_text == "body text"
    assert outcome.caption == ""
    assert outcome.chunks_created == 1


def test_process_image_no_content_skips() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_text="", caption=""),
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.chunks_created == 0
    assert outcome.skipped == "no_extracted_content"


def test_process_image_unreadable_image_skips() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_text="x", caption="y"),
        file_reader=_FakeFileReader(data=None),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.skipped == "unreadable_image"
    assert "not found" in outcome.read_error


def test_process_image_downloads_http_url() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    downloader = _FakeDownloader(b"from-http")
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(caption="c"),
        downloader=downloader,
    )
    outcome = _run(svc, _payload(image_url="https://example.com/img.png"))
    assert downloader.urls == ["https://example.com/img.png"]
    assert outcome.image_bytes == len(b"from-http")
    assert outcome.chunks_created == 1


def test_process_image_local_path_fallback() -> None:
    import tempfile

    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        handle.write(b"local-bytes")
        handle.flush()
        svc = _service(
            kb=kb,
            vlm=_FakeVLM(caption="c"),
            downloader=_FakeDownloader(b"fallback"),
        )
        outcome = _run(
            svc,
            _payload(
                image_url="https://example.com/img.png",
                image_local_path=handle.name,
            ),
        )
        assert outcome.image_bytes == len(b"local-bytes")


# ── Service: error paths ──────────────────────────────────────────────


def test_process_image_vlm_disabled_raises() -> None:
    kb = _kb(vlm_config=None)
    svc = _service(kb=kb, vlm=_FakeVLM(caption="c"))
    with pytest.raises(ValidationError) as excinfo:
        _run(svc)
    assert excinfo.value.code == "image_multimodal.vlm_disabled"


def test_process_image_vlm_unavailable_raises() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(kb=kb, vlm=None)
    with pytest.raises(ValidationError) as excinfo:
        _run(svc)
    assert excinfo.value.code == "image_multimodal.vlm_unavailable"


def test_process_image_ocr_failure_records_error_and_keeps_caption() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_error=RuntimeError("vlm down"), caption="a chart"),
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.ocr_error == "vlm down"
    assert outcome.caption == "a chart"
    assert outcome.chunks_created == 1


def test_process_image_caption_failure_records_error_and_keeps_ocr() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(ocr_text="body", caption_error=RuntimeError("vlm down")),
        file_reader=_FakeFileReader(b"bytes"),
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.caption_error == "vlm down"
    assert outcome.ocr_text == "body"
    assert outcome.chunks_created == 1


# ── Service: indexing ─────────────────────────────────────────────────


def test_process_image_indexes_via_composite() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    engine_service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(engine_service)
    chunk_service = _FakeChunkService()
    svc = _service(
        kb=kb,
        knowledge=_doc_row(tenant_id=1, knowledge_base_id="kb-1"),
        vlm=_FakeVLM(ocr_text="ocr", caption="caption"),
        chunk_service=chunk_service,
        file_reader=_FakeFileReader(b"bytes"),
        embedder=_FakeEmbedder(),
        engine=engine,
    )
    outcome = _run(svc, _payload(enable_ocr=True))
    assert outcome.chunks_created == 2
    assert outcome.indexed is True
    assert len(engine_service.indexed) == 2
    assert {info.source_id for info in engine_service.indexed} == {
        c.id for c in chunk_service.created
    }
    assert {c.status for c in chunk_service.rows.values()} == {CHUNK_STATUS_INDEXED}


def test_process_image_indexing_disabled_marks_indexed() -> None:
    kb = _kb(
        vlm_config={"enabled": True, "model_id": "vlm-1"},
        indexing_strategy={"vector_enabled": False, "keyword_enabled": False},
    )
    chunk_service = _FakeChunkService()
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(caption="caption"),
        chunk_service=chunk_service,
        file_reader=_FakeFileReader(b"bytes"),
        embedder=_FakeEmbedder(),
        engine=_summary_engine(_FakeEngineService(engine_type=RetrieverEngineType.SQLITE)),
    )
    outcome = _run(svc)
    assert outcome.chunks_created == 1
    assert outcome.indexed is True
    assert {c.status for c in chunk_service.rows.values()} == {CHUNK_STATUS_INDEXED}


def test_process_image_index_failure_keeps_chunks_unindexed() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    engine_service = _FakeEngineService(
        engine_type=RetrieverEngineType.SQLITE,
        batch_error=RuntimeError("index down"),
    )
    chunk_service = _FakeChunkService()
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(caption="caption"),
        chunk_service=chunk_service,
        file_reader=_FakeFileReader(b"bytes"),
        embedder=_FakeEmbedder(),
        engine=_summary_engine(engine_service),
    )
    outcome = _run(svc)
    assert outcome.chunks_created == 1
    assert outcome.indexed is False
    assert len(engine_service.indexed) == 0
    assert {c.status for c in chunk_service.rows.values()} == {CHUNK_STATUS_DEFAULT}


def test_process_image_index_skipped_when_embedder_unavailable() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    chunk_service = _FakeChunkService()
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(caption="caption"),
        chunk_service=chunk_service,
        file_reader=_FakeFileReader(b"bytes"),
        embedder=None,
        engine=_summary_engine(_FakeEngineService(engine_type=RetrieverEngineType.SQLITE)),
    )
    outcome = _run(svc)
    assert outcome.indexed is False
    assert {c.status for c in chunk_service.rows.values()} == {CHUNK_STATUS_DEFAULT}


def test_finalize_runs_on_inner_failure() -> None:
    kb = _kb(vlm_config={"enabled": True, "model_id": "vlm-1"})
    finalizer = _FakeFinalizer()
    svc = _service(
        kb=kb,
        vlm=_FakeVLM(caption="c"),
        file_reader=_FakeFileReader(data=None),
        finalizer=finalizer,
    )
    outcome = _run(svc)
    assert outcome.skipped == "unreadable_image"
    assert finalizer.calls == [(1, "k-1", "kb-1")]


# ── Integration against the real schema ────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema; skips without a DB."""
    reset_settings_cache()
    engine = DatabaseEngine(url=get_settings().database_url, poolclass=NullPool)
    try:
        await engine.prewarm()
    except Exception as exc:
        await engine.close()
        pytest.skip(f"integration database unavailable: {exc}")
    async with engine.session_factory() as s:
        yield s
        await s.rollback()
    await engine.close()


async def test_integration_image_multimodal_round_trip(db_session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="mm-docs",
        kb_type="document",
        vlm_config={"enabled": True, "model_id": "vlm-1", "description_language": "English"},
        embedding_model_id="embed-1",
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(db_session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        type="file",
        title="chart.png",
        source="file.png",
        file_name="chart.png",
        file_type="png",
        parse_status=PARSE_STATUS_PROCESSING,
    )
    chunk_repo = ChunkRepository(db_session)
    parent = _parent_chunk(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
    )
    stored_parent = (await chunk_repo.create_many([parent]))[0]

    engine_service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(engine_service)
    svc = ImageMultimodalService(
        chunk_service=ChunkService(chunk_repo=ChunkRepository(db_session)),
        kb_service=kb_service,
        knowledge_repo=KnowledgeRepository(db_session),
        vlm_resolver=_FakeVLMResolver(
            _FakeVLM(ocr_text="# 图表\n\n营收数据", caption="A quarterly revenue chart.")
        ),
        file_reader=_FakeFileReader(b"image-bytes"),
        embedding_resolver=_FakeEmbeddingResolver(_FakeEmbedder()),
        index_engine_resolver=_FakeIndexEngineResolver(engine),
        finalizer=_FakeFinalizer(),
    )
    outcome = await svc.process_image(
        ctx=_ctx(),
        payload=ImageMultimodalPayload(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            knowledge_base_id=kb.id,
            chunk_id=stored_parent.id,
            image_url="local://images/chart.png",
            enable_ocr=True,
            enable_caption=True,
        ),
    )
    assert outcome.chunks_created == 2
    assert outcome.indexed is True
    assert len(engine_service.indexed) == 2

    children = await chunk_repo.list_by_parent_id(tenant_id, stored_parent.id)
    assert len(children) == 2
    types = {c.chunk_type for c in children}
    assert types == {CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}
    assert all(c.status == CHUNK_STATUS_INDEXED for c in children)
    stored_images = json.loads(children[0].image_info or "[]")
    assert stored_images[0]["url"] == "local://images/chart.png"
    assert stored_images[0]["ocr_text"] == "# 图表\n\n营收数据"
    assert stored_images[0]["caption"] == "A quarterly revenue chart."


async def test_integration_orphan_drop(db_session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="mm-orphan",
        kb_type="document",
        vlm_config={"enabled": True, "model_id": "vlm-1"},
    )
    svc = ImageMultimodalService(
        chunk_service=ChunkService(chunk_repo=ChunkRepository(db_session)),
        kb_service=kb_service,
        knowledge_repo=KnowledgeRepository(db_session),
        vlm_resolver=_FakeVLMResolver(_FakeVLM(caption="c")),
        file_reader=_FakeFileReader(b"bytes"),
    )
    # The knowledge row does not exist, so the task is dropped without retry.
    outcome = await svc.process_image(
        ctx=_ctx(),
        payload=ImageMultimodalPayload(
            tenant_id=tenant_id,
            knowledge_id="missing-knowledge",
            knowledge_base_id=kb.id,
            chunk_id="c-missing",
            image_url="local://images/none.png",
        ),
    )
    assert outcome.skipped == "orphaned"
