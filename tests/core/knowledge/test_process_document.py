"""Unit and integration tests for the document processing pipeline.

Unit tests drive the pipeline and its stage modules against spec'd fake
repositories and seams with closure-captured state. The integration
section runs the full pipeline against the real applied schema (the
``documents`` and ``chunks`` tables) with mock docreader / embedder /
index seams and skips when the database is unreachable, so the unit
suite stays runnable without Postgres.

Tenant ids: ``documents.tenant_id`` is BIGINT while ``chunks.tenant_id``
is INTEGER. Tests that write chunks use an int32-safe tenant counter;
``make_test_tenant_id`` mints the document-only rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint
from typing import cast

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.types import IndexInfo
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject
from src.core.knowledge.documents.chunk_pipeline import (
    ParsedChunk,
    chunk_markdown,
    resolve_parent_child_configs,
    resolve_splitter_config,
)
from src.core.knowledge.documents.index_pipeline import (
    IndexEngine,
    build_index_infos,
    build_knowledge_index_content,
    chunk_embedding_content,
)
from src.core.knowledge.documents.parse_pipeline import (
    ParseResult,
    ReadRequest,
    parse_document,
)
from src.core.knowledge.documents.process_document import (
    DocumentProcessPipeline,
    PostProcessPayload,
    ProcessOutcome,
    TenantStorageInfo,
    build_chunk_rows,
    kb_needs_embedding,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_NONE,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER column; ids minted here stay inside
# the 32-bit range so the pipeline's chunk writes fit the schema.
_used_int32_tenant_ids: set[int] = set()


def _int32_tenant_id() -> int:
    """Return a tenant id that fits the ``chunks.tenant_id`` INTEGER column."""
    while True:
        candidate = randint(1, 2**31 - 1)
        if candidate not in _used_int32_tenant_ids:
            _used_int32_tenant_ids.add(candidate)
            return candidate


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _row(
    *,
    tenant_id: int | None = None,
    id: str | None = None,
    knowledge_base_id: str | None = None,
    parse_status: str = PARSE_STATUS_PENDING,
    enable_status: str = "disabled",
    type: str = "file",
    title: str = "fixture document",
    file_type: str = "pdf",
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or _kbid(),
            "type": type,
            "title": title,
            "description": None,
            "source": "fixture-source.pdf",
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": SUMMARY_STATUS_NONE,
            "enable_status": enable_status,
            "embedding_model_id": None,
            "file_name": "fixture.pdf",
            "file_type": file_type,
            "file_size": 1024,
            "file_hash": None,
            "file_path": "obj/fixture.pdf",
            "storage_size": 0,
            "metadata": None,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
        }
    )


def _kb(
    *,
    tenant_id: int,
    chunking_config: JsonObject | None = None,
    indexing_strategy: JsonObject | None = None,
    embedding_model_id: str = "embed-model",
    vector_store_id: str | None = None,
    asr_config: JsonObject | None = None,
) -> KnowledgeBaseInfo:
    """Build a knowledge-base projection for the mocked KB service."""
    return KnowledgeBaseInfo(
        id=_kbid(),
        name="test-kb",
        tenant_id=tenant_id,
        chunking_config=chunking_config,
        indexing_strategy=indexing_strategy,
        embedding_model_id=embedding_model_id,
        vector_store_id=vector_store_id,
        asr_config=asr_config,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── Fake seams ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _TaskContext:
    """Minimal embedding ``Context`` for the pipeline."""

    is_background_task: bool = True


class _FakeKnowledgeRepo:
    """In-memory ``KnowledgeRepository`` double with closure state."""

    def __init__(self, rows: list[Document] | None = None) -> None:
        self.rows: dict[str, Document] = {row.id: row for row in (rows or [])}

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        row = self.rows.get(id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return row

    async def update(self, row: Document) -> Document:
        self.rows[row.id] = row
        return row


class _FakeChunkRepo:
    """In-memory ``ChunkRepository`` double with closure state."""

    def __init__(self) -> None:
        self.rows: list[Chunk] = []
        self.deleted: list[str] = []
        self.create_error: Exception | None = None

    async def delete_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        now: datetime,
    ) -> int:
        self.deleted.append(knowledge_id)
        return 0

    async def create_many(self, rows: list[Chunk]) -> list[Chunk]:
        if self.create_error is not None:
            raise self.create_error
        self.rows.extend(rows)
        return rows


class _FakeKBService:
    """KB-service double returning a canned projection."""

    def __init__(self, kb: KnowledgeBaseInfo) -> None:
        self.kb = kb

    async def get_knowledge_base_by_id(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        return self.kb


class _FakeReader:
    """Document-reader double returning a canned parse result."""

    def __init__(
        self,
        result: ParseResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or ParseResult(markdown_content="")
        self.error = error
        self.requests: list[ReadRequest] = []

    async def read(self, request: ReadRequest) -> ParseResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeFileReader:
    """Storage seam double returning canned bytes for any stored object."""

    async def read_file(self, *, file_path: str) -> bytes:
        return b"stored-bytes"


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


class _FakeEngine:
    """Retrieval-index double satisfying the ``IndexEngine`` protocol."""

    def __init__(self, estimate: int = 1234) -> None:
        self.estimate = estimate
        self.indexed: list[IndexInfo] = []
        self.deleted: list[str] = []
        self.batch_error: Exception | None = None

    async def batch_index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
    ) -> None:
        if self.batch_error is not None:
            raise self.batch_error
        self.indexed.extend(index_info_list)

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        self.deleted.extend(knowledge_id_list)

    def estimate_storage_size(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
    ) -> int:
        return self.estimate


class _FakeEmbeddingResolver:
    """Embedding-resolver double returning a canned embedder."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder
        self.requests: list[str] = []

    async def resolve_embedder(self, *, embedding_model_id: str) -> Embedder | None:
        self.requests.append(embedding_model_id)
        return self.embedder


class _FakeIndexResolver:
    """Index-engine-resolver double returning a canned engine."""

    def __init__(self, engine: IndexEngine | None = None) -> None:
        self.engine = engine
        self.requests: list[tuple[int, str | None]] = []

    async def resolve_engine(
        self,
        *,
        tenant_id: int,
        vector_store_id: str | None,
    ) -> IndexEngine | None:
        self.requests.append((tenant_id, vector_store_id))
        return self.engine


class _FakeDispatcher:
    """Post-process dispatcher double recording payloads."""

    def __init__(self) -> None:
        self.payloads: list[PostProcessPayload] = []
        self.error: Exception | None = None

    async def dispatch(self, *, payload: PostProcessPayload) -> None:
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)


class _FakeStorage:
    """Tenant storage-accounting double for the quota gate."""

    def __init__(self, storage: TenantStorageInfo | None = None) -> None:
        self.storage = storage or TenantStorageInfo()
        self.adjusted: list[tuple[int, int]] = []

    async def get_storage(self, *, tenant_id: int) -> TenantStorageInfo:
        return self.storage

    async def adjust_storage_used(self, *, tenant_id: int, delta: int) -> None:
        self.adjusted.append((tenant_id, delta))


@dataclass
class _Harness:
    """Everything a pipeline test needs to act and assert."""

    pipeline: DocumentProcessPipeline
    knowledge_repo: _FakeKnowledgeRepo
    chunk_repo: _FakeChunkRepo
    reader: _FakeReader
    engine: _FakeEngine
    dispatcher: _FakeDispatcher
    storage: _FakeStorage


def _make_pipeline(
    *,
    row: Document,
    kb: KnowledgeBaseInfo,
    reader_result: ParseResult | None = None,
    reader_error: Exception | None = None,
    embedder: _FakeEmbedder | None = None,
    engine: _FakeEngine | None = None,
    with_dispatcher: bool = True,
    with_storage: bool = False,
    with_reader: bool = True,
) -> _Harness:
    """Assemble a pipeline over fake seams plus the backing fakes."""
    knowledge_repo = _FakeKnowledgeRepo(rows=[row])
    chunk_repo = _FakeChunkRepo()
    reader = _FakeReader(result=reader_result, error=reader_error)
    dispatcher = _FakeDispatcher() if with_dispatcher else None
    storage = _FakeStorage() if with_storage else None
    pipeline = DocumentProcessPipeline(
        knowledge_repo=cast(KnowledgeRepository, knowledge_repo),
        kb_service=cast(KBService, _FakeKBService(kb)),
        chunk_repo=cast(ChunkRepository, chunk_repo),
        reader=reader if with_reader else None,
        file_reader=_FakeFileReader(),
        embedding_resolver=_FakeEmbeddingResolver(embedder=embedder),
        index_engine_resolver=_FakeIndexResolver(engine=engine),
        post_process_dispatcher=dispatcher,
        storage_resolver=storage,
    )
    return _Harness(
        pipeline=pipeline,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        reader=reader,
        engine=engine or _FakeEngine(),
        dispatcher=dispatcher or _FakeDispatcher(),
        storage=storage or _FakeStorage(),
    )


async def _run(
    pipeline: DocumentProcessPipeline,
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    file_type: str = "pdf",
    url: str = "",
    language: str = "",
) -> ProcessOutcome:
    """Drive one pipeline run with canned inputs."""
    return await pipeline.run(
        ctx=_TaskContext(),
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        file_path="obj/fixture.pdf",
        file_name="fixture.pdf",
        file_type=file_type,
        url=url,
        language=language,
        now=_NOW,
    )


# ── parse_pipeline ────────────────────────────────────────────────────


class TestParseDocument:
    async def test_calls_reader_with_request(self) -> None:
        # Arrange
        reader = _FakeReader(result=ParseResult(markdown_content="# hello"))
        request = ReadRequest(file_content=b"data", file_name="a.md", file_type="md")

        # Act
        result = await parse_document(reader=reader, request=request)

        # Assert
        assert result.markdown_content == "# hello"
        assert reader.requests == [request]

    async def test_loads_file_bytes_through_file_reader(self) -> None:
        # Arrange
        class _FileReader:
            async def read_file(self, *, file_path: str) -> bytes:
                return b"stored-bytes"

        reader = _FakeReader(result=ParseResult(markdown_content="loaded"))
        request = ReadRequest(file_path="obj/a.pdf", file_name="a.pdf", file_type="pdf")

        # Act
        result = await parse_document(reader=reader, request=request, file_reader=_FileReader())

        # Assert
        assert result.markdown_content == "loaded"
        assert reader.requests[0].file_content == b"stored-bytes"

    async def test_url_request_skips_file_read(self) -> None:
        # Arrange
        reader = _FakeReader(result=ParseResult(markdown_content="web"))
        request = ReadRequest(url="https://example.com/page")

        # Act
        result = await parse_document(reader=reader, request=request)

        # Assert
        assert result.markdown_content == "web"
        assert reader.requests[0].file_content is None

    async def test_raises_without_any_content_source(self) -> None:
        # Arrange
        reader = _FakeReader()
        request = ReadRequest(file_name="a.md")

        # Act / Assert
        with pytest.raises(ExternalServiceError) as excinfo:
            await parse_document(reader=reader, request=request)
        assert excinfo.value.code == "document_parse.no_source"


# ── chunk_pipeline ────────────────────────────────────────────────────


class TestResolveSplitterConfig:
    def test_maps_configured_fields(self) -> None:
        # Act
        cfg = resolve_splitter_config(
            {
                "chunk_size": 100,
                "chunk_overlap": 10,
                "separators": ["\n\n"],
                "strategy": "heading",
                "token_limit": 500,
                "languages": ["en"],
            }
        )

        # Assert
        assert cfg.chunk_size == 100
        assert cfg.chunk_overlap == 10
        assert cfg.separators == ["\n\n"]
        assert cfg.strategy == "heading"
        assert cfg.token_limit == 500
        assert cfg.languages == ["en"]

    def test_empty_config_yields_zero_values(self) -> None:
        # Act
        cfg = resolve_splitter_config(None)

        # Assert
        assert cfg.chunk_size == 0
        assert cfg.chunk_overlap == 0
        assert cfg.separators == []
        assert cfg.strategy == ""
        assert cfg.token_limit == 0

    def test_non_scalar_values_are_dropped(self) -> None:
        # Act
        cfg = resolve_splitter_config({"chunk_size": "big", "separators": ["a", 1, None]})

        # Assert
        assert cfg.chunk_size == 0
        assert cfg.separators == ["a"]


class TestResolveParentChildConfigs:
    def test_defaults_fill_parent_and_child_sizes(self) -> None:
        # Act
        parent, child = resolve_parent_child_configs(None)

        # Assert
        assert parent.chunk_size == 4096
        assert child.chunk_size == 384
        assert child.chunk_overlap == 384 // 5

    def test_configured_sizes_win(self) -> None:
        # Act
        parent, child = resolve_parent_child_configs(
            {"parent_chunk_size": 2000, "child_chunk_size": 100, "chunk_overlap": 12}
        )

        # Assert
        assert parent.chunk_size == 2000
        assert parent.chunk_overlap == 12
        assert child.chunk_size == 100
        assert child.chunk_overlap == 20


class TestChunkMarkdown:
    def test_flat_mode_produces_positional_chunks(self) -> None:
        # Arrange
        text = "alpha beta gamma delta\n\n" * 20

        # Act
        result = chunk_markdown(text, {"chunk_size": 20, "chunk_overlap": 0})

        # Assert
        assert result.is_parent_child is False
        assert len(result.chunks) > 1
        assert result.parent_chunks == []
        assert result.chunks[0].start == 0
        assert result.chunks[0].end == len(result.chunks[0].content)

    def test_parent_child_mode_links_children_to_parents(self) -> None:
        # Arrange
        text = "alpha beta gamma delta\n\n" * 100

        # Act
        result = chunk_markdown(
            text,
            {
                "enable_parent_child": True,
                "parent_chunk_size": 100,
                "child_chunk_size": 30,
            },
        )

        # Assert
        assert result.is_parent_child is True
        assert len(result.parent_chunks) > 0
        assert len(result.chunks) > 0
        assert any(child.parent_index >= 0 for child in result.chunks)

    def test_empty_markdown_yields_no_chunks(self) -> None:
        # Act
        result = chunk_markdown("", None)

        # Assert
        assert result.chunks == []
        assert result.parent_chunks == []


# ── index_pipeline ────────────────────────────────────────────────────


class TestBuildKnowledgeIndexContent:
    def test_prepends_document_title(self) -> None:
        assert build_knowledge_index_content("Report", "body") == "Report\nbody"

    def test_blank_title_passes_content_through(self) -> None:
        assert build_knowledge_index_content("  ", "body") == "body"


class TestChunkEmbeddingContent:
    def test_prepends_context_header(self) -> None:
        chunk = _chunk(content=" body ", context_header="# Section")
        assert chunk_embedding_content(chunk) == "# Section\n\nbody"

    def test_without_header_trims_content(self) -> None:
        chunk = _chunk(content="  body  ", context_header="")
        assert chunk_embedding_content(chunk) == "body"


class TestBuildIndexInfos:
    def test_builds_one_entry_per_chunk(self) -> None:
        # Arrange
        chunks = [_chunk(content="one", id="c-1"), _chunk(content="two", id="c-2")]

        # Act
        infos = build_index_infos(
            chunks=chunks,
            knowledge_id="kid",
            knowledge_base_id="kbid",
            title="Doc",
        )

        # Assert
        assert [info.chunk_id for info in infos] == ["c-1", "c-2"]
        assert [info.source_id for info in infos] == ["c-1", "c-2"]
        assert [info.knowledge_id for info in infos] == ["kid", "kid"]
        assert [info.content for info in infos] == ["Doc\none", "Doc\ntwo"]
        assert all(info.is_enabled for info in infos)


def _chunk(*, content: str, id: str | None = None, context_header: str = "") -> Chunk:
    """Build a chunk row for index-content tests."""
    return Chunk(
        id=id or f"chunk-{uuid.uuid4().hex[:12]}",
        tenant_id=1,
        knowledge_base_id="kbid",
        knowledge_id="kid",
        content=content,
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        context_header=context_header,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── build_chunk_rows / kb_needs_embedding ─────────────────────────────


class TestBuildChunkRows:
    def test_flat_mode_links_text_rows_in_order(self) -> None:
        # Arrange
        chunks = [
            ParsedChunk(content="a", seq=0, start=0, end=1),
            ParsedChunk(content="b", seq=1, start=1, end=2),
            ParsedChunk(content="c", seq=2, start=2, end=3),
        ]

        # Act
        all_rows, text_rows = build_chunk_rows(
            tenant_id=1,
            knowledge_id="kid",
            knowledge_base_id="kbid",
            chunks=chunks,
            parent_chunks=[],
            now=_NOW,
        )

        # Assert
        assert len(all_rows) == 3
        assert all(row.chunk_type == "text" for row in all_rows)
        assert all_rows[0].next_chunk_id == all_rows[1].id
        assert all_rows[1].pre_chunk_id == all_rows[0].id
        assert all_rows[1].next_chunk_id == all_rows[2].id
        assert all_rows[2].pre_chunk_id == all_rows[1].id
        assert text_rows == all_rows

    def test_parent_child_mode_keeps_parent_rows_unindexed(self) -> None:
        # Arrange
        parents = [
            ParsedChunk(content="parent one", seq=0, start=0, end=10),
            ParsedChunk(content="parent two", seq=1, start=11, end=21),
        ]
        chunks = [
            ParsedChunk(content="child one", seq=0, start=0, end=9, parent_index=0),
            ParsedChunk(content="child two", seq=1, start=11, end=20, parent_index=1),
            ParsedChunk(content="standalone", seq=2, start=30, end=39, parent_index=-1),
        ]

        # Act
        all_rows, text_rows = build_chunk_rows(
            tenant_id=1,
            knowledge_id="kid",
            knowledge_base_id="kbid",
            chunks=chunks,
            parent_chunks=parents,
            now=_NOW,
        )

        # Assert
        parent_rows = [row for row in all_rows if row.chunk_type == "parent_text"]
        assert len(parent_rows) == 2
        assert parent_rows[0].next_chunk_id == parent_rows[1].id
        assert parent_rows[1].pre_chunk_id == parent_rows[0].id
        assert all(row.parent_chunk_id is None for row in parent_rows)

        child_rows = [row for row in all_rows if row.chunk_type == "text"]
        assert len(child_rows) == 3
        assert child_rows[0].parent_chunk_id == parent_rows[0].id
        assert child_rows[1].parent_chunk_id == parent_rows[1].id
        assert child_rows[2].parent_chunk_id is None
        # Only parented children join the index subset.
        assert text_rows == [child_rows[0], child_rows[1]]

    def test_blank_content_chunks_are_skipped(self) -> None:
        # Arrange
        chunks = [ParsedChunk(content="   ", seq=0)]

        # Act
        all_rows, text_rows = build_chunk_rows(
            tenant_id=1,
            knowledge_id="kid",
            knowledge_base_id="kbid",
            chunks=chunks,
            parent_chunks=[],
            now=_NOW,
        )

        # Assert
        assert all_rows == []
        assert text_rows == []


class TestKbNeedsEmbedding:
    def test_empty_strategy_defaults_to_enabled(self) -> None:
        assert kb_needs_embedding(None) is True
        assert kb_needs_embedding({}) is True

    def test_vector_only(self) -> None:
        assert kb_needs_embedding({"vector_enabled": True, "keyword_enabled": False}) is True

    def test_keyword_only(self) -> None:
        assert kb_needs_embedding({"vector_enabled": False, "keyword_enabled": True}) is True

    def test_all_disabled(self) -> None:
        assert kb_needs_embedding({"vector_enabled": False, "keyword_enabled": False}) is False


# ── Pipeline unit tests (mock seams) ──────────────────────────────────


class TestPipelineRun:
    async def test_success_indexes_chunks_and_finalizes(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20, "chunk_overlap": 0})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(estimate=4321),
        )

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_PROCESSING
        assert outcome.enable_status == "enabled"
        assert outcome.storage_size == 4321
        assert outcome.text_chunk_count > 0
        assert outcome.skipped is False
        assert len(harness.chunk_repo.rows) > 0
        assert len(harness.engine.indexed) == outcome.text_chunk_count
        stored = harness.knowledge_repo.rows[row.id]
        assert stored.parse_status == PARSE_STATUS_PROCESSING
        assert stored.enable_status == "enabled"
        assert stored.storage_size == 4321
        assert stored.processed_at == _NOW
        assert len(harness.dispatcher.payloads) == 1
        assert harness.dispatcher.payloads[0].knowledge_id == row.id

    async def test_document_without_text_chunks_completes_immediately(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=""),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(),
        )

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_COMPLETED
        assert outcome.text_chunk_count == 0
        assert outcome.skipped is False
        assert harness.knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_COMPLETED
        # No chunks to enrich -> no post-process fan-out.
        assert harness.dispatcher.payloads == []

    async def test_chunks_complete_when_post_process_unwired(self) -> None:
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20, "chunk_overlap": 0})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(),
            with_dispatcher=False,
        )

        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        assert outcome.parse_status == PARSE_STATUS_COMPLETED
        assert outcome.enable_status == "enabled"
        assert outcome.text_chunk_count > 0
        assert harness.knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_COMPLETED
        assert harness.dispatcher.payloads == []

    async def test_completed_document_is_skipped_idempotently(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_COMPLETED, enable_status="enabled")
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(row=row, kb=kb)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.skipped is True
        assert outcome.parse_status == PARSE_STATUS_COMPLETED
        assert harness.chunk_repo.rows == []
        assert harness.engine.indexed == []

    async def test_deleting_document_is_skipped(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_DELETING)
        harness = _make_pipeline(row=row, kb=_kb(tenant_id=tid))

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.skipped is True
        assert outcome.parse_status == PARSE_STATUS_DELETING

    async def test_cancelled_document_is_skipped(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_CANCELLED)
        harness = _make_pipeline(row=row, kb=_kb(tenant_id=tid))

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.skipped is True
        assert outcome.parse_status == PARSE_STATUS_CANCELLED

    async def test_missing_document_is_skipped(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid)
        knowledge_repo = _FakeKnowledgeRepo()
        chunk_repo = _FakeChunkRepo()
        pipeline = DocumentProcessPipeline(
            knowledge_repo=cast(KnowledgeRepository, knowledge_repo),
            kb_service=cast(KBService, _FakeKBService(kb)),
            chunk_repo=cast(ChunkRepository, chunk_repo),
            reader=_FakeReader(),
        )

        # Act
        outcome = await _run(pipeline, tenant_id=tid, knowledge_id=row.id, knowledge_base_id=kb.id)

        # Assert
        assert outcome.skipped is True
        assert chunk_repo.rows == []

    async def test_missing_reader_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(row=row, kb=kb, with_reader=False)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert outcome.error_message == "document parsing service is not configured"
        assert harness.knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_FAILED

    async def test_parse_error_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_error=ExternalServiceError(code="docreader.rpc_error", message="read failed"),
        )

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert outcome.error_message is not None
        assert "read failed" in outcome.error_message
        assert harness.knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_FAILED

    async def test_video_file_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, file_type="mp4")
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(row=row, kb=kb)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            file_type="mp4",
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert outcome.error_message == "video files are not supported"

    async def test_image_without_multimodal_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, file_type="png")
        kb = _kb(tenant_id=tid)
        harness = _make_pipeline(row=row, kb=kb)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            file_type="png",
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert "multimodal" in (outcome.error_message or "")

    async def test_audio_without_asr_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, file_type="mp3")
        kb = _kb(tenant_id=tid, asr_config=None)
        harness = _make_pipeline(row=row, kb=kb)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            file_type="mp3",
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert "ASR" in (outcome.error_message or "")

    async def test_chunk_write_error_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(),
        )
        harness.chunk_repo.create_error = RuntimeError("db down")

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert "create chunks failed" in (outcome.error_message or "")
        assert harness.engine.indexed == []

    async def test_batch_index_error_rolls_back_and_fails(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        engine = _FakeEngine()
        engine.batch_error = RuntimeError("index down")
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=engine,
        )

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert "batch index failed" in (outcome.error_message or "")
        # Rollback deleted the freshly written chunks and index rows on top
        # of the idempotent pre-parse cleanup (two delete passes per leg).
        assert harness.chunk_repo.deleted == [row.id, row.id]
        assert harness.engine.deleted == [row.id, row.id]
        assert harness.knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_FAILED

    async def test_embedding_disabled_still_persists_chunks(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(
            tenant_id=tid,
            chunking_config={"chunk_size": 20},
            indexing_strategy={"vector_enabled": False, "keyword_enabled": False},
        )
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(),
        )

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_PROCESSING
        assert len(harness.chunk_repo.rows) > 0
        assert harness.engine.indexed == []
        assert outcome.storage_size == 0

    async def test_storage_quota_exceeded_marks_document_failed(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(estimate=5000),
            with_storage=True,
        )
        harness.storage.storage = TenantStorageInfo(storage_quota=1000, storage_used=0)

        # Act
        outcome = await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        assert outcome.error_message == "storage quota exceeded"
        assert harness.engine.indexed == []

    async def test_success_adjusts_storage_usage(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(markdown_content=text),
            embedder=_FakeEmbedder(),
            engine=_FakeEngine(estimate=777),
            with_storage=True,
        )

        # Act
        await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
        )

        # Assert
        assert harness.storage.adjusted == [(tid, 777)]

    async def test_url_import_adopts_extracted_title(self) -> None:
        # Arrange
        tid = make_test_tenant_id()
        row = _row(tenant_id=tid, type="url", title="")
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        harness = _make_pipeline(
            row=row,
            kb=kb,
            reader_result=ParseResult(
                markdown_content=text,
                metadata={"title": "Extracted Page"},
            ),
        )

        # Act
        await _run(
            harness.pipeline,
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            url="https://example.com/page",
        )

        # Assert
        assert harness.knowledge_repo.rows[row.id].title == "Extracted Page"


# ── Integration against the real schema ───────────────────────────────


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


def _integration_pipeline(
    *,
    kb: KnowledgeBaseInfo,
    reader: _FakeReader,
    embedder: _FakeEmbedder | None = None,
    engine: _FakeEngine | None = None,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
) -> DocumentProcessPipeline:
    """Assemble a pipeline over real repositories + fake seams."""
    return DocumentProcessPipeline(
        knowledge_repo=knowledge_repo,
        kb_service=cast(KBService, _FakeKBService(kb)),
        chunk_repo=chunk_repo,
        reader=reader,
        file_reader=_FakeFileReader(),
        embedding_resolver=_FakeEmbeddingResolver(embedder=embedder),
        index_engine_resolver=_FakeIndexResolver(engine=engine),
    )


class TestPipelineIntegration:
    async def test_full_pipeline_persists_chunks_and_finalizes(
        self,
        db_session: AsyncSession,
    ) -> None:
        # Arrange
        tid = _int32_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_PENDING)
        knowledge_repo = KnowledgeRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        await knowledge_repo.create(row)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20, "chunk_overlap": 0})
        text = "alpha beta gamma delta\n\n" * 20
        engine = _FakeEngine(estimate=99)
        pipeline = _integration_pipeline(
            kb=kb,
            reader=_FakeReader(result=ParseResult(markdown_content=text)),
            embedder=_FakeEmbedder(),
            engine=engine,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )

        # Act
        outcome = await pipeline.run(
            ctx=_TaskContext(),
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            file_path="obj/fixture.pdf",
            file_name="fixture.pdf",
            file_type="pdf",
            now=_NOW,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_COMPLETED
        assert outcome.enable_status == "enabled"
        persisted = await knowledge_repo.get_by_id(tid, row.id)
        assert persisted is not None
        assert persisted.parse_status == PARSE_STATUS_COMPLETED
        assert persisted.enable_status == "enabled"
        assert persisted.storage_size == 99
        chunks = await chunk_repo.list_by_knowledge_id(tid, row.id)
        assert len(chunks) > 0
        assert all(chunk.tenant_id == tid for chunk in chunks)
        assert len(engine.indexed) == len(chunks)

    async def test_completed_document_is_skipped_without_writes(
        self,
        db_session: AsyncSession,
    ) -> None:
        # Arrange
        tid = _int32_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_COMPLETED)
        knowledge_repo = KnowledgeRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        await knowledge_repo.create(row)
        kb = _kb(tenant_id=tid)
        pipeline = _integration_pipeline(
            kb=kb,
            reader=_FakeReader(),
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )

        # Act
        outcome = await pipeline.run(
            ctx=_TaskContext(),
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            now=_NOW,
        )

        # Assert
        assert outcome.skipped is True
        assert await chunk_repo.count_by_knowledge_base_id(tid, row.knowledge_base_id) == 0

    async def test_index_failure_soft_deletes_written_chunks(
        self,
        db_session: AsyncSession,
    ) -> None:
        # Arrange
        tid = _int32_tenant_id()
        row = _row(tenant_id=tid, parse_status=PARSE_STATUS_PENDING)
        knowledge_repo = KnowledgeRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        await knowledge_repo.create(row)
        kb = _kb(tenant_id=tid, chunking_config={"chunk_size": 20})
        text = "alpha beta gamma delta\n\n" * 20
        engine = _FakeEngine()
        engine.batch_error = RuntimeError("index down")
        pipeline = _integration_pipeline(
            kb=kb,
            reader=_FakeReader(result=ParseResult(markdown_content=text)),
            embedder=_FakeEmbedder(),
            engine=engine,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )

        # Act
        outcome = await pipeline.run(
            ctx=_TaskContext(),
            tenant_id=tid,
            knowledge_id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            file_path="obj/fixture.pdf",
            file_name="fixture.pdf",
            file_type="pdf",
            now=_NOW,
        )

        # Assert
        assert outcome.parse_status == PARSE_STATUS_FAILED
        persisted = await knowledge_repo.get_by_id(tid, row.id)
        assert persisted is not None
        assert persisted.parse_status == PARSE_STATUS_FAILED
        assert persisted.error_message is not None
        assert "batch index failed" in persisted.error_message
        # Rollback soft-deleted the freshly written chunks.
        assert await chunk_repo.list_by_knowledge_id(tid, row.id) == []
        # Cleanup + rollback delete passes target the same knowledge id.
        assert engine.deleted == [row.id, row.id]
