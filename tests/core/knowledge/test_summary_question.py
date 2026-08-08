"""Unit + integration tests for ``process_summary`` / ``generate_questions``.

Unit tests drive the standalone modules with stateful repository mocks
(closure-captured storage, the same pattern used across the core service
tests) and a scripted chat seam: they cover validation, error
classification, content reconstruction, the stale guards, and the LLM
fallback.

Integration tests run against the real applied schema. ``chunks`` carries
an INTEGER (32-bit) ``tenant_id`` column, so those tests use an int32-safe
tenant id (a local counter) instead of ``make_test_tenant_id``'s BIGINT
range, which would overflow it.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.llm import ChatOptions, ChatResponse, Message
from src.common.exception import AIProviderError, ConflictError, NotFoundError, ValidationError
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_SUMMARY,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.question_gen import (
    append_custom_instructions,
    generate_questions,
    parse_questions,
    resolve_question_generation_config,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.summary import (
    SummaryResult,
    custom_metadata_text,
    language_name_for,
    process_summary,
    real_text_rune_count,
    sample_long_content,
    strip_image_markup,
)
from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    PARSE_STATUS_COMPLETED,
    SUMMARY_STATUS_COMPLETED,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_PROCESSING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids from
# this counter so they stay inside the range.
_INT32_TENANT_BASE = 3_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)

# Shared prompt template used across the question-generation unit tests.
_QUESTION_PROMPT = (
    "Generate {{question_count}} questions for '{{doc_name}}':\n"
    "{{content}}\n{{context}}"
)


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Sample rows ───────────────────────────────────────────────────────


def _sample_doc(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    parse_status: str = PARSE_STATUS_COMPLETED,
    title: str = "Q3 budget",
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks."""
    return Document.model_validate(
        {
            "id": id or f"doc-{uuid.uuid4().hex[:12]}",
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or f"kb-{uuid.uuid4().hex[:12]}",
            "type": "file",
            "title": title,
            "description": None,
            "source": "budget-2026.pdf",
            "channel": CHANNEL_WEB,
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": "budget-2026.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "file_hash": None,
            "file_path": None,
            "storage_size": 2048,
            "metadata": None,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
            **columns,
        }
    )


def _sample_chunk(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    knowledge_id: str | None = None,
    chunk_index: int = 0,
    chunk_type: str = CHUNK_TYPE_TEXT,
    content: str = "chunk text",
    start_at: int = 0,
    content_revision: int = 0,
    **columns: object,
) -> Chunk:
    """Build a persisted-shape chunk row for seeding mocks."""
    return Chunk.model_validate(
        {
            "id": id or f"ch-{uuid.uuid4().hex[:12]}",
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or f"kb-{uuid.uuid4().hex[:12]}",
            "knowledge_id": knowledge_id or f"kn-{uuid.uuid4().hex[:12]}",
            "content": content,
            "chunk_index": chunk_index,
            "is_enabled": True,
            "start_at": start_at,
            "end_at": start_at + len(content),
            "pre_chunk_id": None,
            "next_chunk_id": None,
            "chunk_type": chunk_type,
            "parent_chunk_id": None,
            "image_info": None,
            "relation_chunks": None,
            "indirect_relation_chunks": None,
            "metadata": None,
            "tag_id": None,
            "status": CHUNK_STATUS_STORED,
            "content_hash": None,
            "flags": CHUNK_FLAG_RECOMMENDED,
            "seq_id": 0,
            "source_content": "",
            "content_revision": content_revision,
            "index_status": "ready",
            "last_editor_id": "",
            "context_header": "",
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **columns,
        }
    )


def _sample_kb(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    kb_type: str = "document",
    summary_model_id: str = "model-sum",
    embedding_model_id: str = "",
    indexing_strategy: dict[str, object] | None = None,
    question_generation_config: dict[str, object] | None = None,
) -> KnowledgeBaseInfo:
    """Build a knowledge-base service shape for mocking ``KBService``."""
    return KnowledgeBaseInfo(
        id=id or f"kb-{uuid.uuid4().hex[:12]}",
        name="test-kb",
        type=kb_type,
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        summary_model_id=summary_model_id,
        embedding_model_id=embedding_model_id,
        indexing_strategy=indexing_strategy,
        question_generation_config=question_generation_config,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── Fake seams ────────────────────────────────────────────────────────


class _FakeChat:
    """Scripted chat seam: records calls and returns canned output."""

    def __init__(
        self,
        *,
        content: str = "",
        questions: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.questions = questions
        self.error = error
        self.calls: list[tuple[list[Message], ChatOptions | None]] = []

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        if self.error is not None:
            raise self.error
        self.calls.append((messages, opts))
        if self.questions is not None:
            return ChatResponse(content="\n".join(self.questions))
        return ChatResponse(content=self.content)


# ── Repository mocks (stateful via side_effect closures) ──────────────


def _make_knowledge_repo() -> tuple[AsyncMock, dict[str, Document]]:
    """Knowledge-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        row = rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def _update(row: Document) -> Document:
        rows[row.id] = row
        return row

    repo.get_by_id.side_effect = _get_by_id
    repo.update.side_effect = _update
    return repo, rows


def _make_chunk_repo() -> tuple[AsyncMock, dict[str, Chunk]]:
    """Chunk-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=ChunkRepository)
    rows: dict[str, Chunk] = {}

    async def _list_by_knowledge_id(tenant_id: int, knowledge_id: str) -> list[Chunk]:
        out = [
            chunk
            for chunk in rows.values()
            if chunk.tenant_id == tenant_id
            and chunk.knowledge_id == knowledge_id
            and chunk.deleted_at is None
            and chunk.chunk_type == CHUNK_TYPE_TEXT
        ]
        return sorted(out, key=lambda c: c.chunk_index)

    async def _find_all(cols: dict[str, object]) -> list[Chunk]:
        out: list[Chunk] = []
        for row in rows.values():
            if row.deleted_at is not None:
                continue
            if all(getattr(row, key) == value for key, value in cols.items()):
                out.append(row)
        return out

    async def _get_by_id(tenant_id: int, id: str) -> Chunk:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            raise NotFoundError(
                code="chunk.not_found",
                message=f"chunk {id} not found",
            )
        return row

    async def _update(row: Chunk) -> Chunk:
        rows[row.id] = row
        return row

    async def _create(row: Chunk) -> Chunk:
        rows[row.id] = row
        return row

    repo.list_by_knowledge_id.side_effect = _list_by_knowledge_id
    repo.find_all_by_column_values.side_effect = _find_all
    repo.get_by_id.side_effect = _get_by_id
    repo.update.side_effect = _update
    repo.create.side_effect = _create
    return repo, rows


def _make_kb_service(*kbs: KnowledgeBaseInfo) -> AsyncMock:
    """``KBService`` mock resolving the supplied knowledge bases by id."""
    svc = AsyncMock(spec=KBService)
    by_id = {kb.id: kb for kb in kbs}

    async def _get_by_id(*, knowledge_base_id: str) -> KnowledgeBaseInfo:
        kb = by_id.get(knowledge_base_id)
        if kb is None:
            raise NotFoundError(
                code="knowledge_base.not_found",
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return kb

    svc.get_knowledge_base_by_id.side_effect = _get_by_id
    return svc


# ── Helper unit tests ─────────────────────────────────────────────────


def test_strip_image_markup_keeps_ocr_text() -> None:
    content = "![img](a.png)\n<image_original>![img](a.png)</image_original>\n<image_ocr>OCR line</image_ocr>"
    stripped = strip_image_markup(content)
    assert "![img]" not in stripped
    assert "<image_original>" not in stripped
    assert "OCR line" in stripped
    assert real_text_rune_count(content) == len("OCR line")


def test_sample_long_content_keeps_short_content() -> None:
    assert sample_long_content("short", 100) == "short"


def test_sample_long_content_truncates_when_window_too_small() -> None:
    assert sample_long_content("abcdefgh", 4) == "abcd"


def test_custom_metadata_text_sorts_and_skips_blanks() -> None:
    assert custom_metadata_text(None) == ""
    assert custom_metadata_text({}) == ""
    text = custom_metadata_text({"b": "two", "a": "one", "skip": None, "blank": "   "})
    assert text == "a: one\nb: two"


def test_language_name_for_unknown_locale_returns_locale() -> None:
    assert language_name_for("zh-CN") == "Chinese (Simplified)"
    assert language_name_for("en-US") == "English"
    assert language_name_for("xx-XX") == "xx-XX"


def test_parse_questions_strips_markers_and_caps_count() -> None:
    raw = (
        "1. First question here\n"
        "- Second question there\n"
        "\n"
        "* Third question everywhere\n"
        "  4) Fourth question now"
    )
    assert parse_questions(raw, 10) == [
        "First question here",
        "Second question there",
        "Third question everywhere",
        "Fourth question now",
    ]
    assert parse_questions(raw, 2) == ["First question here", "Second question there"]


def test_parse_questions_skips_short_lines() -> None:
    assert parse_questions("ok\nno\nwhat about this longer line", 10) == [
        "what about this longer line"
    ]


def test_append_custom_instructions_empty_is_noop() -> None:
    assert append_custom_instructions("base", "   ") == "base"


def test_append_custom_instructions_wraps_guidance() -> None:
    out = append_custom_instructions("base prompt", "Focus on X.")
    assert "base prompt" in out
    assert "<question_generation_business_instructions>" in out
    assert "Focus on X." in out


def test_resolve_question_generation_config_defaults_to_three() -> None:
    kb = _sample_kb(id="kb-1", tenant_id=7)
    row = _sample_doc(id="kn-1", tenant_id=7, knowledge_base_id="kb-1")
    count, instructions = resolve_question_generation_config(kb, row)
    assert count == 3
    assert instructions == ""


def test_resolve_question_generation_config_applies_override_and_clamp() -> None:
    kb = _sample_kb(
        id="kb-1",
        tenant_id=7,
        question_generation_config={"question_count": 5, "custom_instructions": "Be strict."},
    )
    row = _sample_doc(id="kn-1", tenant_id=7, knowledge_base_id="kb-1")
    count, instructions = resolve_question_generation_config(kb, row)
    assert count == 5
    assert instructions == "Be strict."
    # an explicit count wins and is clamped to the hard cap
    count, _ = resolve_question_generation_config(kb, row, override_count=99)
    assert count == 10


# ── process_summary unit tests ────────────────────────────────────────


async def test_summary_generates_description_and_summary_chunk() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        knowledge_id=doc.id,
        content="The quick brown fox jumps over the lazy dog.",
    )
    fake = _FakeChat(content="A concise summary.")

    result = await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize this document.",
    )

    assert isinstance(result, SummaryResult)
    assert result.summary == "A concise summary."
    assert result.knowledge.description == "A concise summary."
    assert result.knowledge.summary_status == SUMMARY_STATUS_COMPLETED
    assert result.summary_chunk_id is not None
    stored = k_rows[doc.id]
    assert stored.summary_status == SUMMARY_STATUS_COMPLETED
    assert stored.description == "A concise summary."
    summary_chunks = [c for c in c_rows.values() if c.chunk_type == CHUNK_TYPE_SUMMARY]
    assert len(summary_chunks) == 1
    assert summary_chunks[0].content == "# Summary\nA concise summary."
    assert summary_chunks[0].parent_chunk_id == "c-1"
    assert len(fake.calls) == 1
    messages, opts = fake.calls[0]
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert opts is not None
    assert opts.temperature == 0.3


async def test_summary_folds_custom_metadata_into_prompt() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        custom_metadata={"author": "finance team"},
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, _c_rows = _make_chunk_repo()
    c_rows = _c_rows
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id
    )
    fake = _FakeChat(content="summary")

    await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize.",
    )

    messages, _ = fake.calls[0]
    user_text = messages[1].content
    assert "Document metadata:" in user_text
    assert "author: finance team" in user_text
    assert "Document content:" in user_text


async def test_summary_renders_language_placeholder() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, _c_rows = _make_chunk_repo()
    _c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id
    )
    fake = _FakeChat(content="summary")

    await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Answer in {{language}}",
        language="zh-CN",
    )

    messages, _ = fake.calls[0]
    assert messages[0].content == "Answer in Chinese (Simplified)"


async def test_summary_reconstructs_content_by_start_at() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=1, start_at=0, content="Alpha part ",
    )
    c_rows["c-2"] = _sample_chunk(
        id="c-2", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=0, start_at=11, content="beta part",
    )
    fake = _FakeChat(content="summary")

    await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize.",
    )

    messages, _ = fake.calls[0]
    assert "Alpha part beta part" in messages[1].content


async def test_summary_updates_existing_summary_chunk() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id
    )
    c_rows["c-sum"] = _sample_chunk(
        id="c-sum", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=1, chunk_type=CHUNK_TYPE_SUMMARY, content="# Summary\nold",
    )
    fake = _FakeChat(content="new summary")

    result = await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize.",
    )

    assert result.summary_chunk_id == "c-sum"
    assert c_rows["c-sum"].content == "# Summary\nnew summary"
    assert c_rows["c-sum"].source_content == "# Summary\nnew summary"
    assert c_rows["c-sum"].is_enabled is True
    assert len([c for c in c_rows.values() if c.chunk_type == CHUNK_TYPE_SUMMARY]) == 1


async def test_summary_skips_summary_chunk_when_no_embedding_model_needed() -> None:
    tenant_id = 7
    kb = _sample_kb(
        id="kb-1",
        tenant_id=tenant_id,
        summary_model_id="model-sum",
        indexing_strategy={
            "vector_enabled": False,
            "keyword_enabled": False,
            "wiki_enabled": False,
            "graph_enabled": False,
        },
    )
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id
    )
    fake = _FakeChat(content="summary")

    result = await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize.",
    )

    assert result.summary_chunk_id is None
    assert not any(c.chunk_type == CHUNK_TYPE_SUMMARY for c in c_rows.values())


async def test_summary_validates_scope() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await process_summary(
            tenant_id=0,
            knowledge_id="kn-1",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await process_summary(
            tenant_id=7,
            knowledge_id="  ",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.id_required"


async def test_summary_raises_not_found_for_missing_document() -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await process_summary(
            tenant_id=7,
            knowledge_id="kn-missing",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_summary_requires_configured_summary_model() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc

    with pytest.raises(ValidationError) as exc_info:
        await process_summary(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(),
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(kb),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.summary_model_not_configured"


async def test_summary_raises_when_no_enabled_text_chunks() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        is_enabled=False,
    )

    with pytest.raises(ValidationError) as exc_info:
        await process_summary(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(),
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            kb_service=_make_kb_service(kb),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.summary_no_text_chunks"


async def test_summary_insufficient_content_marks_failed_without_llm() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="![scanned](scan.png)",
    )
    fake = _FakeChat(content="summary")

    with pytest.raises(ValidationError) as exc_info:
        await process_summary(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=fake,
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            kb_service=_make_kb_service(kb),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.summary_insufficient_content"
    assert k_rows[doc.id].summary_status == SUMMARY_STATUS_FAILED
    assert k_rows[doc.id].description == ""
    assert fake.calls == []


async def test_summary_falls_back_to_first_chunk_on_llm_error() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="The quick brown fox jumps over the lazy dog.",
    )
    fake = _FakeChat(error=RuntimeError("boom"))

    result = await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt="Summarize.",
    )

    assert result.summary == "The quick brown fox jumps over the lazy dog."
    assert k_rows[doc.id].description == "The quick brown fox jumps over the lazy dog."
    assert k_rows[doc.id].summary_status == SUMMARY_STATUS_COMPLETED


async def test_summary_discards_when_metadata_changed() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        custom_metadata={"topic": "budget"},
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id
    )
    state = {"reads": 0}

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        state["reads"] += 1
        row = k_rows[id]
        if state["reads"] >= 2:
            return row.model_copy(update={"custom_metadata": {"topic": "changed"}})
        return row

    k_repo.get_by_id.side_effect = _get_by_id

    with pytest.raises(ConflictError) as exc_info:
        await process_summary(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(content="summary"),
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            kb_service=_make_kb_service(kb),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.summary_superseded"
    # summary_status is left untouched by the stale guard
    assert k_rows[doc.id].summary_status == SUMMARY_STATUS_PROCESSING


async def test_summary_discards_when_chunk_revision_changed() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="The quick brown fox jumps over the lazy dog.",
        content_revision=0,
    )

    async def _get_chunk(tenant_id: int, id: str) -> Chunk:
        row = c_rows[id]
        return row.model_copy(update={"content_revision": row.content_revision + 1})

    c_repo.get_by_id.side_effect = _get_chunk

    with pytest.raises(ConflictError) as exc_info:
        await process_summary(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(content="summary"),
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            kb_service=_make_kb_service(kb),
            prompt="Summarize.",
        )
    assert exc_info.value.code == "knowledge.summary_superseded"


# ── generate_questions unit tests ─────────────────────────────────────


async def test_questions_generate_and_bind_metadata() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1", title="Budget")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=0, start_at=0, content="Alpha paragraph about finance.",
    )
    c_rows["c-2"] = _sample_chunk(
        id="c-2", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=1, start_at=40, content="Beta paragraph about budgeting.",
    )
    fake = _FakeChat(questions=["What drives cost growth?", "How is the budget split?"])

    result = await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
    )

    assert len(result) == 4
    meta = c_rows["c-1"].metadata
    assert meta is not None
    assert [q["question"] for q in meta["generated_questions"]] == [
        "What drives cost growth?",
        "How is the budget split?",
    ]
    assert all(q["content_revision"] == 0 for q in meta["generated_questions"])
    assert len(fake.calls) == 2
    # the second chunk is asked with the first chunk as preceding context
    messages, _ = fake.calls[1]
    user_text = messages[0].content
    assert "Generate 3 questions" in user_text
    assert "<surrounding_context>" in user_text
    assert "<preceding_content>" in user_text


async def test_questions_applies_kb_config_and_custom_instructions() -> None:
    tenant_id = 7
    kb = _sample_kb(
        id="kb-1",
        tenant_id=tenant_id,
        summary_model_id="model-sum",
        question_generation_config={
            "question_count": 5,
            "custom_instructions": "Focus on financial terms.",
        },
    )
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="Alpha paragraph about finance.",
    )
    fake = _FakeChat(questions=["One?", "Two?"])

    await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
    )

    messages, _ = fake.calls[0]
    user_text = messages[0].content
    assert "Generate 5 questions" in user_text
    assert "<question_generation_business_instructions>" in user_text
    assert "Focus on financial terms." in user_text


async def test_questions_explicit_count_wins_and_clamps() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="Alpha paragraph about finance.",
    )
    fake = _FakeChat(questions=["One?", "Two?"])

    await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
        question_count=8,
    )
    assert "Generate 8 questions" in fake.calls[0][0][0].content

    await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
        question_count=99,
    )
    assert "Generate 10 questions" in fake.calls[1][0][0].content


async def test_questions_validates_scope_and_prompt() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await generate_questions(
            tenant_id=0,
            knowledge_id="kn-1",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt=_QUESTION_PROMPT,
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await generate_questions(
            tenant_id=7,
            knowledge_id="",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt=_QUESTION_PROMPT,
        )
    assert exc_info.value.code == "knowledge.id_required"

    with pytest.raises(ValidationError) as exc_info:
        await generate_questions(
            tenant_id=7,
            knowledge_id="kn-1",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt="   ",
        )
    assert exc_info.value.code == "knowledge.questions_prompt_not_configured"


async def test_questions_raises_not_found_for_missing_document() -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await generate_questions(
            tenant_id=7,
            knowledge_id="kn-missing",
            chat=_FakeChat(),
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(),
            prompt=_QUESTION_PROMPT,
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_questions_requires_configured_summary_model() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc

    with pytest.raises(ValidationError) as exc_info:
        await generate_questions(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(),
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            kb_service=_make_kb_service(kb),
            prompt=_QUESTION_PROMPT,
        )
    assert exc_info.value.code == "knowledge.summary_model_not_configured"


async def test_questions_raises_ai_provider_error_on_llm_failure() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="Alpha paragraph about finance.",
    )

    with pytest.raises(AIProviderError) as exc_info:
        await generate_questions(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chat=_FakeChat(error=RuntimeError("boom")),
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            kb_service=_make_kb_service(kb),
            prompt=_QUESTION_PROMPT,
        )
    assert exc_info.value.code == "knowledge.question_generation_failed"


async def test_questions_skips_stale_chunk_revision() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="Alpha paragraph about finance.",
        content_revision=0,
    )

    async def _get_chunk(tenant_id: int, id: str) -> Chunk:
        row = c_rows[id]
        return row.model_copy(update={"content_revision": row.content_revision + 1})

    c_repo.get_by_id.side_effect = _get_chunk

    result = await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=_FakeChat(questions=["One?", "Two?"]),
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
    )

    assert result == []
    assert c_rows["c-1"].metadata is None


async def test_questions_skips_empty_chunks() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        content="   ",
    )
    c_rows["c-2"] = _sample_chunk(
        id="c-2", tenant_id=tenant_id, knowledge_base_id="kb-1", knowledge_id=doc.id,
        chunk_index=1, content="Real content to ask about.",
    )
    fake = _FakeChat(questions=["One question here?"])

    result = await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
    )

    assert len(result) == 1
    assert len(fake.calls) == 1


async def test_questions_returns_empty_when_no_text_chunks() -> None:
    tenant_id = 7
    kb = _sample_kb(id="kb-1", tenant_id=tenant_id, summary_model_id="model-sum")
    doc = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-1")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[doc.id] = doc

    result = await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=_FakeChat(questions=["One?"]),
        knowledge_repo=k_repo,
        chunk_repo=_make_chunk_repo()[0],
        kb_service=_make_kb_service(kb),
        prompt=_QUESTION_PROMPT,
    )

    assert result == []


# ── Integration (real applied schema) ─────────────────────────────────


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


def _integration_chunk(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    chunk_index: int,
    content: str,
    start_at: int = 0,
) -> Chunk:
    """Build a chunk row ready for real-DB inserts."""
    return Chunk(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=start_at,
        end_at=start_at + len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type=CHUNK_TYPE_TEXT,
        parent_chunk_id=None,
        image_info=None,
        metadata=None,
        tag_id=None,
        status=CHUNK_STATUS_STORED,
        content_hash=None,
        flags=CHUNK_FLAG_RECOMMENDED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def test_integration_process_summary_round_trip(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="summary-docs",
        kb_type="document",
        summary_model_id="model-sum",
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        type="manual",
        title="Summary me",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(session)
    await chunk_repo.create_many(
        [
            _integration_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                knowledge_id=doc.id,
                chunk_index=0,
                start_at=0,
                content="First paragraph with enough real text for a summary.",
            ),
            _integration_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                knowledge_id=doc.id,
                chunk_index=1,
                start_at=50,
                content="Second paragraph continues the same topic.",
            ),
        ]
    )
    fake = _FakeChat(content="Integrated summary text.")
    await session.commit()

    result = await process_summary(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        kb_service=kb_service,
        prompt="Summarize.",
    )
    await session.commit()

    assert result.summary == "Integrated summary text."
    stored = await KnowledgeRepository(session).get_by_id(tenant_id, doc.id)
    assert stored is not None
    assert stored.description == "Integrated summary text."
    assert stored.summary_status == SUMMARY_STATUS_COMPLETED
    all_chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": doc.id}
    )
    summary_chunks = [c for c in all_chunks if c.chunk_type == CHUNK_TYPE_SUMMARY]
    assert len(summary_chunks) == 1
    assert summary_chunks[0].content == "# Summary\nIntegrated summary text."
    assert summary_chunks[0].parent_chunk_id is not None


async def test_integration_generate_questions_round_trip(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="question-docs",
        kb_type="document",
        summary_model_id="model-sum",
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        type="manual",
        title="Question me",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(session)
    first = _integration_chunk(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        chunk_index=0,
        start_at=0,
        content="Alpha paragraph about finance.",
    )
    second = _integration_chunk(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        chunk_index=1,
        start_at=40,
        content="Beta paragraph about budgeting.",
    )
    await chunk_repo.create_many([first, second])
    fake = _FakeChat(questions=["What drives cost growth?", "How is the budget split?"])
    await session.commit()

    result = await generate_questions(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=fake,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        kb_service=kb_service,
        prompt=_QUESTION_PROMPT,
    )
    await session.commit()

    assert len(result) == 4
    stored = await chunk_repo.get_by_id(tenant_id, first.id)
    assert stored.metadata is not None
    questions = stored.metadata["generated_questions"]
    assert len(questions) == 2
    assert questions[0]["content_revision"] == 0
