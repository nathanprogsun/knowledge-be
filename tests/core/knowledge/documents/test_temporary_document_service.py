"""Unit tests for the temporary-document model, service helpers, and service.

Pure tests (no DB): model defaults, extension / size validation,
query-term extraction, content selection under budget, the content
chunking transform, and the lifecycle orchestration against an
in-memory repository stub.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.common.exception import ValidationError
from src.core.knowledge.documents.temporary_document import (
    TEMPORARY_DOCUMENT_CHUNK_SIZE,
    TEMPORARY_DOCUMENT_MAX_PROMPT_PARTS,
    TemporaryDocumentCreateOptions,
    TemporaryDocumentService,
    analyze_content,
    approx_text_content_runes,
    image_refs_of,
    is_image_extension,
    is_text_extension,
    is_visual_document_query,
    max_upload_bytes,
    query_terms,
    select_content_with_budget,
    supported_extension,
)
from src.db.models.temporary_document import (
    TEMPORARY_DOCUMENT_STATUS_READY,
    TEMPORARY_DOCUMENT_STATUS_UPLOADED,
    TemporaryDocument,
)

# ── Model defaults ────────────────────────────────────────────────────


def _make_doc(**overrides: object) -> TemporaryDocument:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "tenant_id": 7,
        "session_id": "session-1",
        "resource_ref": "stor/abc",
        "file_name": "note.md",
        "file_type": ".md",
        "file_size": 1024,
        "expires_at": now + timedelta(hours=24),
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return TemporaryDocument(**values)


def test_model_defaults() -> None:
    row = _make_doc()
    assert row.status == TEMPORARY_DOCUMENT_STATUS_UPLOADED
    assert row.mime_type == ""
    assert row.content is None
    assert row.chunks == []
    assert row.image_refs == []
    assert row.metadata == {}
    assert row.processing_options == {}
    assert row.token_count == 0
    assert row.chunk_count == 0
    assert row.error_message is None
    assert row.started_at is None
    assert row.ready_at is None
    assert row.deleted_at is None


def test_model_primary_key_metadata() -> None:
    row = _make_doc()
    assert TemporaryDocument.primary_keys == ("id",)
    assert TemporaryDocument.db_generated_columns == ()
    assert row.primary_key_to_value() == {"id": row.id}


# ── Extension / size helpers ──────────────────────────────────────────


def test_supported_extension_lowercases() -> None:
    assert supported_extension(".PDF")
    assert supported_extension(".pdf")
    assert supported_extension(".markdown")
    assert not supported_extension(".exe")
    assert not supported_extension(".url")


def test_text_and_image_extension_classifiers() -> None:
    assert is_text_extension(".txt")
    assert is_text_extension(".md")
    assert not is_text_extension(".pdf")
    assert is_image_extension(".png")
    assert is_image_extension(".jpeg")
    assert not is_image_extension(".pdf")


def test_max_upload_bytes_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_FILE_SIZE_MB", raising=False)
    assert max_upload_bytes() == 50 * 1024 * 1024

    monkeypatch.setenv("MAX_FILE_SIZE_MB", "3")
    assert max_upload_bytes() == 3 * 1024 * 1024

    monkeypatch.setenv("MAX_FILE_SIZE_MB", "not-a-number")
    assert max_upload_bytes() == 50 * 1024 * 1024


# ── Query / content helpers ───────────────────────────────────────────


def test_query_terms_words_and_han_bigrams() -> None:
    assert query_terms("Asyncio Event Loop") == ["asyncio", "event", "loop"]
    assert query_terms("hello hello world") == ["hello", "world"]
    assert query_terms("知识库") == ["知识库", "知识", "识库"]
    assert query_terms("a b") == []


def test_is_visual_document_query() -> None:
    assert is_visual_document_query("看下这张图")
    assert is_visual_document_query("show me the diagram")
    assert not is_visual_document_query("what is the summary")


def test_image_refs_of_is_lenient() -> None:
    raw = [
        {"url": "https://cdn/x.png", "original_ref": "p1", "mime_type": "image/png"},
        {"url": ""},  # skipped: empty url
        {"mime_type": "image/png"},  # skipped: missing url
        "not-a-dict",  # skipped
    ]
    images = image_refs_of(raw)  # type: ignore[arg-type]
    assert len(images) == 1
    assert images[0].url == "https://cdn/x.png"
    assert images[0].original_ref == "p1"
    assert image_refs_of(None) == []


def test_approx_text_content_runes_ignores_images() -> None:
    md = "![](.png) 你好"
    assert approx_text_content_runes(md) == 2


# ── Content chunking transform ────────────────────────────────────────


def test_analyze_content_empty() -> None:
    chunks, total = analyze_content("")
    assert chunks == []
    assert total == 0


def test_analyze_content_produces_chunks() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20
    chunks, total = analyze_content(text)
    assert len(chunks) >= 1
    assert total >= 1
    assert all(c.seq >= 0 for c in chunks)
    assert all(c.token_count >= 0 for c in chunks)
    assert all(c.end > c.start for c in chunks)


def test_analyze_content_respects_chunk_size() -> None:
    paragraph = ("word " * 200).strip() + "\n\n"
    chunks, _total = analyze_content(paragraph * 40)
    assert len(chunks) > 1
    assert all(len(c.content) <= TEMPORARY_DOCUMENT_CHUNK_SIZE + 1 for c in chunks)


# ── Content selection under budget ────────────────────────────────────


def _ready_doc(
    chunks: list[dict[str, object]], *, token_count: int, content: str
) -> TemporaryDocument:
    return _make_doc(
        status=TEMPORARY_DOCUMENT_STATUS_READY,
        token_count=token_count,
        chunk_count=len(chunks),
        chunks=chunks,
        content=content,
    )


def test_select_content_inlines_when_chunks_empty() -> None:
    doc = _ready_doc([], token_count=0, content="plain text")
    content, selected, total = select_content_with_budget(doc, "anything", 12000)
    assert content == "plain text"
    assert selected == 0
    assert total == 0


def test_select_content_inlines_when_short() -> None:
    chunks = [{"seq": 0, "content": "hi", "start": 0, "end": 2, "token_count": 2}]
    doc = _ready_doc(chunks, token_count=2, content="hi")
    content, selected, total = select_content_with_budget(doc, "unrelated", 12000)
    assert content == "hi"
    assert selected == 1
    assert total == 1


def test_select_content_ranks_matching_chunk() -> None:
    chunks = [
        {
            "seq": 0,
            "content": "asyncio event loop scheduling details",
            "context_header": "Python",
            "start": 0,
            "end": 40,
            "token_count": 8000,
        },
        {
            "seq": 1,
            "content": "database schema design considerations",
            "context_header": "Database",
            "start": 40,
            "end": 80,
            "token_count": 8000,
        },
    ]
    doc = _ready_doc(chunks, token_count=20000, content="full extracted text")
    content, selected, total = select_content_with_budget(doc, "asyncio", 12000)
    assert selected == 1
    assert total == 2
    assert "Python" in content
    assert "asyncio event loop" in content
    assert "database schema" not in content


def test_select_content_joins_multiple_with_separator() -> None:
    chunks = [
        {"seq": 0, "content": "first part", "start": 0, "end": 10, "token_count": 200},
        {"seq": 1, "content": "second part", "start": 10, "end": 20, "token_count": 200},
    ]
    doc = _ready_doc(chunks, token_count=20000, content="x")
    content, selected, _total = select_content_with_budget(doc, "", 12000)
    assert selected == 2
    assert content == "first part\n\n---\n\nsecond part"


def test_select_content_respects_budget_and_max_parts() -> None:
    chunks = [
        {
            "seq": i,
            "content": f"chunk number {i} unique marker",
            "start": i * 10,
            "end": i * 10 + 10,
            "token_count": 9000,
        }
        for i in range(20)
    ]
    doc = _ready_doc(chunks, token_count=200000, content="x")
    content, selected, total = select_content_with_budget(doc, "unique", 12000)
    assert total == 20
    assert 1 <= selected <= TEMPORARY_DOCUMENT_MAX_PROMPT_PARTS
    assert "chunk number" in content


# ── Service lifecycle (in-memory repo stub) ───────────────────────────


class _StubRepo:
    """Duck-typed repository capturing calls for the service tests."""

    def __init__(self) -> None:
        self.created: list[TemporaryDocument] = []
        self.expired: list[TemporaryDocument] = []
        self.deleted: list[tuple[int, str, str]] = []

    async def create(self, row: TemporaryDocument) -> TemporaryDocument:
        self.created.append(row)
        return row

    async def list_expired(self, *, before: datetime, limit: int) -> list[TemporaryDocument]:
        return self.expired[:limit]

    async def delete_scoped(
        self,
        *,
        tenant_id: int,
        session_id: str,
        document_id: str,
        now: datetime,
    ) -> bool:
        self.deleted.append((tenant_id, session_id, document_id))
        return True


async def test_create_records_uploaded_row() -> None:
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    row = await service.create(
        tenant_id=7,
        session_id="session-1",
        resource_ref="stor/abc",
        file_name="report.md",
        mime_type="text/markdown",
        file_size=2048,
    )
    assert len(repo.created) == 1
    assert row.status == TEMPORARY_DOCUMENT_STATUS_UPLOADED
    assert row.file_type == ".md"
    assert row.mime_type == "text/markdown"
    assert row.resource_ref == "stor/abc"
    assert row.processing_options == {}
    assert len(row.id) == 36
    assert row.expires_at - row.created_at == timedelta(hours=24)


async def test_create_persists_processing_options() -> None:
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    options = TemporaryDocumentCreateOptions(parser_engine="docx", image_understanding=True)
    row = await service.create(
        tenant_id=7,
        session_id="session-1",
        resource_ref="stor/abc",
        file_name="a.docx",
        mime_type="",
        file_size=1024,
        options=options,
    )
    assert row.processing_options == {
        "parser_engine": "docx",
        "image_understanding": True,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tenant_id": 0, "session_id": "s", "file_name": "a.md"},
        {"tenant_id": 7, "session_id": "  ", "file_name": "a.md"},
        {"tenant_id": 7, "session_id": "s", "file_name": "a.exe"},
        {"tenant_id": 7, "session_id": "s", "file_name": "   "},
    ],
)
async def test_create_rejects_invalid_input(kwargs: dict[str, object]) -> None:
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        await service.create(
            resource_ref="stor/abc",
            mime_type="",
            file_size=1024,
            **kwargs,
        )


async def test_create_rejects_oversize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        await service.create(
            tenant_id=7,
            session_id="session-1",
            resource_ref="stor/abc",
            file_name="big.md",
            mime_type="",
            file_size=1024 * 1024 + 1,
        )
    assert repo.created == []


async def test_create_reduces_file_name_to_basename() -> None:
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    row = await service.create(
        tenant_id=7,
        session_id="session-1",
        resource_ref="stor/abc",
        file_name="../dir/note.md",
        mime_type="",
        file_size=100,
    )
    assert row.file_name == "note.md"


async def test_delete_soft_deletes_scoped() -> None:
    repo = _StubRepo()
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    result = await service.delete(tenant_id=7, session_id="session-1", document_id="doc-1")
    assert result is True
    assert repo.deleted == [(7, "session-1", "doc-1")]


async def test_cleanup_expired_sweeps_stale_rows() -> None:
    repo = _StubRepo()
    repo.expired = [_make_doc(id=str(uuid.uuid4())), _make_doc(id=str(uuid.uuid4()))]
    service = TemporaryDocumentService(repo=repo)  # type: ignore[arg-type]
    removed = await service.cleanup_expired()
    assert removed == 2
    assert len(repo.deleted) == 2
    for tenant_id, _session_id, _document_id in repo.deleted:
        assert tenant_id == 7
    # Second sweep finds nothing left.
    repo.expired = []
    assert await service.cleanup_expired() == 0
