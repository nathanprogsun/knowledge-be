"""Unit tests for the ARQ ``document_process`` worker task.

The handler is a thin shim over
:func:`src.core.knowledge.documents.process_document.process_document`; the
tests patch the core function so they run without a database / AI
provider. They cover:

- payload validation (required ids, optional fields, defaults),
- registry wiring (the handler registers under ``"document_process"``),
- delegation: the parsed fields reach the core function with the right
  names and types,
- result serialisation: the returned dict matches
  :class:`ProcessOutcome` semantics.

No real ARQ broker, no real DB, no real provider calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.common.exception import ExternalServiceError
from src.core.knowledge.documents.create_file import StorageResolver
from src.core.knowledge.documents.file_url_store import FileUrlDownload
from src.core.knowledge.documents.parse_pipeline import ParseResult, ReadRequest
from src.core.knowledge.documents.process_document import (
    DocumentProcessPipeline,
    ProcessOutcome,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks.document_process import (
    parse_payload,
    task_document_process,
)


def _make_ctx() -> WorkerContext:
    """Build the minimal ARQ context the handler receives."""
    from datetime import UTC, datetime

    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


def _base_payload() -> dict[str, Any]:
    """Required-field payload shared across delegation tests."""
    return {
        "tenant_id": 1,
        "knowledge_id": "k-1",
        "knowledge_base_id": "kb-1",
    }


class _StubPipeline:
    """Object standing in for ``DocumentProcessPipeline`` in injection tests.

    The worker handler only checks identity (so it forwards the same
    object into the core call); it never instantiates the real
    pipeline, so a sentinel is sufficient.
    """


# ── Registry ─────────────────────────────────────────────────────────


def test_handler_registered_under_document_process() -> None:
    """The decorator registers the handler at import time."""
    assert get_task("document_process") is task_document_process


# ── Payload model ────────────────────────────────────────────────────


def test_parse_payload_accepts_minimum_required_fields() -> None:
    parsed = parse_payload(_base_payload())
    assert parsed.tenant_id == 1
    assert parsed.knowledge_id == "k-1"
    assert parsed.knowledge_base_id == "kb-1"
    assert parsed.file_path == ""
    assert parsed.url == ""
    assert parsed.file_url == ""
    assert parsed.enable_multimodel is False
    assert parsed.enable_question_generation is False
    assert parsed.question_count == 3
    assert parsed.language == ""
    assert parsed.request_id == ""


def test_parse_payload_accepts_all_optional_fields() -> None:
    parsed = parse_payload(
        {
            **_base_payload(),
            "request_id": "req-1",
            "file_path": "tenants/1/docs/file.pdf",
            "file_name": "file.pdf",
            "file_type": "pdf",
            "url": "https://example.com/page",
            "file_url": "https://cdn.example.com/file.pdf",
            "enable_multimodel": True,
            "enable_question_generation": True,
            "question_count": 7,
            "language": "en-US",
        }
    )
    assert parsed.request_id == "req-1"
    assert parsed.file_path == "tenants/1/docs/file.pdf"
    assert parsed.file_name == "file.pdf"
    assert parsed.file_type == "pdf"
    assert parsed.url == "https://example.com/page"
    assert parsed.file_url == "https://cdn.example.com/file.pdf"
    assert parsed.enable_multimodel is True
    assert parsed.enable_question_generation is True
    assert parsed.question_count == 7
    assert parsed.language == "en-US"


def test_parse_payload_rejects_missing_tenant() -> None:
    payload = _base_payload()
    payload.pop("tenant_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_knowledge_id() -> None:
    payload = _base_payload()
    payload.pop("knowledge_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_knowledge_base_id() -> None:
    payload = _base_payload()
    payload.pop("knowledge_base_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_ignores_unknown_fields() -> None:
    parsed = parse_payload({**_base_payload(), "extra": "ignored"})
    assert parsed.tenant_id == 1


# ── Delegation ───────────────────────────────────────────────────────


async def test_task_delegates_to_core_process_document() -> None:
    captured: dict[str, Any] = {}

    async def _fake_core(**kwargs: Any) -> ProcessOutcome:
        captured.update(kwargs)
        return ProcessOutcome(
            parse_status="processing",
            enable_status="enabled",
            summary_status="none",
            storage_size=0,
            text_chunk_count=0,
        )

    with patch(
        "src.workers.tasks.document_process._core_process_document",
        side_effect=_fake_core,
    ):
        result = await task_document_process(
            _make_ctx(),
            **_base_payload(),
            file_path="tenants/1/docs/x.pdf",
            file_name="x.pdf",
            file_type="pdf",
            enable_multimodel=True,
            language="en-US",
            request_id="req-1",
        )

    assert captured["tenant_id"] == 1
    assert captured["knowledge_id"] == "k-1"
    assert captured["knowledge_base_id"] == "kb-1"
    assert captured["file_path"] == "tenants/1/docs/x.pdf"
    assert captured["file_name"] == "x.pdf"
    assert captured["file_type"] == "pdf"
    assert captured["enable_multimodel"] is True
    assert captured["language"] == "en-US"
    assert captured["request_id"] == "req-1"
    assert captured["url"] == ""
    assert captured["pipeline"] is None

    assert result == {
        "parse_status": "processing",
        "enable_status": "enabled",
        "summary_status": "none",
        "storage_size": 0,
        "error_message": None,
        "text_chunk_count": 0,
        "skipped": False,
    }


async def test_task_uses_file_url_when_url_blank() -> None:
    captured: dict[str, Any] = {}

    async def _fake_core(**kwargs: Any) -> ProcessOutcome:
        captured.update(kwargs)
        return ProcessOutcome(parse_status="pending")

    with patch(
        "src.workers.tasks.document_process._core_process_document",
        side_effect=_fake_core,
    ):
        await task_document_process(
            _make_ctx(),
            **_base_payload(),
            file_url="https://cdn.example.com/a.pdf",
        )

    assert captured["url"] == "https://cdn.example.com/a.pdf"


async def test_task_forwards_injected_pipeline() -> None:
    pipeline = _StubPipeline()
    captured: dict[str, Any] = {}

    async def _fake_core(**kwargs: Any) -> ProcessOutcome:
        captured.update(kwargs)
        return ProcessOutcome(parse_status="pending")

    with patch(
        "src.workers.tasks.document_process._core_process_document",
        side_effect=_fake_core,
    ):
        await task_document_process(
            _make_ctx(),
            pipeline=pipeline,
            **_base_payload(),
        )

    assert captured["pipeline"] is pipeline


async def test_task_serialises_error_outcome() -> None:
    async def _fake_core(**kwargs: Any) -> ProcessOutcome:
        return ProcessOutcome(
            parse_status="failed",
            error_message="boom",
            skipped=True,
        )

    with patch(
        "src.workers.tasks.document_process._core_process_document",
        side_effect=_fake_core,
    ):
        result = await task_document_process(_make_ctx(), **_base_payload())

    assert result["parse_status"] == "failed"
    assert result["error_message"] == "boom"
    assert result["skipped"] is True


async def test_task_rejects_invalid_payload() -> None:
    """Invalid payloads surface as Pydantic validation errors."""
    with pytest.raises(ValidationError):
        await task_document_process(_make_ctx(), tenant_id="not-an-int")


# ── file_url persist bytes ────────────────────────────────────────────

_NOW = datetime(2026, 9, 5, tzinfo=UTC)


def _doc(
    *,
    knowledge_type: str,
    source: str,
    file_path: str | None = None,
    file_name: str | None = "report.pdf",
    file_type: str = "pdf",
) -> Document:
    return Document.model_validate(
        {
            "id": "k-1",
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "type": knowledge_type,
            "title": "report.pdf",
            "source": source,
            "channel": "web",
            "parse_status": "pending",
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "disabled",
            "file_name": file_name,
            "file_type": file_type,
            "file_size": 0,
            "file_path": file_path,
            "storage_size": 0,
            "custom_metadata": {},
            "created_at": _NOW,
            "updated_at": _NOW,
        }
    )


class _Repo:
    def __init__(self, row: Document) -> None:
        self.row = row

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        if self.row.tenant_id != tenant_id or self.row.id != id:
            return None
        return self.row

    async def update(self, row: Document) -> Document:
        self.row = row
        return row


class _Chunks:
    async def delete_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str, now: datetime
    ) -> int:
        return 0

    async def create_many(self, rows: list[Chunk]) -> list[Chunk]:
        return rows


class _KB:
    def __init__(self) -> None:
        self.info = KnowledgeBaseInfo(
            id="kb-1",
            name="kb",
            tenant_id=1,
            created_at=_NOW,
            updated_at=_NOW,
        )

    async def get_knowledge_base_by_id(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        return self.info


class _Reader:
    def __init__(self) -> None:
        self.requests: list[ReadRequest] = []

    async def read(self, request: ReadRequest) -> ParseResult:
        self.requests.append(request)
        return ParseResult(markdown_content="# body")


class _SaveBytes:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, str, bool]] = []

    async def save_bytes(self, *, data: bytes, tenant_id: int, file_name: str, temp: bool) -> str:
        self.calls.append((data, tenant_id, file_name, temp))
        return "local://1/kb/report.pdf"


class _Resolver:
    def __init__(self, service: _SaveBytes) -> None:
        self.service = service

    async def resolve_file_service(self, *, knowledge_base_id: str, tenant_id: int) -> _SaveBytes:
        return self.service


class _Downloader:
    def __init__(self, *, data: bytes = b"%PDF-1.4", content_type: str = "application/pdf") -> None:
        self.data = data
        self.content_type = content_type
        self.urls: list[str] = []

    async def download(self, *, url: str) -> FileUrlDownload:
        self.urls.append(url)
        return FileUrlDownload(data=self.data, content_type=self.content_type)


class _FailingDownloader:
    async def download(self, *, url: str) -> FileUrlDownload:
        raise ExternalServiceError(
            code="knowledge.file_url_fetch_failed",
            message="file URL download failed",
        )


class _TaskCtx:
    is_background_task: bool = True


def _pipeline(
    *,
    repo: _Repo,
    reader: _Reader,
    resolver: _Resolver,
    downloader: _Downloader | _FailingDownloader,
) -> DocumentProcessPipeline:
    return DocumentProcessPipeline(
        knowledge_repo=cast(KnowledgeRepository, repo),
        kb_service=cast(KBService, _KB()),
        chunk_repo=cast(ChunkRepository, _Chunks()),
        reader=reader,
        file_service_resolver=cast(StorageResolver, resolver),
        file_url_downloader=downloader,
    )


async def test_pipeline_file_url_empty_path_calls_save_bytes() -> None:
    source = "https://cdn.example.com/report.pdf"
    repo = _Repo(_doc(knowledge_type="file_url", source=source, file_path=""))
    reader = _Reader()
    file_service = _SaveBytes()
    downloader = _Downloader()
    pipeline = _pipeline(
        repo=repo,
        reader=reader,
        resolver=_Resolver(file_service),
        downloader=downloader,
    )

    outcome = await pipeline.run(
        ctx=_TaskCtx(),
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        file_path="",
        file_name="report.pdf",
        file_type="pdf",
        url=source,
        now=_NOW,
    )

    assert file_service.calls == [(b"%PDF-1.4", 1, "report.pdf", False)]
    assert downloader.urls == [source]
    assert repo.row.file_path == "local://1/kb/report.pdf"
    assert reader.requests[0].file_content == b"%PDF-1.4"
    assert reader.requests[0].url == ""
    assert outcome.parse_status == "completed"
    assert outcome.skipped is False


async def test_pipeline_url_row_skips_save_bytes() -> None:
    source = "https://finance.sina.com.cn/stock/page"
    repo = _Repo(
        _doc(
            knowledge_type="url",
            source=source,
            file_path=None,
            file_name=None,
            file_type="html",
        )
    )
    reader = _Reader()
    file_service = _SaveBytes()
    downloader = _Downloader(data=b"<html></html>", content_type="text/html")
    pipeline = _pipeline(
        repo=repo,
        reader=reader,
        resolver=_Resolver(file_service),
        downloader=downloader,
    )

    await pipeline.run(
        ctx=_TaskCtx(),
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        file_path="",
        file_name="",
        file_type="html",
        url=source,
        now=_NOW,
    )

    assert file_service.calls == []
    assert downloader.urls == []
    assert repo.row.file_path is None
    assert reader.requests[0].url == source
    assert reader.requests[0].file_content is None


async def test_pipeline_file_url_fetch_failure_marks_failed() -> None:
    source = "https://cdn.example.com/missing.pdf"
    repo = _Repo(_doc(knowledge_type="file_url", source=source, file_path=""))
    pipeline = _pipeline(
        repo=repo,
        reader=_Reader(),
        resolver=_Resolver(_SaveBytes()),
        downloader=_FailingDownloader(),
    )

    outcome = await pipeline.run(
        ctx=_TaskCtx(),
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        file_path="",
        file_name="missing.pdf",
        file_type="pdf",
        url=source,
        now=_NOW,
    )

    assert outcome.parse_status == "failed"
    assert outcome.error_message == "file URL download failed"
    assert repo.row.parse_status == "failed"
    assert repo.row.file_path == ""
