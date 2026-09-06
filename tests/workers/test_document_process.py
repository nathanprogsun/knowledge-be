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

from typing import Any, cast
from unittest.mock import patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.knowledge.documents.process_document import ProcessOutcome
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
