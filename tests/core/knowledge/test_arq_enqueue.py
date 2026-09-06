"""Unit tests for the ARQ document enqueuer."""

from __future__ import annotations

from typing import cast

import pytest
from arq.connections import ArqRedis

from src.common.exception import ExternalServiceError
from src.core.knowledge.documents.arq_enqueue import ArqDocumentEnqueuer
from src.core.knowledge.documents.reparse import (
    DocumentProcessPayload as ReparseProcessPayload,
)
from src.core.knowledge.documents.upload_pipeline import (
    DocumentProcessPayload as UploadProcessPayload,
)


class _Job:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class _FakeRedis:
    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def enqueue_job(
        self,
        function: str,
        *args: object,
        **kwargs: object,
    ) -> _Job | None:
        self.calls.append((function, kwargs))
        if not self.succeed:
            return None
        return _Job("job-1")


def _upload_payload() -> UploadProcessPayload:
    return UploadProcessPayload(
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        file_path="",
        file_name="",
        file_type="html",
        url="https://example.com/page",
    )


async def test_dispatch_enqueues_document_process() -> None:
    redis = _FakeRedis()
    enqueuer = ArqDocumentEnqueuer(cast(ArqRedis, redis), queue_name="kb:queue")
    job_id = await enqueuer.dispatch(payload=_upload_payload())
    assert job_id == "job-1"
    function, kwargs = redis.calls[0]
    assert function == "document_process"
    assert kwargs["_queue_name"] == "kb:queue"
    assert kwargs["url"] == "https://example.com/page"
    assert kwargs["knowledge_id"] == "k-1"


async def test_reparse_uses_file_url_when_url_blank() -> None:
    redis = _FakeRedis()
    enqueuer = ArqDocumentEnqueuer(cast(ArqRedis, redis))
    await enqueuer.enqueue_document_process(
        tenant_id=1,
        payload=ReparseProcessPayload(
            tenant_id=1,
            knowledge_id="k-2",
            knowledge_base_id="kb-1",
            file_url="https://cdn.example.com/a.pdf",
            file_name="a.pdf",
            file_type="pdf",
        ),
    )
    _function, kwargs = redis.calls[0]
    assert kwargs["url"] == "https://cdn.example.com/a.pdf"
    assert kwargs["file_url"] == "https://cdn.example.com/a.pdf"


async def test_dispatch_raises_when_broker_dedups() -> None:
    redis = _FakeRedis(succeed=False)
    enqueuer = ArqDocumentEnqueuer(cast(ArqRedis, redis))
    with pytest.raises(ExternalServiceError, match="Failed to enqueue"):
        await enqueuer.dispatch(payload=_upload_payload())


async def test_manual_process_enqueues_named_task() -> None:
    redis = _FakeRedis()
    enqueuer = ArqDocumentEnqueuer(cast(ArqRedis, redis))
    await enqueuer.enqueue_manual_process(
        tenant_id=1,
        knowledge_id="k-3",
        content="# hello",
    )
    function, kwargs = redis.calls[0]
    assert function == "manual_process"
    assert kwargs["content"] == "# hello"
    assert kwargs["knowledge_id"] == "k-3"
