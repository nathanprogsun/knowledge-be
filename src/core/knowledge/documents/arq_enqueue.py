"""ARQ-backed document parse enqueue.

Implements the create-file dispatcher and the reparse enqueuer against
one Redis pool so URL, file, and reparse submissions share a queue name
with ``python -m src.workers.main``.
"""

from __future__ import annotations

from dataclasses import replace

from arq import create_pool
from arq.connections import ArqRedis
from arq.jobs import Job

from src.common.arq_redis import redis_settings_from_url
from src.common.exception import ExternalServiceError
from src.core.knowledge.documents.reparse import (
    DocumentProcessPayload as ReparseProcessPayload,
)
from src.core.knowledge.documents.upload_pipeline import (
    DocumentProcessPayload as UploadProcessPayload,
)

_ENQUEUE_FAILED_CODE = "document.enqueue_failed"
_ENQUEUE_FAILED_MESSAGE = "Failed to enqueue processing task"
_DOCUMENT_PROCESS_TASK = "document_process"
_MANUAL_PROCESS_TASK = "manual_process"
_DEFAULT_QUEUE_NAME = "arq:queue"


class ArqDocumentEnqueuer:
    """Broker-backed dispatcher + reparse enqueuer."""

    def __init__(
        self,
        redis: ArqRedis,
        *,
        queue_name: str = _DEFAULT_QUEUE_NAME,
    ) -> None:
        self._redis = redis
        self._queue_name = queue_name

    async def dispatch(self, *, payload: UploadProcessPayload) -> str:
        """Enqueue a create-path document-process job and return its id."""
        return await self._enqueue_document(
            tenant_id=payload.tenant_id,
            knowledge_id=payload.knowledge_id,
            knowledge_base_id=payload.knowledge_base_id,
            file_path=payload.file_path,
            file_name=payload.file_name,
            file_type=payload.file_type,
            url=payload.url,
            enable_multimodel=payload.enable_multimodel,
            enable_question_generation=payload.enable_question_generation,
            question_count=payload.question_count,
            language=payload.language,
        )

    async def enqueue_document_process(
        self,
        *,
        tenant_id: int,
        payload: ReparseProcessPayload,
    ) -> None:
        """Enqueue a reparse document-process job."""
        await self._enqueue_document(
            tenant_id=payload.tenant_id,
            knowledge_id=payload.knowledge_id,
            knowledge_base_id=payload.knowledge_base_id,
            file_path=payload.file_path or "",
            file_name=payload.file_name or "",
            file_type=payload.file_type or "",
            url=payload.url or payload.file_url or "",
            file_url=payload.file_url or "",
            enable_multimodel=payload.enable_multimodel,
            enable_question_generation=payload.enable_question_generation,
            question_count=payload.question_count,
            language=payload.language or "",
        )

    async def enqueue_manual_process(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        content: str,
    ) -> None:
        """Enqueue a manual Markdown parse job."""
        job = await self._redis.enqueue_job(
            _MANUAL_PROCESS_TASK,
            _queue_name=self._queue_name,
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id="",
            content=content,
        )
        self._require_job(job)

    async def _enqueue_document(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
        file_path: str,
        file_name: str,
        file_type: str,
        url: str,
        enable_multimodel: bool,
        enable_question_generation: bool,
        question_count: int,
        language: str,
        file_url: str = "",
    ) -> str:
        job = await self._redis.enqueue_job(
            _DOCUMENT_PROCESS_TASK,
            _queue_name=self._queue_name,
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            url=url,
            file_url=file_url,
            enable_multimodel=enable_multimodel,
            enable_question_generation=enable_question_generation,
            question_count=question_count,
            language=language,
        )
        return self._require_job(job)

    def _require_job(self, job: Job | None) -> str:
        if job is None:
            raise ExternalServiceError(
                code=_ENQUEUE_FAILED_CODE,
                message=_ENQUEUE_FAILED_MESSAGE,
            )
        return job.job_id


async def connect_arq_pool(redis_url: str) -> ArqRedis:
    """Open a short-retry ARQ pool so API startup does not hang on Redis."""
    settings = replace(
        redis_settings_from_url(redis_url),
        conn_timeout=1,
        conn_retries=1,
        conn_retry_delay=0,
    )
    return await create_pool(settings)


__all__ = [
    "ArqDocumentEnqueuer",
    "connect_arq_pool",
]
