"""Worker-side document-process composition.

The worker process must not import ``db``. This module builds the engine,
reader, and per-job pipeline inside ``core`` and exposes a runner the
ARQ handler can call from its startup-wired context.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.ai.docreader.client import new_client
from src.common.session_provider import session_scope
from src.core.knowledge.documents.docreader_adapter import DocReaderAdapter
from src.core.knowledge.documents.factory import build_document_process_pipeline
from src.core.knowledge.documents.parse_pipeline import DocumentReader
from src.core.knowledge.documents.process_document import ProcessOutcome
from src.core.knowledge.documents.process_document import (
    process_document as run_process_document,
)
from src.db.base import DatabaseEngine
from src.settings import get_settings


class DocumentProcessRuntime:
    """Session factory + reader used for one worker process lifetime."""

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        reader: DocumentReader,
    ) -> None:
        self._engine = engine
        self._reader = reader

    async def run_document_process(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
        file_path: str = "",
        file_name: str = "",
        file_type: str = "",
        url: str = "",
        enable_multimodel: bool = False,
        language: str = "",
        request_id: str = "",
    ) -> ProcessOutcome:
        """Open a job session, compose the pipeline, and process one document."""
        async with session_scope(self._engine.session_factory) as session:
            pipeline = build_document_process_pipeline(session, reader=self._reader)
            return await run_process_document(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                file_path=file_path,
                file_name=file_name,
                file_type=file_type,
                url=url,
                enable_multimodel=enable_multimodel,
                language=language,
                request_id=request_id,
                now=datetime.now(UTC),
                pipeline=pipeline,
            )

    async def aclose(self) -> None:
        """Dispose the engine and close the reader channel when it is ours."""
        if isinstance(self._reader, DocReaderAdapter):
            self._reader.close()
        await self._engine.close()


def build_document_process_runtime() -> DocumentProcessRuntime:
    """Construct the worker runtime from process settings."""
    settings = get_settings()
    engine = DatabaseEngine(
        url=settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    reader = DocReaderAdapter(new_client(settings.docreader_addr))
    return DocumentProcessRuntime(engine=engine, reader=reader)


__all__ = [
    "DocumentProcessRuntime",
    "build_document_process_runtime",
]
