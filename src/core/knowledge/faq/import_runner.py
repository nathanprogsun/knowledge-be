"""FAQ batch-import runner — executes the merged import pipeline.

The FAQ file import (``import_faq``) is synchronous: the parse, validate
and persist pass runs inside the request. This runner adds the task
bookkeeping the progress endpoint needs — a generated ``task_id`` and a
recorded, completed progress object — so the same import can be reported
back to a later polling request. Mirrors the initialization domain's
download-task store: the completed progress is retained process-wide
because the poller is a different request.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import cast

from src.common.json import JsonObject
from src.core.contracts.knowledge import FAQImportTaskProgress
from src.core.knowledge.documents.faq_import import (
    FAQ_BATCH_MODE_APPEND,
    FailedEntry,
    FAQImportResult,
    ImportedEntry,
    import_faq,
)
from src.core.knowledge.faq.task_ids import generate_task_id
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.faq_repository import FaqRepository

#: Wire ``status`` of a completed (synchronous) FAQ import.
_STATUS_COMPLETED = "completed"

#: Wire ``progress`` percent of a finished import.
_PROGRESS_COMPLETE = 100


class FAQImportTaskStore:
    """Process-wide in-memory store of completed FAQ import tasks.

    FAQ imports run synchronously inside the request that started them,
    but the progress endpoint is polled by a later request, so the
    completed progress is retained here keyed by ``task_id``. A bounded /
    durable store is future work; the initialisation domain's download
    tasks use the same in-memory pattern.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, FAQImportTaskProgress] = {}

    def put(self, progress: FAQImportTaskProgress) -> None:
        """Record a completed import task."""
        self._tasks[progress.task_id] = progress

    def get(self, task_id: str) -> FAQImportTaskProgress | None:
        """Return the recorded progress, or ``None`` when unknown."""
        return self._tasks.get(task_id)


class FAQImportRunner:
    """Run one FAQ file import and persist its completed progress."""

    def __init__(
        self,
        *,
        faq_repo: FaqRepository,
        chunk_repo: ChunkRepository,
        task_store: FAQImportTaskStore,
    ) -> None:
        self._faq_repo = faq_repo
        self._chunk_repo = chunk_repo
        self._task_store = task_store

    async def run(
        self,
        *,
        file_data: bytes,
        filename: str,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        mode: str | None = None,
        dry_run: bool = False,
    ) -> FAQImportTaskProgress:
        """Run the merged import pipeline and record the outcome.

        The pipeline is synchronous, so the returned progress always
        describes a completed task; the progress endpoint reads it back
        by ``task_id``.
        """
        started = datetime.now(UTC)
        result = await import_faq(
            file_data=file_data,
            filename=filename,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            faq_repo=self._faq_repo,
            chunk_repo=self._chunk_repo,
            mode=mode or FAQ_BATCH_MODE_APPEND,
            dry_run=dry_run,
        )
        finished = datetime.now(UTC)
        progress = _result_to_progress(
            result=result,
            task_id=generate_task_id(tenant_id=tenant_id),
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            dry_run=dry_run,
            started=started,
            finished=finished,
        )
        self._task_store.put(progress)
        return progress

    def get_progress(self, task_id: str) -> FAQImportTaskProgress | None:
        """Return the recorded progress for ``task_id``, or ``None``."""
        return self._task_store.get(task_id)


def _result_to_progress(
    *,
    result: FAQImportResult,
    task_id: str,
    knowledge_base_id: str,
    knowledge_id: str,
    dry_run: bool,
    started: datetime,
    finished: datetime,
) -> FAQImportTaskProgress:
    """Project an import result onto the task-progress wire shape.

    The merged pipeline cannot produce partial failures, a failed-entries
    download URL, or a merge report, so those fields stay at their
    contract defaults.
    """
    processed = result.success_count + result.failed_count
    return FAQImportTaskProgress(
        task_id=task_id,
        kb_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        status=_STATUS_COMPLETED,
        progress=_PROGRESS_COMPLETE,
        total=result.total,
        processed=processed,
        success_count=result.success_count,
        failed_count=result.failed_count,
        skipped_count=result.skipped_count,
        failed_entries=[_failed_to_json(entry) for entry in result.failed_entries] or None,
        success_entries=[_imported_to_json(entry) for entry in result.success_entries] or None,
        created_at=int(started.timestamp()),
        updated_at=int(finished.timestamp()),
        dry_run=dry_run,
        import_mode=result.mode,
        imported_at=finished if not dry_run else None,
        processing_time=int((finished - started).total_seconds() * 1000),
    )


def _failed_to_json(entry: FailedEntry) -> JsonObject:
    """Project one failed-entry report onto the wire detail shape."""
    return cast(JsonObject, asdict(entry))


def _imported_to_json(entry: ImportedEntry) -> JsonObject:
    """Project one imported-entry summary onto the wire detail shape."""
    return cast(JsonObject, asdict(entry))


__all__ = [
    "FAQImportRunner",
    "FAQImportTaskStore",
]
