"""FAQ-domain request-scoped service factories.

Repositories are built per request on the shared ``AsyncSession`` so a
mutation and its bookkeeping share one transactional unit of work; ``web``
never imports ``db``. Mirrors the pattern in
``src.core.infra.models.factory``.

The import runner additionally owns the process-wide import-task store
(the same memoisation the initialisation domain uses for its download
tasks): FAQ imports finish synchronously, but the progress endpoint is
read by a later request, so the completed progress must outlive the
request that started the import.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.faq.import_runner import FAQImportRunner, FAQImportTaskStore
from src.core.knowledge.faq.service.faq_service import FAQService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.faq_repository import FaqRepository


@lru_cache(maxsize=1)
def get_faq_import_task_store() -> FAQImportTaskStore:
    """Return the process-wide FAQ import-task store."""
    return FAQImportTaskStore()


def build_faq_service(session: AsyncSession) -> FAQService:
    """Per-request ``FAQService`` with a fresh repository."""
    return FAQService(faq_repo=FaqRepository(session))


def build_faq_import_runner(
    session: AsyncSession,
    *,
    task_store: FAQImportTaskStore | None = None,
) -> FAQImportRunner:
    """Per-request FAQ import runner over fresh repositories.

    ``task_store`` is injectable for tests; production uses the
    process-wide store so progress survives across requests.
    """
    return FAQImportRunner(
        faq_repo=FaqRepository(session),
        chunk_repo=ChunkRepository(session),
        task_store=task_store if task_store is not None else get_faq_import_task_store(),
    )


__all__ = [
    "build_faq_import_runner",
    "build_faq_service",
    "get_faq_import_task_store",
]
