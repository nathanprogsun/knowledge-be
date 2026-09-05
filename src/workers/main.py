"""ARQ worker entry point.

Builds the ARQ ``Worker`` from settings + registry and runs it. Invoke
with ``python -m src.workers.main``.
"""

from __future__ import annotations

import asyncio

from arq.worker import Function

from src.core.knowledge.documents.process_runtime import build_document_process_runtime
from src.workers import tasks  # noqa: F401  (registers task handlers)
from src.workers.base import BaseWorker, WorkerContext
from src.workers.registry import all_functions
from src.workers.runtime_ctx import (
    document_process_runtime_from_ctx,
    put_document_process_runtime,
)
from src.workers.settings import get_worker_settings


class DefaultWorker(BaseWorker):
    """Worker serving every task currently in the registry."""

    @property
    def functions(self) -> list[Function]:
        return all_functions()

    async def startup(self, ctx: WorkerContext) -> None:
        """Compose the document-process runtime once per worker process."""
        put_document_process_runtime(ctx, build_document_process_runtime())

    async def shutdown(self, ctx: WorkerContext) -> None:
        """Dispose the document-process engine and reader channel."""
        runtime = document_process_runtime_from_ctx(ctx)
        if runtime is not None:
            await runtime.aclose()


async def main() -> None:
    """Build the default worker and run it until stopped."""
    worker = DefaultWorker(get_worker_settings())
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
