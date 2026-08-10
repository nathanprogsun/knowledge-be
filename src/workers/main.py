"""ARQ worker entry point.

Builds the ARQ ``Worker`` from settings + registry and runs it. Invoke
with ``python -m src.workers.main``.
"""

from __future__ import annotations

import asyncio

from arq.worker import Function

from src.workers.base import BaseWorker
from src.workers.registry import all_functions
from src.workers.settings import get_worker_settings


class DefaultWorker(BaseWorker):
    """Worker serving every task currently in the registry."""

    @property
    def functions(self) -> list[Function]:
        return all_functions()


async def main() -> None:
    """Build the default worker and run it until stopped."""
    worker = DefaultWorker(get_worker_settings())
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
