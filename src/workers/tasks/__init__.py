"""ARQ worker task handlers.

Each module in this package registers itself with the worker registry
at import time via the :func:`register_task` decorator. Importing the
package (or any subpackage member) is what makes the handler visible
to :func:`src.workers.registry.all_functions`.

New task modules should be added here so the default worker process
loads them on startup.
"""

from __future__ import annotations

from src.workers.tasks import document_process  # noqa: F401

__all__ = ["document_process"]