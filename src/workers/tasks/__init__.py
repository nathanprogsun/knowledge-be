"""ARQ worker task handlers.

Each module in this package registers itself with the worker registry
at import time via the :func:`register_task` decorator. Importing the
package (or any subpackage member) is what makes the handler visible
to :func:`src.workers.registry.all_functions`.

The :mod:`src.workers.main` entry point imports this package for its
side effect so that the default worker serves the full set.
"""

from __future__ import annotations

from src.workers.tasks import document_process  # noqa: F401
from src.workers.tasks import manual_process  # noqa: F401
from src.workers.tasks import temporary_document  # noqa: F401

__all__ = ["document_process", "manual_process", "temporary_document"]
