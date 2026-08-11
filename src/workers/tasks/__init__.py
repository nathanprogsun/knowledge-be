"""ARQ worker task handlers.

Each module in this package registers itself with the worker registry
at import time via the :func:`register_task` decorator. Importing the
package (or any subpackage member) is what makes the handler visible
to :func:`src.workers.registry.all_functions`.

The :mod:`src.workers.main` entry point imports this package for its
side effect so that the default worker serves the full set.
"""

from __future__ import annotations

from src.workers.tasks import chunk_extract  # noqa: F401
from src.workers.tasks import datasource_sync  # noqa: F401
from src.workers.tasks import datatable_summary  # noqa: F401
from src.workers.tasks import document_process  # noqa: F401
from src.workers.tasks import faq_import  # noqa: F401
from src.workers.tasks import image_multimodal  # noqa: F401
from src.workers.tasks import kb_clone  # noqa: F401
from src.workers.tasks import kb_delete  # noqa: F401
from src.workers.tasks import index_delete  # noqa: F401
from src.workers.tasks import knowledge_list_delete  # noqa: F401
from src.workers.tasks import knowledge_list_reparse  # noqa: F401
from src.workers.tasks import knowledge_move  # noqa: F401
from src.workers.tasks import knowledge_post_process  # noqa: F401
from src.workers.tasks import manual_process  # noqa: F401
from src.workers.tasks import question_generation  # noqa: F401
from src.workers.tasks import summary_generation  # noqa: F401
from src.workers.tasks import temporary_document  # noqa: F401
from src.workers.tasks import wiki_finalize  # noqa: F401
from src.workers.tasks import wiki_ingest  # noqa: F401

__all__ = [
    "chunk_extract",
    "datasource_sync",
    "datatable_summary",
    "document_process",
    "faq_import",
    "image_multimodal",
    "index_delete",
    "kb_clone",
    "kb_delete",
    "knowledge_list_delete",
    "knowledge_list_reparse",
    "knowledge_move",
    "knowledge_post_process",
    "manual_process",
    "question_generation",
    "summary_generation",
    "temporary_document",
    "wiki_finalize",
    "wiki_ingest",
]
