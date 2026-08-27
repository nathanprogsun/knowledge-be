"""ARQ worker task handlers.

Each module in this package registers itself with the worker registry
at import time via the :func:`register_task` decorator. Importing the
package (or any subpackage member) is what makes the handler visible
to :func:`src.workers.registry.all_functions`.

The :mod:`src.workers.main` entry point imports this package for its
side effect so that the default worker serves the full set.
"""

from __future__ import annotations

from src.workers.tasks import (
    chunk_extract,
    datasource_sync,
    datatable_summary,
    document_process,
    faq_import,
    image_multimodal,
    index_delete,
    kb_clone,
    kb_delete,
    knowledge_list_delete,
    knowledge_list_reparse,
    knowledge_move,
    knowledge_post_process,
    manual_process,
    question_generation,
    summary_generation,
    temporary_document,
    wiki_finalize,
    wiki_ingest,
)

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
