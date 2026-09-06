"""Typed extras the worker startup stores on the ARQ context dict."""

from __future__ import annotations

from typing import TypedDict, cast

from src.core.knowledge.documents.process_runtime import DocumentProcessRuntime
from src.workers.base import WorkerContext


class WorkerRuntimeBag(TypedDict, total=False):
    """Optional keys the default worker writes during ``startup``."""

    document_process_runtime: DocumentProcessRuntime


def put_document_process_runtime(
    ctx: WorkerContext,
    runtime: DocumentProcessRuntime,
) -> None:
    """Attach the composed document-process runtime to the ARQ context."""
    bag = cast(WorkerRuntimeBag, ctx)
    bag["document_process_runtime"] = runtime


def document_process_runtime_from_ctx(ctx: WorkerContext) -> DocumentProcessRuntime | None:
    """Return the startup-wired runtime, or ``None`` in unit tests."""
    bag = cast(WorkerRuntimeBag, ctx)
    return bag.get("document_process_runtime")


__all__ = [
    "WorkerRuntimeBag",
    "document_process_runtime_from_ctx",
    "put_document_process_runtime",
]
