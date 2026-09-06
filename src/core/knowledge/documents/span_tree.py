"""Project persisted processing spans onto the SPA timeline tree.

``GET /knowledge/{id}/spans`` needs a single root with nested children.
The store keeps a flat list keyed by ``parent_span_id``; this module
rebuilds that tree and picks the current-stage / last-error hints the
timeline header reads.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.knowledge.documents.span_tracker import (
    ROOT_SPAN_NAME,
    SPAN_KIND_ROOT,
    SPAN_KIND_STAGE,
    SPAN_STATUS_FAILED,
    SPAN_STATUS_PENDING,
    SPAN_STATUS_RUNNING,
    SpanProgress,
)
from src.db.models.knowledge_processing_span import KnowledgeProcessingSpan

_ACTIVE = frozenset({SPAN_STATUS_RUNNING, SPAN_STATUS_PENDING})


class SpanNode(BaseModel):
    """One node in the nested processing-span tree."""

    model_config = ConfigDict(frozen=True)

    span_id: str = ""
    parent_span_id: str | None = None
    name: str
    kind: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None
    error_code: str = ""
    error_message: str = ""
    input: JsonObject | None = None
    output: JsonObject | None = None
    metadata: JsonObject | None = None
    children: list[SpanNode] = Field(default_factory=list)


class SpanLastError(BaseModel):
    """Most recently finished failed span, for the timeline banner."""

    model_config = ConfigDict(frozen=True)

    name: str
    error_code: str
    error_message: str
    finished_at: datetime | None = None


class SpansRead(BaseModel):
    """GET ``/knowledge/{id}/spans`` payload the timeline already consumes."""

    model_config = ConfigDict(frozen=True)

    knowledge_id: str
    attempt: int
    latest_attempt: int
    current_attempt: int
    parse_status: str
    current_stage: str = ""
    trace: SpanNode
    last_error: SpanLastError | None = None


def empty_trace() -> SpanNode:
    """Placeholder root when the document has never been tracked."""
    return SpanNode(
        name=ROOT_SPAN_NAME,
        kind=SPAN_KIND_ROOT,
        status=SPAN_STATUS_PENDING,
    )


def spans_read_payload(
    *,
    knowledge_id: str,
    parse_status: str,
    progress: SpanProgress | None,
) -> SpansRead:
    """Build the GET data object the timeline already consumes."""
    if progress is None or not progress.spans:
        attempt = 0 if progress is None else progress.attempt
        latest = 0 if progress is None else progress.latest_attempt
        return SpansRead(
            knowledge_id=knowledge_id,
            attempt=attempt,
            latest_attempt=latest,
            current_attempt=attempt,
            parse_status=parse_status,
            trace=empty_trace(),
        )
    return SpansRead(
        knowledge_id=knowledge_id,
        attempt=progress.attempt,
        latest_attempt=progress.latest_attempt,
        current_attempt=progress.attempt,
        parse_status=parse_status,
        current_stage=_current_stage(progress.spans),
        trace=_build_tree(progress.spans),
        last_error=_last_error(progress.spans),
    )


def _span_node(row: KnowledgeProcessingSpan, children: list[SpanNode]) -> SpanNode:
    return SpanNode(
        span_id=row.span_id,
        parent_span_id=row.parent_span_id,
        name=row.name,
        kind=row.kind,
        status=row.status,
        started_at=row.started_at,
        finished_at=row.finished_at,
        duration_ms=row.duration_ms,
        error_code=row.error_code or "",
        error_message=row.error_message or "",
        input=row.input,
        output=row.output,
        metadata=row.metadata,
        children=children,
    )


def _build_tree(spans: tuple[KnowledgeProcessingSpan, ...]) -> SpanNode:
    """Nest rows under their parent; prefer the persisted root span."""
    by_id = {row.span_id: row for row in spans}
    child_ids: dict[str, list[str]] = {}
    roots: list[str] = []
    for row in spans:
        parent = row.parent_span_id
        if parent and parent in by_id:
            child_ids.setdefault(parent, []).append(row.span_id)
        else:
            roots.append(row.span_id)
    if not roots:
        return empty_trace()
    preferred = next(
        (row.span_id for row in spans if row.kind == SPAN_KIND_ROOT),
        roots[0],
    )
    seen: set[str] = set()

    def attach(span_id: str) -> SpanNode:
        seen.add(span_id)
        kids = [attach(child_id) for child_id in child_ids.get(span_id, []) if child_id not in seen]
        return _span_node(by_id[span_id], kids)

    tree = attach(preferred)
    extras = [attach(span_id) for span_id in roots if span_id not in seen]
    if extras:
        return tree.model_copy(update={"children": [*tree.children, *extras]})
    return tree


def _current_stage(spans: tuple[KnowledgeProcessingSpan, ...]) -> str:
    """First in-flight stage name, if any."""
    for row in spans:
        if row.kind == SPAN_KIND_STAGE and row.status in _ACTIVE:
            return row.name
    return ""


def _last_error(spans: tuple[KnowledgeProcessingSpan, ...]) -> SpanLastError | None:
    """Most recently finished failed span, for the timeline banner."""
    failed = [row for row in spans if row.status == SPAN_STATUS_FAILED]
    if not failed:
        return None
    row = max(failed, key=lambda item: item.finished_at or item.updated_at)
    return SpanLastError(
        name=row.name,
        error_code=row.error_code or "",
        error_message=row.error_message or "",
        finished_at=row.finished_at,
    )


__all__ = [
    "SpanLastError",
    "SpanNode",
    "SpansRead",
    "empty_trace",
    "spans_read_payload",
]
