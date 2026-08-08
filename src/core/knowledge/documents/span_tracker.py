"""Processing-span tracker — per-attempt progress trees for the pipeline.

Facade for recording and querying per-attempt progress trees, mirroring
Langfuse's vocabulary (root / span / generation) so the UI's mental model
matches what operators already use for LLM call observability.

Lifecycle::

    root, attempt = await tracker.open_attempt(knowledge_id, trace_id)
    stage = await tracker.begin_stage(knowledge_id, attempt, "chunking", {"n": 3})
    await tracker.end_span(stage, {"chunks": 12})
    await tracker.finalize_attempt(knowledge_id, attempt, "done")

All operations are best-effort: a store error is logged and swallowed so
a tracker hiccup never breaks the parsing pipeline. ``documents``
``parse_status`` remains the authoritative source of completion.

The module takes its persistence dependency (a ``SpanStore``, satisfied
by ``KnowledgeSpanRepository``) and an optional ``Heartbeat`` seam as
constructor arguments — the web layer composes these later. Validation
that can break a caller (progress querying for a knowledge that was
never tracked) raises ``ValidationError`` / ``NotFoundError``; pipeline
state transitions keep nil-safe semantics so a bad argument degrades to
a no-op instead of an exception.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from src.app_logging import logger
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.db.models.knowledge_processing_span import KnowledgeProcessingSpan

# ── Span vocabulary (mirrors the upstream contract) ────────────────────

# Span kinds — kept narrow because every kind has dedicated rendering on
# the frontend timeline.
SPAN_KIND_ROOT = "root"
SPAN_KIND_STAGE = "stage"
SPAN_KIND_SUBSPAN = "subspan"
SPAN_KIND_GENERATION = "generation"

# Span statuses. "failed" means this span itself errored; "cancelled"
# means an upstream span failed and this one was abandoned without
# running — the UI renders the two causes differently.
SPAN_STATUS_PENDING = "pending"
SPAN_STATUS_RUNNING = "running"
SPAN_STATUS_DONE = "done"
SPAN_STATUS_FAILED = "failed"
SPAN_STATUS_SKIPPED = "skipped"
SPAN_STATUS_CANCELLED = "cancelled"

# Stage names — the closed set the UI builds its five-segment timeline
# from. Adding a stage requires a coordinated frontend release. Subspan
# names are free-form and do not go through this list.
STAGE_DOCREADER = "docreader"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_MULTIMODAL = "multimodal"
STAGE_POSTPROCESS = "postprocess"

ALL_STAGES: tuple[str, ...] = (
    STAGE_DOCREADER,
    STAGE_CHUNKING,
    STAGE_EMBEDDING,
    STAGE_MULTIMODAL,
    STAGE_POSTPROCESS,
)

# Stage DAG used to cascade-cancel dependents when a stage fails. A
# chunking failure silently turns embedding / multimodal / postprocess
# into "cancelled" so the timeline shows a clear blast radius instead of
# three pending spinners. Multimodal does NOT depend on embedding; they
# share chunking as their upstream and are otherwise independent.
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    STAGE_DOCREADER: (),
    STAGE_CHUNKING: (STAGE_DOCREADER,),
    STAGE_EMBEDDING: (STAGE_CHUNKING,),
    STAGE_MULTIMODAL: (STAGE_CHUNKING,),
    STAGE_POSTPROCESS: (STAGE_EMBEDDING, STAGE_MULTIMODAL),
}

# Root span name every attempt's trace hangs off.
ROOT_SPAN_NAME = "knowledge_processing"

# Matches the spans ``name`` column (VARCHAR(255)); names beyond this are
# truncated with an 8-hex hash suffix so concurrent subspans stay distinct.
MAX_SPAN_NAME_LEN = 255

# Failures record the message verbatim for admin views, bounded to keep
# the trace table readable.
MAX_ERROR_MESSAGE_LEN = 1024
MAX_ERROR_DETAIL_LEN = 8192

# Terminal statuses accepted by ``finalize_attempt``.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        SPAN_STATUS_DONE,
        SPAN_STATUS_FAILED,
        SPAN_STATUS_CANCELLED,
        SPAN_STATUS_SKIPPED,
    }
)

# Sentinel for a persisted row without a start timestamp; mirrors the
# zero-time fallback so a duration can never be computed from it.
_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def fit_span_name(name: str) -> str:
    """Truncate a span name to fit the DB column, rune-aware.

    Wiki ingestion builds names like ``postprocess.wiki.page[<slug>]``
    that can exceed 255 characters when the slug is a long romanized
    entity name; when truncated an 8-hex hash suffix keeps concurrent
    subspans distinct. Truncation is rune-aware to match PostgreSQL
    VARCHAR character semantics and avoid splitting multi-byte
    sequences.
    """
    runes = list(name)
    if len(runes) <= MAX_SPAN_NAME_LEN:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    suffix = f"~{digest[:4].hex()}"
    suffix_runes = list(suffix)
    keep = MAX_SPAN_NAME_LEN - len(suffix_runes)
    if keep < 1:
        if len(suffix_runes) > MAX_SPAN_NAME_LEN:
            return "".join(suffix_runes[:MAX_SPAN_NAME_LEN])
        return suffix
    return "".join(runes[:keep]) + suffix


def new_span_id() -> str:
    """Return a hex-only span id (32 chars, no dashes).

    Stripping the dashes saves bytes per row and keeps the id friendly
    to paste into queries.
    """
    return uuid.uuid4().hex


def stages_depending_on(stage: str) -> list[str]:
    """Transitive closure of stages that depend on ``stage``.

    Reverse-walks the stage DAG; bounded to five members so the naive
    walk is cheap.
    """
    out: list[str] = []
    seen: set[str] = set()
    frontier: list[str] = [stage]
    while frontier:
        nxt: list[str] = []
        for candidate in ALL_STAGES:
            if candidate in seen:
                continue
            if any(dep in frontier for dep in STAGE_DEPENDENCIES[candidate]):
                seen.add(candidate)
                out.append(candidate)
                nxt.append(candidate)
        frontier = nxt
    return out


def is_main_pipeline_stage(name: str) -> bool:
    """Report whether ``name`` is one of the five mandatory stages.

    A failure in any of these terminally invalidates the attempt and
    must close the root as failed. Optional downstream stages added
    later do not match — they can fail individually without poisoning
    the parsed document.
    """
    return name in ALL_STAGES


@dataclass(frozen=True)
class Span:
    """In-memory handle the pipeline holds while a span is executing.

    Carries enough context for End/Fail/Skip to write back without
    re-querying the store. Returned (and required) from every begin
    operation.
    """

    knowledge_id: str
    attempt: int
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    status: str
    started_at: datetime


@dataclass(frozen=True)
class SpanProgress:
    """Progress-query projection: the attempt's flat span list."""

    knowledge_id: str
    attempt: int
    latest_attempt: int
    spans: tuple[KnowledgeProcessingSpan, ...]


class SpanStore(Protocol):
    """Persistence surface the tracker needs (satisfied by the repo)."""

    async def upsert(self, row: KnowledgeProcessingSpan) -> KnowledgeProcessingSpan: ...
    async def next_attempt(self, knowledge_id: str) -> int: ...
    async def latest_attempt(self, knowledge_id: str) -> int: ...
    async def list_by_attempt(
        self,
        knowledge_id: str,
        attempt: int,
    ) -> list[KnowledgeProcessingSpan]: ...
    async def cancel_descendants(
        self,
        knowledge_id: str,
        attempt: int,
        parent_span_id: str,
        reason: str,
    ) -> int: ...
    async def cancel_all_open_spans(
        self,
        knowledge_id: str,
        attempt: int,
        error_code: str,
        reason: str,
        *,
        now: datetime,
    ) -> int: ...
    async def cancel_open_spans_by_name(
        self,
        knowledge_id: str,
        attempt: int,
        name: str,
        error_code: str,
        reason: str,
        *,
        now: datetime,
    ) -> int: ...


class Heartbeat(Protocol):
    """Optional side-channel advancing the parent row's ``updated_at``."""

    async def touch(self, *, knowledge_id: str) -> None: ...


class SpanTracker:
    """Per-attempt progress tree recorder + progress query.

    Best-effort by design: every tracking operation swallows store
    errors (logging them) so a tracker failure never breaks the parsing
    pipeline. The progress query is the one surface that classifies
    errors for API callers.
    """

    def __init__(
        self,
        *,
        span_store: SpanStore,
        heartbeat: Heartbeat | None = None,
    ) -> None:
        self._store = span_store
        self._heartbeat = heartbeat
        # In-process duration cache (span_id → started_at). Cross-process
        # workers won't find their parent's start here — duration falls
        # back to the persisted ``started_at`` when the cache misses.
        self._starts: dict[str, datetime] = {}
        self._starts_lock = threading.Lock()

    # ── Attempt lifecycle ───────────────────────────────────────────

    async def open_attempt(
        self,
        knowledge_id: str,
        langfuse_trace_id: str = "",
    ) -> tuple[Span | None, int]:
        """Create a new root span for the next attempt.

        Returns ``(root_span, attempt)``; ``root_span`` is ``None`` when
        the store write failed (the attempt number is still reported so
        the caller can record partial progress). Call at the start of a
        parse / reparse, before any other begin operation.
        """
        if knowledge_id == "":
            return None, 0
        attempt = await self._store.next_attempt(knowledge_id)
        now = datetime.now(UTC)
        root_id = new_span_id()
        meta: JsonObject = {}
        if langfuse_trace_id:
            meta["langfuse_trace_id"] = langfuse_trace_id
        row = self._build_row(
            knowledge_id=knowledge_id,
            attempt=attempt,
            span_id=root_id,
            parent_span_id=None,
            name=ROOT_SPAN_NAME,
            kind=SPAN_KIND_ROOT,
            status=SPAN_STATUS_RUNNING,
            metadata=meta,
            started_at=now,
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] open_attempt failed kid={}:", knowledge_id, exc_info=True
            )
            return None, attempt
        self._record_start(root_id, now)
        await self._touch_knowledge_heartbeat(knowledge_id, SPAN_KIND_ROOT)
        return (
            Span(
                knowledge_id=knowledge_id,
                attempt=attempt,
                span_id=root_id,
                parent_span_id=None,
                name=ROOT_SPAN_NAME,
                kind=SPAN_KIND_ROOT,
                status=SPAN_STATUS_RUNNING,
                started_at=now,
            ),
            attempt,
        )

    async def finalize_attempt(
        self,
        *,
        knowledge_id: str,
        attempt: int,
        status: str = "",
        output: JsonObject | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        """Close the root span for (knowledge, attempt) terminally.

        Idempotent: re-closing an already-terminal root is a no-op so
        callers from multiple paths (success orchestrator, dead-letter
        handler, housekeeping) can fire without coordination. ``status``
        defaults to ``done``; output/error are written verbatim.
        """
        await self._finalize_attempt(
            knowledge_id,
            attempt,
            status,
            output,
            error_code,
            error_message,
        )

    async def _finalize_attempt(
        self,
        knowledge_id: str,
        attempt: int,
        status: str,
        output: JsonObject | None,
        error_code: str,
        error_message: str,
    ) -> None:
        if knowledge_id == "" or attempt <= 0:
            return
        if not status:
            status = SPAN_STATUS_DONE
        if status not in _TERMINAL_STATUSES:
            raise ValidationError(
                code="span.invalid_status",
                message=f"invalid terminal status: {status}",
            )
        try:
            rows = await self._store.list_by_attempt(knowledge_id, attempt)
        except Exception:
            logger.warning(
                "[SpanTracker] finalize_attempt list failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return
        root = next((r for r in rows if r.kind == SPAN_KIND_ROOT), None)
        if root is None:
            # No root means nothing to close — an attempt that predates
            # the tracker or whose open write failed.
            return
        if root.status in _TERMINAL_STATUSES:
            return
        now = datetime.now(UTC)
        duration_ms = 0
        if root.started_at is not None:
            duration_ms = int((now - root.started_at).total_seconds() * 1000)
        row = self._build_row(
            knowledge_id=root.knowledge_id,
            attempt=root.attempt,
            span_id=root.span_id,
            parent_span_id=root.parent_span_id,
            name=root.name,
            kind=root.kind,
            status=status,
            input=root.input,
            output=output,
            metadata=root.metadata,
            error_code=(error_code or "").strip(),
            error_message=_truncate(error_message, MAX_ERROR_MESSAGE_LEN),
            started_at=root.started_at,
            finished_at=now,
            duration_ms=duration_ms,
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] finalize_attempt upsert failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return
        await self._touch_knowledge_heartbeat(knowledge_id, SPAN_KIND_ROOT)

    async def abort_attempt(
        self,
        *,
        knowledge_id: str,
        attempt: int,
        error_code: str = "",
        error_message: str = "",
        reason: str = "",
    ) -> None:
        """Cascade-cancel every open span of the attempt, then close root.

        The user-initiated cancel counterpart to ``finalize_attempt``.
        The flat sweep ignores the tree shape on purpose: fan-out stages
        end their own row the moment they finish dispatching async work,
        so their children can still be ``running`` under a terminal
        parent — a tree walk that stops at terminal parents would orphan
        those leaves. Idempotent.
        """
        if knowledge_id == "" or attempt <= 0:
            return
        reason = reason or "user cancelled"
        error_code = error_code or "USER_CANCELLED"
        try:
            swept = await self._store.cancel_all_open_spans(
                knowledge_id,
                attempt,
                error_code,
                reason,
                now=datetime.now(UTC),
            )
            if swept > 0:
                logger.info(
                    "[SpanTracker] abort_attempt swept {} open span(s) for kid={} attempt={}",
                    swept,
                    knowledge_id,
                    attempt,
                )
        except Exception:
            # Fall through to closing the root anyway — closing the root
            # is more important than perfectly closing every child.
            logger.warning(
                "[SpanTracker] abort_attempt sweep failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
        await self._finalize_attempt(
            knowledge_id,
            attempt,
            SPAN_STATUS_CANCELLED,
            None,
            error_code,
            error_message,
        )

    # ── Stage / subspan lifecycle ───────────────────────────────────

    async def begin_stage(
        self,
        *,
        knowledge_id: str,
        attempt: int,
        stage: str,
        input: JsonObject | None = None,
    ) -> Span | None:
        """Start one of the canonical stages, returning its handle.

        Looks up the root span for (knowledge, attempt) as the stage's
        parent. Re-entry (retry, double-call from adjacent code paths)
        reuses the existing row so the timeline never shows two segments
        for the same stage. Returns ``None`` when the store write fails.
        """
        if knowledge_id == "" or stage == "":
            return None
        try:
            rows = await self._store.list_by_attempt(knowledge_id, attempt)
        except Exception:
            logger.warning(
                "[SpanTracker] begin_stage list failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return None
        root_id = ""
        existing: KnowledgeProcessingSpan | None = None
        for r in rows:
            if r.kind == SPAN_KIND_ROOT and root_id == "":
                root_id = r.span_id
            if r.kind == SPAN_KIND_STAGE and r.name == stage:
                existing = r
        if root_id == "":
            # Pipeline started before the tracker was wired. Synthesize
            # a rootless stage so we still record something.
            logger.warning(
                "[SpanTracker] begin_stage: no root for kid={} attempt={}, recording rootless",
                knowledge_id,
                attempt,
            )
        now = datetime.now(UTC)
        if existing is not None:
            # Re-entry path: keep the original span_id so any subspan
            # already referencing it stays attached. Reset to running
            # and refresh started_at; clear terminal-only fields so the
            # row reads cleanly as "running again".
            row = self._build_row(
                knowledge_id=existing.knowledge_id,
                attempt=existing.attempt,
                span_id=existing.span_id,
                parent_span_id=existing.parent_span_id,
                name=existing.name,
                kind=existing.kind,
                status=SPAN_STATUS_RUNNING,
                input=input,
                started_at=now,
                duration_ms=0,
                now=now,
            )
            try:
                await self._store.upsert(row)
            except Exception:
                logger.warning(
                    "[SpanTracker] begin_stage re-enter failed kid={} stage={}:",
                    knowledge_id,
                    stage,
                    exc_info=True,
                )
                return None
            self._record_start(existing.span_id, now)
            await self._touch_knowledge_heartbeat(knowledge_id, SPAN_KIND_STAGE)
            return Span(
                knowledge_id=existing.knowledge_id,
                attempt=existing.attempt,
                span_id=existing.span_id,
                parent_span_id=existing.parent_span_id,
                name=existing.name,
                kind=existing.kind,
                status=SPAN_STATUS_RUNNING,
                started_at=now,
            )
        span_id = new_span_id()
        row = self._build_row(
            knowledge_id=knowledge_id,
            attempt=attempt,
            span_id=span_id,
            parent_span_id=root_id,
            name=stage,
            kind=SPAN_KIND_STAGE,
            status=SPAN_STATUS_RUNNING,
            input=input,
            started_at=now,
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] begin_stage failed kid={} stage={}:",
                knowledge_id,
                stage,
                exc_info=True,
            )
            return None
        self._record_start(span_id, now)
        await self._touch_knowledge_heartbeat(knowledge_id, SPAN_KIND_STAGE)
        return Span(
            knowledge_id=knowledge_id,
            attempt=attempt,
            span_id=span_id,
            parent_span_id=root_id,
            name=stage,
            kind=SPAN_KIND_STAGE,
            status=SPAN_STATUS_RUNNING,
            started_at=now,
        )

    async def begin_sub_span(
        self,
        *,
        parent: Span | None,
        name: str,
        kind: str = SPAN_KIND_SUBSPAN,
        input: JsonObject | None = None,
    ) -> Span | None:
        """Create a child span under ``parent`` (stage or subspan).

        ``kind`` is ``subspan`` or ``generation``; generations are stitched
        to an external trace by ``metadata.langfuse_trace_id``. A retry /
        restart that re-runs the same subtask cancels the previous
        same-name open row first so the UI shows one logical subspan per
        (attempt, name) instead of duplicate stripes.
        """
        if parent is None or name == "":
            return None
        name = fit_span_name(name)
        if kind not in (SPAN_KIND_GENERATION, SPAN_KIND_SUBSPAN):
            kind = SPAN_KIND_SUBSPAN
        try:
            await self._store.cancel_open_spans_by_name(
                parent.knowledge_id,
                parent.attempt,
                name,
                "TASK_SUPERSEDED",
                "superseded by a new run of the same subtask",
                now=datetime.now(UTC),
            )
        except Exception:
            logger.warning(
                "[SpanTracker] supersede {} before begin_sub_span failed:",
                name,
                exc_info=True,
            )
        now = datetime.now(UTC)
        span_id = new_span_id()
        row = self._build_row(
            knowledge_id=parent.knowledge_id,
            attempt=parent.attempt,
            span_id=span_id,
            parent_span_id=parent.span_id,
            name=name,
            kind=kind,
            status=SPAN_STATUS_RUNNING,
            input=input,
            started_at=now,
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] begin_sub_span failed parent={} name={}:",
                parent.span_id,
                name,
                exc_info=True,
            )
            return None
        self._record_start(span_id, now)
        await self._touch_knowledge_heartbeat(parent.knowledge_id, kind)
        return Span(
            knowledge_id=parent.knowledge_id,
            attempt=parent.attempt,
            span_id=span_id,
            parent_span_id=parent.span_id,
            name=name,
            kind=kind,
            status=SPAN_STATUS_RUNNING,
            started_at=now,
        )

    # ── Span transitions ────────────────────────────────────────────

    async def end_span(
        self,
        *,
        span: Span | None,
        output: JsonObject | None = None,
    ) -> None:
        """Mark ``span`` done with optional output. Safe with None."""
        if span is None:
            return
        now = datetime.now(UTC)
        row = self._build_row(
            knowledge_id=span.knowledge_id,
            attempt=span.attempt,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            kind=span.kind,
            status=SPAN_STATUS_DONE,
            output=output,
            started_at=span.started_at,
            finished_at=now,
            duration_ms=self._duration_since(span, now),
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] end_span failed span={}:", span.span_id, exc_info=True
            )
        await self._touch_knowledge_heartbeat(span.knowledge_id, span.kind)

    async def fail_span(
        self,
        *,
        span: Span | None,
        error_code: str = "",
        error_message: str = "",
        error_detail: str | None = None,
    ) -> None:
        """Mark ``span`` failed and cascade-cancel its descendants.

        ``error_detail`` is recorded verbatim (truncated) for admin
        views. A failure in a main pipeline stage also closes the root
        as failed so the trace never shows "running" forever.
        """
        if span is None:
            return
        now = datetime.now(UTC)
        row = self._build_row(
            knowledge_id=span.knowledge_id,
            attempt=span.attempt,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            kind=span.kind,
            status=SPAN_STATUS_FAILED,
            error_code=(error_code or "").strip(),
            error_message=_truncate(error_message, MAX_ERROR_MESSAGE_LEN),
            error_detail=_truncate(error_detail or "", MAX_ERROR_DETAIL_LEN),
            started_at=span.started_at,
            finished_at=now,
            duration_ms=self._duration_since(span, now),
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] fail_span failed span={}:", span.span_id, exc_info=True
            )
        # Cascade: anything downstream of this span gets cancelled. The
        # reason string is what the UI surfaces under each cancelled
        # child's tooltip — keep it short and human.
        reason = f"upstream {span.name} failed"
        if error_code:
            reason = f"{reason} ({error_code})"
        try:
            await self._store.cancel_descendants(
                span.knowledge_id,
                span.attempt,
                span.span_id,
                reason,
            )
        except Exception:
            logger.warning(
                "[SpanTracker] cancel descendants failed span={}:",
                span.span_id,
                exc_info=True,
            )
        if span.kind == SPAN_KIND_STAGE:
            await self._cascade_dependent_stages(span, reason)
            if is_main_pipeline_stage(span.name):
                await self._finalize_attempt(
                    span.knowledge_id,
                    span.attempt,
                    SPAN_STATUS_FAILED,
                    None,
                    error_code,
                    error_message,
                )
        await self._touch_knowledge_heartbeat(span.knowledge_id, span.kind)

    async def skip_span(
        self,
        *,
        span: Span | None,
        reason: str = "",
    ) -> None:
        """Mark ``span`` intentionally not run (e.g. multimodal on a
        text-only document). Distinct from cancelled — skipped is "we
        chose not to" while cancelled is "an upstream broke"."""
        if span is None:
            return
        now = datetime.now(UTC)
        row = self._build_row(
            knowledge_id=span.knowledge_id,
            attempt=span.attempt,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id,
            name=span.name,
            kind=span.kind,
            status=SPAN_STATUS_SKIPPED,
            error_message=reason,
            started_at=span.started_at,
            finished_at=now,
            duration_ms=0,
            now=now,
        )
        try:
            await self._store.upsert(row)
        except Exception:
            logger.warning(
                "[SpanTracker] skip_span failed span={}:", span.span_id, exc_info=True
            )
        await self._touch_knowledge_heartbeat(span.knowledge_id, span.kind)

    # ── Lookups / progress query ────────────────────────────────────

    async def latest_attempt(self, knowledge_id: str) -> int:
        """Return the highest recorded attempt, or 0 when never parsed."""
        try:
            return await self._store.latest_attempt(knowledge_id)
        except Exception:
            logger.warning(
                "[SpanTracker] latest_attempt failed kid={}:",
                knowledge_id,
                exc_info=True,
            )
            return 0

    async def list_attempt_spans(
        self,
        knowledge_id: str,
        attempt: int,
    ) -> tuple[KnowledgeProcessingSpan, ...]:
        """Return the attempt's persisted span rows in insertion order."""
        try:
            rows = await self._store.list_by_attempt(knowledge_id, attempt)
        except Exception:
            logger.warning(
                "[SpanTracker] list_attempt_spans failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return ()
        return tuple(rows)

    async def lookup_stage(
        self,
        *,
        knowledge_id: str,
        attempt: int,
        stage: str,
    ) -> Span | None:
        """Return the stage's handle for an in-flight attempt.

        The cross-process bridge that lets a worker attach subspans to
        the parent stage span created upstream.
        """
        try:
            rows = await self._store.list_by_attempt(knowledge_id, attempt)
        except Exception:
            logger.warning(
                "[SpanTracker] lookup_stage list failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return None
        for r in rows:
            if r.kind == SPAN_KIND_STAGE and r.name == stage:
                return self._to_span(r)
        return None

    async def lookup_span_by_name(
        self,
        *,
        knowledge_id: str,
        attempt: int,
        name: str,
    ) -> Span | None:
        """Return the first span of any kind matching ``name``.

        Lets a fan-out worker attach its subspan under a grouping span
        created earlier by the orchestrator. Returns ``None`` when no
        such span exists (caller should fall back to the stage).
        """
        if name == "" or knowledge_id == "" or attempt <= 0:
            return None
        name = fit_span_name(name)
        try:
            rows = await self._store.list_by_attempt(knowledge_id, attempt)
        except Exception:
            logger.warning(
                "[SpanTracker] lookup_span_by_name list failed kid={} attempt={}:",
                knowledge_id,
                attempt,
                exc_info=True,
            )
            return None
        for r in rows:
            if r.name == name:
                return self._to_span(r)
        return None

    async def get_progress(
        self,
        *,
        knowledge_id: str,
        attempt: int | None = None,
    ) -> SpanProgress:
        """Return the progress projection for the requested attempt.

        ``attempt`` defaults to the latest recorded attempt. Raises
        ``ValidationError`` on a blank knowledge id or a non-positive
        explicit attempt, and ``NotFoundError`` when the knowledge has
        never been tracked. A requested attempt with no rows returns an
        empty span list (the caller synthesizes pending placeholders).
        """
        if not knowledge_id.strip():
            raise ValidationError(
                code="span.knowledge_required",
                message="knowledge ID is required",
            )
        if attempt is not None and attempt < 1:
            raise ValidationError(
                code="span.invalid_attempt",
                message="attempt must be >= 1",
            )
        latest = await self.latest_attempt(knowledge_id)
        current = attempt if attempt is not None else latest
        if current <= 0:
            raise NotFoundError(
                code="span.not_found",
                message="no processing attempt recorded for this knowledge",
            )
        spans = await self.list_attempt_spans(knowledge_id, current)
        return SpanProgress(
            knowledge_id=knowledge_id,
            attempt=current,
            latest_attempt=latest,
            spans=spans,
        )

    # ── Internals ───────────────────────────────────────────────────

    async def _cascade_dependent_stages(self, failed_stage: Span, reason: str) -> None:
        """Flip downstream STAGE rows to ``cancelled`` via the stage DAG.

        A chunking failure leaves embedding / multimodal / postprocess
        as "pending" forever without this. Flipping a dependent stage
        also cascade-cancels any subspan already attached to it so no
        orphan spinners survive under a cancelled parent.
        """
        try:
            rows = await self._store.list_by_attempt(
                failed_stage.knowledge_id,
                failed_stage.attempt,
            )
        except Exception:
            return
        dependents = stages_depending_on(failed_stage.name)
        if not dependents:
            return
        now = datetime.now(UTC)
        for row in rows:
            if row.kind != SPAN_KIND_STAGE:
                continue
            if row.status not in (SPAN_STATUS_PENDING, SPAN_STATUS_RUNNING):
                continue
            if row.name not in dependents:
                continue
            updated = self._build_row(
                knowledge_id=row.knowledge_id,
                attempt=row.attempt,
                span_id=row.span_id,
                parent_span_id=row.parent_span_id,
                name=row.name,
                kind=row.kind,
                status=SPAN_STATUS_CANCELLED,
                error_code="UPSTREAM_FAILED",
                error_message=reason,
                started_at=row.started_at,
                finished_at=now,
                duration_ms=row.duration_ms,
                now=now,
            )
            try:
                await self._store.upsert(updated)
            except Exception:
                logger.warning(
                    "[SpanTracker] cascade dependent stage {}:", row.name, exc_info=True
                )
                continue
            try:
                await self._store.cancel_descendants(
                    row.knowledge_id,
                    row.attempt,
                    row.span_id,
                    reason,
                )
            except Exception:
                logger.warning(
                    "[SpanTracker] cascade descendants of dependent {}:",
                    row.name,
                    exc_info=True,
                )

    async def _touch_knowledge_heartbeat(
        self,
        knowledge_id: str,
        kind: str,
    ) -> None:
        """Advance the parent row's ``updated_at`` via the injected seam.

        Called on root / stage transitions only — subspan and generation
        transitions skip the side-channel because the spans table itself
        is already observable per attempt, and fan-out workloads would
        otherwise hammer the same hot parent row. Best-effort.
        """
        if self._heartbeat is None or knowledge_id == "":
            return
        if kind not in (SPAN_KIND_ROOT, SPAN_KIND_STAGE):
            return
        try:
            await self._heartbeat.touch(knowledge_id=knowledge_id)
        except Exception:
            logger.warning(
                "[SpanTracker] heartbeat failed kid={}:", knowledge_id, exc_info=True
            )

    def _record_start(self, span_id: str, at: datetime) -> None:
        with self._starts_lock:
            self._starts[span_id] = at

    def _take_start(self, span_id: str) -> datetime | None:
        with self._starts_lock:
            return self._starts.pop(span_id, None)

    def _duration_since(self, span: Span, now: datetime) -> int:
        """Elapsed milliseconds, preferring the in-process cache."""
        start = self._take_start(span.span_id)
        if start is not None:
            return int((now - start).total_seconds() * 1000)
        if span.started_at == _EPOCH:
            return 0
        return int((now - span.started_at).total_seconds() * 1000)

    @staticmethod
    def _to_span(row: KnowledgeProcessingSpan) -> Span:
        return Span(
            knowledge_id=row.knowledge_id,
            attempt=row.attempt,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            name=row.name,
            kind=row.kind,
            status=row.status,
            started_at=row.started_at if row.started_at is not None else _EPOCH,
        )

    @staticmethod
    def _build_row(
        *,
        knowledge_id: str,
        attempt: int,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        kind: str,
        status: str,
        input: JsonObject | None = None,
        output: JsonObject | None = None,
        metadata: JsonObject | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_detail: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        now: datetime,
    ) -> KnowledgeProcessingSpan:
        """Build a transition row with service-stamped bookkeeping.

        Error fields default to the empty string (never ``None``) so the
        store's conflict update — which always writes them — clears any
        prior error instead of preserving stale data.
        """
        return KnowledgeProcessingSpan(
            knowledge_id=knowledge_id,
            attempt=attempt,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            status=status,
            input=input,
            output=output,
            metadata=metadata,
            error_code=error_code or "",
            error_message=error_message or "",
            error_detail=error_detail or "",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            created_at=now,
            updated_at=now,
        )


def _truncate(value: str, limit: int) -> str:
    """Truncate ``value`` to ``limit`` characters."""
    return value[:limit]


__all__ = [
    "ALL_STAGES",
    "MAX_ERROR_DETAIL_LEN",
    "MAX_ERROR_MESSAGE_LEN",
    "MAX_SPAN_NAME_LEN",
    "ROOT_SPAN_NAME",
    "SPAN_KIND_GENERATION",
    "SPAN_KIND_ROOT",
    "SPAN_KIND_STAGE",
    "SPAN_KIND_SUBSPAN",
    "SPAN_STATUS_CANCELLED",
    "SPAN_STATUS_DONE",
    "SPAN_STATUS_FAILED",
    "SPAN_STATUS_PENDING",
    "SPAN_STATUS_RUNNING",
    "SPAN_STATUS_SKIPPED",
    "STAGE_CHUNKING",
    "STAGE_DEPENDENCIES",
    "STAGE_DOCREADER",
    "STAGE_EMBEDDING",
    "STAGE_MULTIMODAL",
    "STAGE_POSTPROCESS",
    "Heartbeat",
    "Span",
    "SpanProgress",
    "SpanStore",
    "SpanTracker",
    "fit_span_name",
    "is_main_pipeline_stage",
    "new_span_id",
    "stages_depending_on",
]
