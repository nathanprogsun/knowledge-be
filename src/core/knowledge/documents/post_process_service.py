"""Knowledge post-process orchestration — enrichment fan-out after parse.

Runs once a document has been parsed and split into chunks (including
multimodal OCR/Caption) and fans the enrichment subtasks out:

- closes the multimodal stage span the parent parse left ``running``;
- opens the ``postprocess`` stage span for the attempt;
- reads the knowledge / knowledge-base and decides which subtasks to
  spawn from the merged process config — summary generation, question
  generation (one task per window of text chunks), chunk-extract (graph
  RAG) and wiki ingest;
- flips the row ``processing`` → ``finalizing`` with the pending-subtask
  counter seeded **before** spawning anything, so a parallel cancel /
  delete cannot race the row into ``completed``;
- reconciles the seeded counter against what was actually enqueued,
  releasing shortfall slots so a half-fanned-out batch cannot strand the
  row in ``finalizing``;
- records the outcome on the post-process span and finalizes the attempt.

The service owns the full decision tree but no storage / queue access
directly: every external dependency arrives through an injected seam,
following this module's dependency-injection convention. The worker
wiring layer composes the seams later; a service constructed without its
core seams refuses to run rather than guessing.

This module lives in the core layer and is imported by the worker task —
the dependency points workers → core, never the other way around.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.app_logging import logger
from src.common.exception import ApplicationError
from src.common.json import BindParams, JsonObject, JsonValue
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.process_document import kb_needs_embedding
from src.core.knowledge.documents.span_tracker import (
    SPAN_KIND_STAGE,
    SPAN_KIND_SUBSPAN,
    SPAN_STATUS_RUNNING,
    STAGE_MULTIMODAL,
    STAGE_POSTPROCESS,
    Span,
    SpanTracker,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_NONE,
    SUMMARY_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

# ── Task names (mirror the upstream task-type constants) ──────────────

TASK_SUMMARY_GENERATION = "summary:generation"
TASK_QUESTION_GENERATION = "question:generation"
TASK_CHUNK_EXTRACT = "chunk:extract"
TASK_WIKI_INGEST = "wiki:ingest"

# Question-generation fan-out window: one task per this many text chunks
# so a huge document does not spawn thousands of tiny tasks.
QUESTION_GEN_CHUNK_BATCH_SIZE = 20

# Grouping span the per-batch question subspans nest under, so the trace
# viewer shows one "postprocess.question" node instead of dozens of
# siblings directly beneath the postprocess stage.
POSTPROCESS_QUESTION_GROUP_SPAN = "postprocess.question"

# Question-count default and hard cap, mirroring the upstream clamp.
DEFAULT_QUESTION_COUNT = 3
MAX_QUESTION_COUNT = 10

# Chunk types that carry text into the enrichment stages (a scanned-PDF
# pipeline adds OCR / Caption rows that are also text-like).
_TEXT_LIKE_CHUNK_TYPES: frozenset[str] = frozenset(
    {CHUNK_TYPE_TEXT, CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION}
)


class PostProcessError(ApplicationError):
    """A retryable failure in the post-process orchestration."""

    code = "postprocess.failed"
    message = "knowledge post-process orchestration failed"


# ── Domain value types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgePostProcessPayload:
    """Wire contract of the ``knowledge:post_process`` task."""

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    language: str = ""
    attempt: int = 0


@dataclass(frozen=True)
class PostProcessConfig:
    """Effective enrichment configuration after the KB/override merge."""

    question_generation_enabled: bool
    question_count: int
    graph_enabled: bool


@dataclass(frozen=True)
class PostProcessOutcome:
    """Result of one post-process orchestration run.

    ``skipped`` reports an intentional no-op (aborted knowledge, already
    terminal, or the enrichment fan-out being declined); ``reason`` names
    the skip cause. The enqueued fields mirror the upstream JSON-map
    output so the worker can return them verbatim.
    """

    skipped: bool = False
    reason: str = ""
    chunks_total: int = 0
    enqueued_summary: bool = False
    enqueued_question: bool = False
    enqueued_question_count: int = 0
    enqueued_wiki: bool = False
    wiki_slot_owned: bool = False
    enqueued_graph: bool = False
    enqueued_graph_count: int = 0


# ── Injectable seams ───────────────────────────────────────────────────


@runtime_checkable
class EnrichmentEnqueuer(Protocol):
    """Fan-out seam: places one enrichment subtask on the queue by name.

    The worker wiring layer maps each name onto the ARQ job with the
    matching retry / timeout policy. Returns ``True`` only when the task
    was actually accepted, so the caller can release the seeded
    pending-subtask slot on a skip or failure.
    """

    async def enqueue(self, *, name: str, payload: Mapping[str, JsonValue]) -> bool:
        """Place ``payload`` for the task ``name``; return acceptance."""
        ...


@runtime_checkable
class WikiDispatcher(Protocol):
    """Schedules the KB-scoped wiki ingest batch trigger.

    The durable per-knowledge pending op itself is committed atomically
    by the finalizing handoff (:meth:`FinalizingCoordinator.seed_finalizing_with_wiki`);
    this seam only wakes the batch that drains the queue, so re-invoking
    it on a retry never duplicates durable work.
    """

    async def dispatch_ingest(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        language: str = "",
    ) -> bool:
        """Schedule the ingest trigger; return whether it was accepted."""
        ...


@runtime_checkable
class FinalizingCoordinator(Protocol):
    """Seeds and drains the per-knowledge enrichment subtask counter.

    Each subtask handler atomically decrements the counter on terminal
    exit; the row promotes itself to ``completed`` when it hits zero.
    """

    async def set_finalizing(self, *, knowledge_id: str, expected_subtasks: int) -> bool:
        """Flip ``processing`` → ``finalizing`` with the counter seeded.

        Returns whether the row was actually promoted (a cancel / delete
        racing the caller returns ``False``).
        """
        ...

    async def seed_finalizing_with_wiki(self, *, knowledge_id: str, expected_subtasks: int) -> bool:
        """Promote to ``finalizing`` and persist the pending wiki ingest op
        in the same transaction; return whether the row was promoted."""
        ...

    async def finalize_subtask(self, *, knowledge_id: str) -> bool:
        """Decrement the counter, promoting the row at zero."""
        ...


# ── Effective config resolution ────────────────────────────────────────


def _process_overrides_of(row: Document | None) -> JsonObject | None:
    """Read the per-upload process overrides from a document's metadata."""
    metadata = row.metadata if row is not None else None
    if not isinstance(metadata, dict):
        return None
    overrides = metadata.get("process_overrides")
    return overrides if isinstance(overrides, dict) else None


def _wiki_enabled(indexing_strategy: JsonObject | None) -> bool:
    """Return whether the KB's indexing strategy turns wiki on.

    An absent strategy resolves to the default (wiki disabled).
    """
    strategy = indexing_strategy if isinstance(indexing_strategy, dict) else {}
    return bool(strategy.get("wiki_enabled"))


def resolve_post_process_config(
    kb: KnowledgeBaseInfo,
    process_overrides: JsonObject | None,
) -> PostProcessConfig:
    """Merge KB defaults with per-upload overrides for the enrichment run.

    Question generation reads the override config first and falls back to
    the knowledge-base config; a non-positive count defaults to 3 and is
    clamped at 10. Graph extraction is enabled only when both the
    indexing-strategy flag and the extract config are on, mirroring the
    upstream ``IsGraphEnabled`` semantics (a per-upload override still
    needs the extract config enabled).
    """
    base = kb.question_generation_config if isinstance(kb.question_generation_config, dict) else {}
    question_enabled = bool(base.get("enabled"))
    raw_count = base.get("question_count")
    question_count = int(raw_count) if isinstance(raw_count, int) and raw_count > 0 else 0

    overrides = process_overrides if isinstance(process_overrides, dict) else {}
    override_qg = overrides.get("question_generation_config")
    if isinstance(override_qg, dict):
        override_enabled = override_qg.get("enabled")
        if isinstance(override_enabled, bool):
            question_enabled = override_enabled
        raw_count = override_qg.get("question_count")
        if isinstance(raw_count, int) and raw_count > 0:
            question_count = int(raw_count)

    if question_count <= 0:
        question_count = DEFAULT_QUESTION_COUNT
    if question_count > MAX_QUESTION_COUNT:
        question_count = MAX_QUESTION_COUNT

    strategy = kb.indexing_strategy if isinstance(kb.indexing_strategy, dict) else {}
    extract = kb.extract_config if isinstance(kb.extract_config, dict) else {}
    extract_enabled = bool(extract.get("enabled"))
    graph_enabled = bool(strategy.get("graph_enabled")) and extract_enabled
    override_graph = overrides.get("graph_enabled")
    if isinstance(override_graph, bool):
        graph_enabled = override_graph and extract_enabled

    return PostProcessConfig(
        question_generation_enabled=question_enabled,
        question_count=question_count,
        graph_enabled=graph_enabled,
    )


def _batch_count(total: int) -> int:
    """Number of question-generation batches for ``total`` text chunks."""
    if total <= 0:
        return 0
    return (total + QUESTION_GEN_CHUNK_BATCH_SIZE - 1) // QUESTION_GEN_CHUNK_BATCH_SIZE


# ── Orchestrator ───────────────────────────────────────────────────────


class PostProcessService:
    """Orchestrates the post-process enrichment fan-out for one knowledge.

    Stateless over injected seams; the worker wiring layer composes the
    repositories / tracker / enqueuers before any real run. ``run``
    raises :class:`PostProcessError` for retryable failures (the worker
    layer lets ARQ retry them) and returns :class:`PostProcessOutcome`
    for skip / success paths.
    """

    def __init__(
        self,
        *,
        knowledge_repo: KnowledgeRepository | None = None,
        kb_service: KBService | None = None,
        chunk_repo: ChunkRepository | None = None,
        enqueuer: EnrichmentEnqueuer | None = None,
        wiki_dispatcher: WikiDispatcher | None = None,
        tracker: SpanTracker | None = None,
        finalizer: FinalizingCoordinator | None = None,
    ) -> None:
        self._knowledge_repo = knowledge_repo
        self._kb_service = kb_service
        self._chunk_repo = chunk_repo
        self._enqueuer = enqueuer
        self._wiki_dispatcher = wiki_dispatcher
        self._tracker = tracker
        self._finalizer = finalizer

    # ── Entry ──────────────────────────────────────────────────────

    async def run(self, *, payload: KnowledgePostProcessPayload) -> PostProcessOutcome:
        """Run the post-process fan-out for ``payload``.

        Returns an outcome for the skip / success paths. Raises
        :class:`PostProcessError` when a retryable step fails (a lost
        wiki trigger, a rejected finalizing handoff, or an unwired seam).
        """
        self._require_core_seams()
        # Narrow the optional seams for the rest of the run; the guard
        # above already raised for any that are absent.
        assert self._knowledge_repo is not None
        assert self._kb_service is not None
        assert self._chunk_repo is not None
        assert self._tracker is not None

        attempt = payload.attempt
        if attempt <= 0:
            latest = await self._tracker.latest_attempt(payload.knowledge_id)
            if latest > 0:
                attempt = latest

        # Close the multimodal stage span (the parent enqueued it as
        # "running" and we never see the per-image fan-in here other than
        # by reaching post-process). If the parent skipped multimodal
        # entirely, the stage row is already "skipped" and must stay so.
        await self._finish_running_multimodal_stage(payload.knowledge_id, attempt)

        post_span = await self._begin_postprocess_stage(payload.knowledge_id, attempt)

        knowledge = await self._knowledge_repo.get_by_id_only(payload.knowledge_id)
        if knowledge is None:
            logger.warning(
                "knowledge_post_process: knowledge {} not found, aborting.",
                payload.knowledge_id,
            )
            return PostProcessOutcome(skipped=True, reason="knowledge_not_found")

        # Skip post-processing entirely when the knowledge has been
        # cancelled or marked for deletion. We must NOT enqueue summary /
        # question / graph / wiki child tasks for an aborted knowledge and
        # MUST close post_span, otherwise the trace shows a running bar
        # long after the user cancelled.
        if knowledge.parse_status in (PARSE_STATUS_CANCELLED, PARSE_STATUS_DELETING):
            logger.info(
                "knowledge_post_process: knowledge {} aborted ({}), skipping post-processing.",
                payload.knowledge_id,
                knowledge.parse_status,
            )
            await self._skip_span(
                post_span,
                f"knowledge {knowledge.parse_status} before postprocess started",
            )
            return PostProcessOutcome(skipped=True, reason=knowledge.parse_status)

        kb = await self._kb_service.get_knowledge_base_by_id(
            knowledge_base_id=payload.knowledge_base_id
        )
        if kb is None:
            raise PostProcessError(
                code="postprocess.kb_not_found",
                message=f"knowledge base {payload.knowledge_base_id} not found",
            )

        config = resolve_post_process_config(kb, _process_overrides_of(knowledge))

        chunks = await self._chunk_repo.list_by_knowledge_id(
            tenant_id=payload.tenant_id,
            knowledge_id=payload.knowledge_id,
        )
        text_chunks = [c for c in chunks if c.chunk_type in _TEXT_LIKE_CHUNK_TYPES]

        will_spawn_summary = len(text_chunks) > 0
        will_spawn_question = (
            will_spawn_summary
            and kb_needs_embedding(kb.indexing_strategy)
            and config.question_generation_enabled
        )
        will_spawn_wiki = _wiki_enabled(kb.indexing_strategy) and len(text_chunks) > 0

        # Question generation fans out one task per window of plain text
        # chunks (OCR / Caption chunks were never fed to question
        # generation), sorted by ``start_at`` so the per-batch boundary
        # context matches the legacy reading order.
        question_chunks: list[Chunk] = []
        if will_spawn_question:
            question_chunks = sorted(
                (c for c in text_chunks if c.chunk_type == CHUNK_TYPE_TEXT),
                key=lambda c: c.start_at,
            )
        question_batch_count = _batch_count(len(question_chunks))

        graph_chunk_count = len(text_chunks) if config.graph_enabled else 0

        expected_subtasks = 0
        if will_spawn_summary:
            expected_subtasks += 1
        expected_subtasks += question_batch_count
        if will_spawn_wiki:
            expected_subtasks += 1
        expected_subtasks += graph_chunk_count

        # ── Processing → finalizing handoff ───────────────────────
        entered_finalizing = False
        wiki_slot_owned = False

        if knowledge.parse_status == PARSE_STATUS_FINALIZING and will_spawn_wiki:
            # A previous delivery may have persisted the wiki op but failed
            # to schedule its batch trigger. Retry only the trigger:
            # appending a second pending op would duplicate durable work.
            if self._wiki_dispatcher is None:
                raise PostProcessError(
                    code="postprocess.seams_unwired",
                    message="wiki dispatcher is not wired",
                )
            dispatched = await self._wiki_dispatcher.dispatch_ingest(
                tenant_id=payload.tenant_id,
                knowledge_base_id=payload.knowledge_base_id,
                knowledge_id=payload.knowledge_id,
                language=payload.language,
            )
            if not dispatched:
                await self._fail_span(
                    post_span,
                    "WIKI_TRIGGER_ENQUEUE_FAILED",
                    "wiki dispatcher rejected the ingest trigger",
                )
                raise PostProcessError(
                    code="postprocess.wiki_trigger_failed",
                    message="retry wiki ingest trigger was rejected",
                )
            logger.info(
                "knowledge_post_process: re-enqueued wiki ingest trigger for {}",
                payload.knowledge_id,
            )
            output: JsonObject = {"retried_wiki_trigger": True}
            await self._end_span(post_span, output)
            await self._finalize_attempt(payload.knowledge_id, attempt, output)
            return PostProcessOutcome(
                skipped=False,
                reason="",
                wiki_slot_owned=True,
                enqueued_wiki=True,
            )
        if knowledge.parse_status != PARSE_STATUS_PROCESSING:
            # The row was already in some other state (deleting / failed /
            # completed) when we arrived. Don't touch parse_status and
            # don't spawn enrichment.
            logger.info(
                "knowledge_post_process: knowledge {} is in {}, skipping enrichment fan-out.",
                payload.knowledge_id,
                knowledge.parse_status,
            )
            output = {
                "skipped": "non_processing_status",
                "observed_status": knowledge.parse_status,
            }
            await self._end_span(post_span, output)
            await self._finalize_attempt(payload.knowledge_id, attempt, output)
            return PostProcessOutcome(skipped=True, reason=knowledge.parse_status)
        if expected_subtasks == 0:
            # Nothing to enrich — fast path keeps the previous behaviour so
            # users without summary/question/graph see 'completed' immediately.
            updates: BindParams = {
                "parse_status": PARSE_STATUS_COMPLETED,
                "updated_at": datetime.now(UTC),
            }
            if len(text_chunks) > 0:
                updates["summary_status"] = SUMMARY_STATUS_NONE
            try:
                await self._knowledge_repo.update_columns(payload.knowledge_id, updates)
            except Exception as exc:
                logger.warning(
                    "knowledge_post_process: failed to mark {} completed (no subtasks): {}",
                    payload.knowledge_id,
                    exc,
                )
        else:
            # Flip processing to finalizing before fan-out so a parallel
            # cancel/delete cannot race us into completed.
            if self._finalizer is None:
                raise PostProcessError(
                    code="postprocess.seams_unwired",
                    message="finalizing coordinator is not wired",
                )
            if will_spawn_wiki:
                promoted = await self._finalizer.seed_finalizing_with_wiki(
                    knowledge_id=payload.knowledge_id,
                    expected_subtasks=expected_subtasks,
                )
                wiki_slot_owned = promoted
            else:
                promoted = await self._finalizer.set_finalizing(
                    knowledge_id=payload.knowledge_id,
                    expected_subtasks=expected_subtasks,
                )
            if promoted:
                entered_finalizing = True
                summary_status = (
                    SUMMARY_STATUS_PENDING if will_spawn_summary else SUMMARY_STATUS_NONE
                )
                try:
                    await self._knowledge_repo.update_columns(
                        payload.knowledge_id,
                        {
                            "summary_status": summary_status,
                            "updated_at": datetime.now(UTC),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "knowledge_post_process: failed to update summary_status for {}: {}",
                        payload.knowledge_id,
                        exc,
                    )
                logger.info(
                    "knowledge_post_process: knowledge {} entered finalizing (pending_subtasks={}).",
                    payload.knowledge_id,
                    expected_subtasks,
                )
            else:
                # The row was no longer 'processing' (cancel / delete won
                # the race). Skip enrichment entirely.
                logger.info(
                    "knowledge_post_process: knowledge {} no longer in processing, skipping enrichment fan-out.",
                    payload.knowledge_id,
                )
                output = {"skipped": "knowledge_no_longer_processing"}
                await self._end_span(post_span, output)
                await self._finalize_attempt(payload.knowledge_id, attempt, output)
                return PostProcessOutcome(skipped=True, reason="knowledge_no_longer_processing")

        # ── 4. Spawn summary and question tasks ────────────────────
        enqueued_summary = False
        enqueued_question_count = 0
        if will_spawn_summary:
            enqueued_summary = await self._enqueue_summary(payload, attempt)
            if not enqueued_summary:
                try:
                    await self._knowledge_repo.update_columns(
                        payload.knowledge_id,
                        {
                            "summary_status": SUMMARY_STATUS_FAILED,
                            "updated_at": datetime.now(UTC),
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "knowledge_post_process: failed to mark summary_status failed for {}: {}",
                        payload.knowledge_id,
                        exc,
                    )
            if will_spawn_question:
                # Create the question grouping span up front so the
                # per-batch subspans have a parent to nest under.
                group = await self._begin_question_group_span(
                    post_span, question_batch_count, len(question_chunks)
                )
                if group is not None:
                    await self._end_span(
                        group,
                        {
                            "batch_count": question_batch_count,
                            "chunk_count": len(question_chunks),
                        },
                    )
                enqueued_question_count = await self._enqueue_questions(
                    payload, config, attempt, question_chunks
                )

        # ── 5. Spawn graph (chunk-extract) tasks ───────────────────
        enqueued_graph_count = 0
        if graph_chunk_count > 0:
            for index, chunk in enumerate(text_chunks):
                enqueued = await self._enqueue_chunk_extract(
                    payload,
                    chunk,
                    kb.summary_model_id,
                    attempt,
                    index,
                )
                if enqueued:
                    enqueued_graph_count += 1

        # ── 6. Schedule the wiki trigger ───────────────────────────
        wiki_enqueue_error: str | None = None
        if will_spawn_wiki:
            dispatched = await self._dispatch_wiki_trigger(payload)
            if not dispatched:
                wiki_enqueue_error = "wiki dispatcher rejected the ingest trigger"
                logger.warning(
                    "knowledge_post_process: failed to enqueue wiki ingest for {}: {}",
                    payload.knowledge_id,
                    wiki_enqueue_error,
                )
            elif wiki_slot_owned:
                logger.info(
                    "knowledge_post_process: enqueued wiki ingest task for {}",
                    payload.knowledge_id,
                )

        # ── Reconcile the seeded counter ───────────────────────────
        if entered_finalizing:
            await self._release_shortfall(
                payload.knowledge_id,
                planned_owned=_planned_owned(
                    will_spawn_summary,
                    will_spawn_wiki,
                    question_batch_count,
                    graph_chunk_count,
                ),
                actual_owned=_actual_owned(
                    enqueued_summary,
                    wiki_slot_owned,
                    enqueued_question_count,
                    enqueued_graph_count,
                ),
            )

        output = {
            "chunks_total": len(text_chunks),
            "enqueued_summary": enqueued_summary,
            "enqueued_question": enqueued_question_count > 0,
            "enqueued_question_count": enqueued_question_count,
            "enqueued_wiki": wiki_slot_owned and wiki_enqueue_error is None,
            "wiki_slot_owned": wiki_slot_owned,
            "enqueued_graph": enqueued_graph_count > 0,
            "enqueued_graph_count": enqueued_graph_count,
        }
        await self._end_span(post_span, output)
        if wiki_slot_owned and wiki_enqueue_error is not None:
            raise PostProcessError(
                code="postprocess.wiki_trigger_failed",
                message=f"enqueue wiki ingest trigger: {wiki_enqueue_error}",
            )
        await self._finalize_attempt(payload.knowledge_id, attempt, output)
        return PostProcessOutcome(
            chunks_total=len(text_chunks),
            enqueued_summary=enqueued_summary,
            enqueued_question=enqueued_question_count > 0,
            enqueued_question_count=enqueued_question_count,
            enqueued_wiki=wiki_slot_owned and wiki_enqueue_error is None,
            wiki_slot_owned=wiki_slot_owned,
            enqueued_graph=enqueued_graph_count > 0,
            enqueued_graph_count=enqueued_graph_count,
        )

    # ── Subtask dispatch ──────────────────────────────────────────

    async def _enqueue_summary(self, payload: KnowledgePostProcessPayload, attempt: int) -> bool:
        """Enqueue the summary-generation task; true only when accepted."""
        if self._enqueuer is None:
            return False
        task_payload: JsonObject = {
            "tenant_id": payload.tenant_id,
            "knowledge_base_id": payload.knowledge_base_id,
            "knowledge_id": payload.knowledge_id,
            "language": payload.language,
            "attempt": attempt,
        }
        try:
            return await self._enqueuer.enqueue(name=TASK_SUMMARY_GENERATION, payload=task_payload)
        except Exception as exc:
            logger.warning(
                "knowledge_post_process: failed to enqueue summary generation for {}: {}",
                payload.knowledge_id,
                exc,
            )
            return False

    async def _enqueue_questions(
        self,
        payload: KnowledgePostProcessPayload,
        config: PostProcessConfig,
        attempt: int,
        question_chunks: list[Chunk],
    ) -> int:
        """Fan out one question-generation task per batch of text chunks.

        Returns the number of batch tasks actually enqueued; a failed
        enqueue is logged and skipped so the caller's reconciliation
        releases the unowned slot.
        """
        if self._enqueuer is None or not question_chunks or not config.question_generation_enabled:
            return 0
        question_count = config.question_count
        if question_count <= 0:
            question_count = DEFAULT_QUESTION_COUNT
        if question_count > MAX_QUESTION_COUNT:
            question_count = MAX_QUESTION_COUNT

        total = len(question_chunks)
        enqueued = 0
        batch_index = 0
        for start in range(0, total, QUESTION_GEN_CHUNK_BATCH_SIZE):
            end = min(start + QUESTION_GEN_CHUNK_BATCH_SIZE, total)
            batch = question_chunks[start:end]
            task_payload: JsonObject = {
                "tenant_id": payload.tenant_id,
                "knowledge_base_id": payload.knowledge_base_id,
                "knowledge_id": payload.knowledge_id,
                "question_count": question_count,
                "language": payload.language,
                "attempt": attempt,
                "chunk_ids": [c.id for c in batch],
                "batch_index": batch_index,
            }
            if start > 0:
                task_payload["prev_chunk_id"] = question_chunks[start - 1].id
            if end < total:
                task_payload["next_chunk_id"] = question_chunks[end].id
            batch_index += 1
            try:
                accepted = await self._enqueuer.enqueue(
                    name=TASK_QUESTION_GENERATION, payload=task_payload
                )
            except Exception as exc:
                logger.warning(
                    "knowledge_post_process: failed to enqueue question generation batch {} for {}: {}",
                    batch_index - 1,
                    payload.knowledge_id,
                    exc,
                )
                continue
            if accepted:
                enqueued += 1
        return enqueued

    async def _enqueue_chunk_extract(
        self,
        payload: KnowledgePostProcessPayload,
        chunk: Chunk,
        model_id: str,
        attempt: int,
        chunk_index: int,
    ) -> bool:
        """Enqueue one chunk-extract (graph RAG) task; true only when accepted."""
        if self._enqueuer is None:
            return False
        task_payload: JsonObject = {
            "tenant_id": payload.tenant_id,
            "chunk_id": chunk.id,
            "model_id": model_id,
            "knowledge_id": payload.knowledge_id,
            "attempt": attempt,
            "chunk_index": chunk_index,
        }
        try:
            return await self._enqueuer.enqueue(name=TASK_CHUNK_EXTRACT, payload=task_payload)
        except Exception as exc:
            logger.warning(
                "knowledge_post_process: failed to enqueue chunk extract for {}: {}",
                chunk.id,
                exc,
            )
            return False

    async def _dispatch_wiki_trigger(self, payload: KnowledgePostProcessPayload) -> bool:
        """Schedule the wiki ingest batch trigger; ``False`` when unwired."""
        if self._wiki_dispatcher is None:
            return False
        return await self._wiki_dispatcher.dispatch_ingest(
            tenant_id=payload.tenant_id,
            knowledge_base_id=payload.knowledge_base_id,
            knowledge_id=payload.knowledge_id,
            language=payload.language,
        )

    # ── Span tracking ─────────────────────────────────────────────

    async def _finish_running_multimodal_stage(self, knowledge_id: str, attempt: int) -> None:
        """Close the multimodal stage span when image work really ran.

        The canonical stage row also exists when multimodal processing is
        disabled, but that row is already ``skipped`` and must not be
        rewritten to ``done`` with the postprocess queueing delay as its
        duration.
        """
        if self._tracker is None:
            return
        stage = await self._tracker.lookup_stage(
            knowledge_id=knowledge_id, attempt=attempt, stage=STAGE_MULTIMODAL
        )
        if stage is None or stage.kind != SPAN_KIND_STAGE or stage.status != SPAN_STATUS_RUNNING:
            return
        await self._tracker.end_span(span=stage)

    async def _begin_postprocess_stage(self, knowledge_id: str, attempt: int) -> Span | None:
        if self._tracker is None:
            return None
        return await self._tracker.begin_stage(
            knowledge_id=knowledge_id, attempt=attempt, stage=STAGE_POSTPROCESS
        )

    async def _begin_question_group_span(
        self,
        parent: Span | None,
        batch_count: int,
        chunk_count: int,
    ) -> Span | None:
        if self._tracker is None:
            return None
        return await self._tracker.begin_sub_span(
            parent=parent,
            name=POSTPROCESS_QUESTION_GROUP_SPAN,
            kind=SPAN_KIND_SUBSPAN,
            input={
                "batch_count": batch_count,
                "chunk_count": chunk_count,
                "batch_size": QUESTION_GEN_CHUNK_BATCH_SIZE,
            },
        )

    async def _end_span(self, span: Span | None, output: JsonObject) -> None:
        if self._tracker is not None:
            await self._tracker.end_span(span=span, output=output)

    async def _skip_span(self, span: Span | None, reason: str) -> None:
        if self._tracker is not None:
            await self._tracker.skip_span(span=span, reason=reason)

    async def _fail_span(self, span: Span | None, error_code: str, error_message: str) -> None:
        if self._tracker is not None:
            await self._tracker.fail_span(
                span=span, error_code=error_code, error_message=error_message
            )

    async def _finalize_attempt(self, knowledge_id: str, attempt: int, output: JsonObject) -> None:
        if self._tracker is not None:
            await self._tracker.finalize_attempt(
                knowledge_id=knowledge_id, attempt=attempt, output=output
            )

    # ── Reconciliation ────────────────────────────────────────────

    async def _release_shortfall(
        self, knowledge_id: str, *, planned_owned: int, actual_owned: int
    ) -> None:
        """Release subtask slots whose owning task was never enqueued.

        Each release is a clamped decrement that promotes the row to
        ``completed`` if it brings the counter to zero. A slot whose task
        was never enqueued has no owner and would otherwise strand the row
        in ``finalizing``.
        """
        shortfall = planned_owned - actual_owned
        if shortfall <= 0 or self._finalizer is None:
            return
        logger.warning(
            "knowledge_post_process: releasing {} un-enqueued subtask slot(s) for {} (planned={} actual={})",
            shortfall,
            knowledge_id,
            planned_owned,
            actual_owned,
        )
        for _ in range(shortfall):
            try:
                released = await self._finalizer.finalize_subtask(knowledge_id=knowledge_id)
            except Exception as exc:
                logger.warning(
                    "knowledge_post_process: failed to release subtask slot for {}: {}",
                    knowledge_id,
                    exc,
                )
                return
            if not released:
                return

    def _require_core_seams(self) -> None:
        """Reject a run before the worker layer composed the core seams."""
        missing = [
            name
            for name, seam in (
                ("knowledge_repo", self._knowledge_repo),
                ("kb_service", self._kb_service),
                ("chunk_repo", self._chunk_repo),
                ("tracker", self._tracker),
            )
            if seam is None
        ]
        if missing:
            raise PostProcessError(
                code="postprocess.seams_unwired",
                message="post-process orchestration requires composed seams: " + ", ".join(missing),
            )


def _planned_owned(
    will_spawn_summary: bool,
    will_spawn_wiki: bool,
    question_batch_count: int,
    graph_chunk_count: int,
) -> int:
    """Counter total for subtasks whose slots this call planned to own."""
    planned = question_batch_count + graph_chunk_count
    if will_spawn_summary:
        planned += 1
    if will_spawn_wiki:
        planned += 1
    return planned


def _actual_owned(
    enqueued_summary: bool,
    wiki_slot_owned: bool,
    enqueued_question_count: int,
    enqueued_graph_count: int,
) -> int:
    """Counter total actually owned by this call's successful enqueues."""
    actual = enqueued_question_count + enqueued_graph_count
    if enqueued_summary:
        actual += 1
    if wiki_slot_owned:
        actual += 1
    return actual


# ── Worker entry point ─────────────────────────────────────────────────


async def run_post_process(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    language: str = "",
    attempt: int = 0,
    service: PostProcessService | None = None,
) -> PostProcessOutcome:
    """Worker-side entry point for one post-process run.

    Constructs a :class:`PostProcessService` (or accepts an externally
    built one) and forwards the parsed task payload to
    :meth:`PostProcessService.run`. All seams default to ``None``; the
    worker wiring layer is responsible for composing a fully wired
    service (knowledge repo, KB service, chunk repo, tracker, enqueuers,
    finalizing coordinator) before a real run.
    """
    selected = service or PostProcessService()
    return await selected.run(
        payload=KnowledgePostProcessPayload(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            language=language,
            attempt=attempt,
        )
    )


__all__ = [
    "DEFAULT_QUESTION_COUNT",
    "MAX_QUESTION_COUNT",
    "POSTPROCESS_QUESTION_GROUP_SPAN",
    "QUESTION_GEN_CHUNK_BATCH_SIZE",
    "TASK_CHUNK_EXTRACT",
    "TASK_QUESTION_GENERATION",
    "TASK_SUMMARY_GENERATION",
    "TASK_WIKI_INGEST",
    "EnrichmentEnqueuer",
    "FinalizingCoordinator",
    "KnowledgePostProcessPayload",
    "PostProcessConfig",
    "PostProcessError",
    "PostProcessOutcome",
    "PostProcessService",
    "WikiDispatcher",
    "resolve_post_process_config",
    "run_post_process",
]
