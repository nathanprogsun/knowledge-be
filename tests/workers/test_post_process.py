"""Unit tests for the ARQ ``knowledge:post_process`` task and its core
post-process orchestrator.

The worker half patches
:func:`src.core.knowledge.documents.post_process_service.run_post_process`
so the dispatch contract is exercised without the real seams. The service
half drives :class:`PostProcessService` against ``AsyncMock`` seams to
verify the orchestration decision tree: multimodal stage close, abort
skips, the processing → finalizing handoff, question batching, wiki
dispatch, and the shortfall reconciliation.

No real ARQ broker, no real DB, no real provider calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.common.json import JsonObject
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.post_process_service import (
    QUESTION_GEN_CHUNK_BATCH_SIZE,
    KnowledgePostProcessPayload,
    PostProcessConfig,
    PostProcessError,
    PostProcessOutcome,
    PostProcessService,
    resolve_post_process_config,
)
from src.core.knowledge.documents.span_tracker import (
    SPAN_KIND_STAGE,
    SPAN_STATUS_RUNNING,
    Span,
    SpanTracker,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_NONE,
    SUMMARY_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks.knowledge_post_process import (
    parse_payload,
    task_knowledge_post_process,
)

_NOW = datetime(2026, 2, 1, 8, 0, tzinfo=UTC)


# ── Fixture builders ───────────────────────────────────────────────────


def _span(name: str, *, knowledge_id: str = "kn-1", attempt: int = 1) -> Span:
    """A running stage span handle for the named stage."""
    return Span(
        knowledge_id=knowledge_id,
        attempt=attempt,
        span_id=f"span-{name}",
        parent_span_id=None,
        name=name,
        kind=SPAN_KIND_STAGE,
        status=SPAN_STATUS_RUNNING,
        started_at=_NOW,
    )


def _doc(
    *,
    id: str = "kn-1",
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    parse_status: str = PARSE_STATUS_PROCESSING,
    summary_status: str = SUMMARY_STATUS_NONE,
    metadata: JsonObject | None = None,
) -> Document:
    """A persisted-shape document row for seeding the knowledge mock."""
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title="fixture document",
        source="file",
        channel="web",
        parse_status=parse_status,
        pending_subtasks_count=0,
        summary_status=summary_status,
        enable_status="enabled",
        metadata=metadata,
        custom_metadata={},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _kb(
    *,
    id: str = "kb-1",
    tenant_id: int = 1,
    summary_model_id: str = "model-sum",
    indexing_strategy: JsonObject | None = None,
    question_generation_config: JsonObject | None = None,
    extract_config: JsonObject | None = None,
) -> KnowledgeBaseInfo:
    """A service-shape knowledge base for seeding the KB mock."""
    return KnowledgeBaseInfo(
        id=id,
        name="fixture kb",
        tenant_id=tenant_id,
        summary_model_id=summary_model_id,
        indexing_strategy=indexing_strategy,
        question_generation_config=question_generation_config,
        extract_config=extract_config,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _chunk(
    *,
    id: str,
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    knowledge_id: str = "kn-1",
    chunk_type: str = CHUNK_TYPE_TEXT,
    start_at: int = 0,
    content: str = "chunk body",
) -> Chunk:
    """A chunk row for seeding the chunk mock."""
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=0,
        start_at=start_at,
        end_at=start_at + len(content),
        chunk_type=chunk_type,
        created_at=_NOW,
        updated_at=_NOW,
    )


@dataclass
class _ServiceRig:
    """Bundled service under test plus its mock seams."""

    service: PostProcessService
    tracker: AsyncMock
    knowledge_repo: AsyncMock
    chunk_repo: AsyncMock
    kb_service: AsyncMock
    enqueuer: AsyncMock
    finalizer: AsyncMock
    wiki: AsyncMock
    enqueued_names: list[str] = field(default_factory=list)

    def enqueued_payloads(self, name: str) -> list[JsonObject]:
        """The payloads passed to the enqueuer for ``name``."""
        return [
            cast(JsonObject, call.kwargs["payload"])
            for call in self.enqueuer.enqueue.call_args_list
            if call.kwargs["name"] == name
        ]


def _make_rig(
    *,
    doc: Document | None = None,
    kb: KnowledgeBaseInfo | None = None,
    chunks: list[Chunk] | None = None,
    latest: int = 0,
    multimodal: Span | None = None,
    enqueue_accepts: bool | Callable[[str], bool] = True,
    dispatch_accepts: bool = True,
    promoted: bool = True,
) -> _ServiceRig:
    """Build the service with async mocks configured for one scenario."""
    tracker = AsyncMock(spec=SpanTracker)
    tracker.latest_attempt.return_value = latest
    tracker.lookup_stage.return_value = multimodal
    tracker.begin_stage.return_value = _span("postprocess")
    tracker.begin_sub_span.return_value = _span("postprocess.question")

    knowledge_repo = AsyncMock(spec=KnowledgeRepository)
    knowledge_repo.get_by_id_only.return_value = doc
    knowledge_repo.update_columns.return_value = doc

    chunk_repo = AsyncMock(spec=ChunkRepository)
    chunk_repo.list_by_knowledge_id.return_value = chunks or []

    kb_service = AsyncMock(spec=KBService)
    kb_service.get_knowledge_base_by_id.return_value = kb

    enqueuer = AsyncMock()
    if callable(enqueue_accepts):
        enqueuer.enqueue.side_effect = lambda name, payload: enqueue_accepts(name)
    else:
        enqueuer.enqueue.return_value = enqueue_accepts

    finalizer = AsyncMock()
    finalizer.set_finalizing.return_value = promoted
    finalizer.seed_finalizing_with_wiki.return_value = promoted
    finalizer.finalize_subtask.return_value = True

    wiki = AsyncMock()
    wiki.dispatch_ingest.return_value = dispatch_accepts

    service = PostProcessService(
        knowledge_repo=knowledge_repo,
        kb_service=kb_service,
        chunk_repo=chunk_repo,
        enqueuer=enqueuer,
        wiki_dispatcher=wiki,
        tracker=tracker,
        finalizer=finalizer,
    )
    return _ServiceRig(
        service=service,
        tracker=tracker,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        kb_service=kb_service,
        enqueuer=enqueuer,
        finalizer=finalizer,
        wiki=wiki,
    )


# ── Registry ─────────────────────────────────────────────────────────


def test_handler_registered_under_knowledge_post_process() -> None:
    """The decorator registers the handler at import time."""
    assert get_task("knowledge:post_process") is task_knowledge_post_process


# ── Worker payload model ─────────────────────────────────────────────


def _base_payload() -> dict[str, Any]:
    """Required-field payload shared across delegation tests."""
    return {
        "tenant_id": 1,
        "knowledge_id": "k-1",
        "knowledge_base_id": "kb-1",
    }


def _make_ctx() -> WorkerContext:
    """Build the minimal ARQ context the handler receives."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=_NOW,
        score=0,
    )


def test_parse_payload_accepts_minimum_required_fields() -> None:
    parsed = parse_payload(_base_payload())
    assert parsed.tenant_id == 1
    assert parsed.knowledge_id == "k-1"
    assert parsed.knowledge_base_id == "kb-1"
    assert parsed.language == ""
    assert parsed.attempt == 0


def test_parse_payload_accepts_all_optional_fields() -> None:
    parsed = parse_payload({**_base_payload(), "language": "en-US", "attempt": 3})
    assert parsed.language == "en-US"
    assert parsed.attempt == 3


def test_parse_payload_rejects_missing_tenant() -> None:
    payload = _base_payload()
    payload.pop("tenant_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_knowledge_id() -> None:
    payload = _base_payload()
    payload.pop("knowledge_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_knowledge_base_id() -> None:
    payload = _base_payload()
    payload.pop("knowledge_base_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_ignores_unknown_fields() -> None:
    parsed = parse_payload({**_base_payload(), "extra": "ignored"})
    assert parsed.tenant_id == 1


# ── Worker delegation ────────────────────────────────────────────────


async def test_task_delegates_to_core_run_post_process() -> None:
    captured: dict[str, Any] = {}

    async def _fake_core(**kwargs: Any) -> PostProcessOutcome:
        captured.update(kwargs)
        return PostProcessOutcome(
            chunks_total=5,
            enqueued_summary=True,
            enqueued_question=True,
            enqueued_question_count=1,
            enqueued_wiki=True,
            wiki_slot_owned=True,
            enqueued_graph=True,
            enqueued_graph_count=2,
        )

    with patch(
        "src.workers.tasks.knowledge_post_process._core_run_post_process",
        side_effect=_fake_core,
    ):
        result = await task_knowledge_post_process(
            _make_ctx(),
            **_base_payload(),
            language="zh-CN",
            attempt=2,
        )

    assert captured["tenant_id"] == 1
    assert captured["knowledge_id"] == "k-1"
    assert captured["knowledge_base_id"] == "kb-1"
    assert captured["language"] == "zh-CN"
    assert captured["attempt"] == 2
    assert captured["service"] is None

    assert result == {
        "skipped": False,
        "reason": "",
        "chunks_total": 5,
        "enqueued_summary": True,
        "enqueued_question": True,
        "enqueued_question_count": 1,
        "enqueued_wiki": True,
        "wiki_slot_owned": True,
        "enqueued_graph": True,
        "enqueued_graph_count": 2,
    }


async def test_task_forwards_injected_service() -> None:
    service = PostProcessService()
    captured: dict[str, Any] = {}

    async def _fake_core(**kwargs: Any) -> PostProcessOutcome:
        captured.update(kwargs)
        return PostProcessOutcome()

    with patch(
        "src.workers.tasks.knowledge_post_process._core_run_post_process",
        side_effect=_fake_core,
    ):
        await task_knowledge_post_process(
            _make_ctx(),
            service=cast(Any, service),
            **_base_payload(),
        )

    assert captured["service"] is service


async def test_task_serialises_skipped_outcome() -> None:
    async def _fake_core(**kwargs: Any) -> PostProcessOutcome:
        return PostProcessOutcome(skipped=True, reason=PARSE_STATUS_CANCELLED)

    with patch(
        "src.workers.tasks.knowledge_post_process._core_run_post_process",
        side_effect=_fake_core,
    ):
        result = await task_knowledge_post_process(_make_ctx(), **_base_payload())

    assert result["skipped"] is True
    assert result["reason"] == PARSE_STATUS_CANCELLED


async def test_task_rejects_invalid_payload() -> None:
    """Invalid payloads surface as Pydantic validation errors."""
    with pytest.raises(ValidationError):
        await task_knowledge_post_process(_make_ctx(), tenant_id="not-an-int")


# ── Effective config resolution ──────────────────────────────────────


def test_resolve_config_defaults_when_kb_unconfigured() -> None:
    kb = _kb()
    config = resolve_post_process_config(kb, None)
    assert config == PostProcessConfig(
        question_generation_enabled=False,
        question_count=3,
        graph_enabled=False,
    )


def test_resolve_config_question_override_wins() -> None:
    kb = _kb(
        question_generation_config={"enabled": True, "question_count": 5},
        extract_config={"enabled": True},
    )
    config = resolve_post_process_config(
        kb,
        {"question_generation_config": {"enabled": False, "question_count": 9}},
    )
    assert config.question_generation_enabled is False
    assert config.question_count == 9


def test_resolve_config_clamps_question_count() -> None:
    kb = _kb(question_generation_config={"enabled": True, "question_count": 99})
    assert resolve_post_process_config(kb, None).question_count == 10


def test_resolve_config_graph_requires_extract_enabled() -> None:
    strategy: JsonObject = {"graph_enabled": True, "vector_enabled": True}
    no_extract = _kb(indexing_strategy=strategy)
    assert resolve_post_process_config(no_extract, None).graph_enabled is False
    with_extract = _kb(indexing_strategy=strategy, extract_config={"enabled": True})
    assert resolve_post_process_config(with_extract, None).graph_enabled is True


def test_resolve_config_graph_override_still_needs_extract() -> None:
    kb = _kb(extract_config={"enabled": True})
    config = resolve_post_process_config(kb, {"graph_enabled": True})
    assert config.graph_enabled is True
    no_extract = _kb()
    assert resolve_post_process_config(no_extract, {"graph_enabled": True}).graph_enabled is False


# ── Service orchestration ────────────────────────────────────────────


async def test_run_skips_when_knowledge_missing() -> None:
    rig = _make_rig(doc=None, kb=_kb())
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is True
    assert outcome.reason == "knowledge_not_found"
    rig.enqueuer.enqueue.assert_not_called()


async def test_run_skips_cancelled_knowledge_and_closes_span() -> None:
    doc = _doc(parse_status=PARSE_STATUS_CANCELLED)
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[_chunk(id="c-1")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is True
    assert outcome.reason == PARSE_STATUS_CANCELLED
    rig.tracker.skip_span.assert_called_once()
    rig.enqueuer.enqueue.assert_not_called()
    rig.tracker.finalize_attempt.assert_not_called()


async def test_run_skips_deleting_knowledge() -> None:
    doc = _doc(parse_status=PARSE_STATUS_DELETING)
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[_chunk(id="c-1")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is True
    assert outcome.reason == PARSE_STATUS_DELETING


async def test_run_skips_when_not_processing() -> None:
    doc = _doc(parse_status=PARSE_STATUS_COMPLETED)
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[_chunk(id="c-1")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is True
    assert outcome.reason == PARSE_STATUS_COMPLETED
    rig.tracker.finalize_attempt.assert_called_once()
    rig.enqueuer.enqueue.assert_not_called()


async def test_run_fast_path_marks_completed_without_chunks() -> None:
    rig = _make_rig(doc=_doc(), kb=_kb(), chunks=[])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is False
    assert outcome.chunks_total == 0
    rig.knowledge_repo.update_columns.assert_called_once()
    _, values = rig.knowledge_repo.update_columns.call_args.args
    assert values["parse_status"] == PARSE_STATUS_COMPLETED
    rig.enqueuer.enqueue.assert_not_called()
    rig.tracker.finalize_attempt.assert_called_once()


async def test_run_fans_out_summary_and_question() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": True},
        question_generation_config={"enabled": True, "question_count": 5},
    )
    chunks = [_chunk(id=f"c-{i}", start_at=i * 10) for i in range(3)]
    rig = _make_rig(doc=doc, kb=kb, chunks=chunks)

    outcome = await rig.service.run(payload=_payload())

    assert outcome.skipped is False
    assert outcome.chunks_total == 3
    assert outcome.enqueued_summary is True
    assert outcome.enqueued_question is True
    assert outcome.enqueued_question_count == 1

    rig.finalizer.set_finalizing.assert_called_once_with(knowledge_id="kn-1", expected_subtasks=2)
    summary_payloads = rig.enqueued_payloads("summary:generation")
    assert len(summary_payloads) == 1
    assert summary_payloads[0]["knowledge_id"] == "kn-1"
    question_payloads = rig.enqueued_payloads("question:generation")
    assert len(question_payloads) == 1
    assert question_payloads[0]["chunk_ids"] == ["c-0", "c-1", "c-2"]
    assert question_payloads[0]["question_count"] == 5
    # summary_status pending reflects the queued summary.
    rig.knowledge_repo.update_columns.assert_called()
    _, status_values = rig.knowledge_repo.update_columns.call_args_list[-1].args
    assert status_values["summary_status"] == SUMMARY_STATUS_PENDING
    # No shortfall: summary (1) + question batch (1) both owned.
    rig.finalizer.finalize_subtask.assert_not_called()
    rig.tracker.finalize_attempt.assert_called_once()


def _payload(**overrides: Any) -> KnowledgePostProcessPayload:
    values: dict[str, Any] = {
        "tenant_id": 1,
        "knowledge_id": "kn-1",
        "knowledge_base_id": "kb-1",
        "language": "zh-CN",
    }
    values.update(overrides)
    return KnowledgePostProcessPayload(**values)


async def test_run_batches_questions_by_window_size() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": True},
        question_generation_config={"enabled": True, "question_count": 3},
    )
    total = QUESTION_GEN_CHUNK_BATCH_SIZE + 3
    chunks = [_chunk(id=f"c-{i}", start_at=i * 10) for i in range(total)]
    rig = _make_rig(doc=doc, kb=kb, chunks=chunks)

    outcome = await rig.service.run(payload=_payload())

    expected_batches = (total + QUESTION_GEN_CHUNK_BATCH_SIZE - 1) // QUESTION_GEN_CHUNK_BATCH_SIZE
    assert outcome.enqueued_question_count == expected_batches
    payloads = rig.enqueued_payloads("question:generation")
    assert len(payloads) == expected_batches
    assert payloads[0]["batch_index"] == 0
    assert payloads[0]["chunk_ids"] == [f"c-{i}" for i in range(QUESTION_GEN_CHUNK_BATCH_SIZE)]
    assert payloads[0]["next_chunk_id"] == f"c-{QUESTION_GEN_CHUNK_BATCH_SIZE}"
    assert payloads[-1]["batch_index"] == expected_batches - 1
    assert payloads[-1]["prev_chunk_id"] == f"c-{QUESTION_GEN_CHUNK_BATCH_SIZE - 1}"
    assert "next_chunk_id" not in payloads[-1]


async def test_run_skips_question_when_embedding_not_needed() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": False, "keyword_enabled": False},
        question_generation_config={"enabled": True, "question_count": 3},
    )
    rig = _make_rig(doc=doc, kb=kb, chunks=[_chunk(id="c-0")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.enqueued_summary is True
    assert outcome.enqueued_question is False
    assert outcome.enqueued_question_count == 0
    assert rig.enqueued_payloads("question:generation") == []
    rig.finalizer.set_finalizing.assert_called_once_with(knowledge_id="kn-1", expected_subtasks=1)


async def test_run_reconciles_shortfall_when_summary_rejected() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": True},
        question_generation_config={"enabled": True, "question_count": 3},
    )
    rig = _make_rig(
        doc=doc,
        kb=kb,
        chunks=[_chunk(id="c-0"), _chunk(id="c-1", start_at=10)],
        enqueue_accepts=lambda name: name == "question:generation",
    )
    outcome = await rig.service.run(payload=_payload())
    # Question batch enqueued (1) but summary rejected → shortfall 1 released.
    assert outcome.enqueued_summary is False
    assert outcome.enqueued_question_count == 1
    rig.finalizer.finalize_subtask.assert_awaited_once_with(knowledge_id="kn-1")
    # summary_status was marked failed.
    _, status_values = rig.knowledge_repo.update_columns.call_args_list[-1].args
    assert status_values["summary_status"] == "failed"


async def test_run_releases_shortfall_when_graph_extract_rejected() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"graph_enabled": True, "vector_enabled": True},
        extract_config={"enabled": True},
    )
    chunks = [_chunk(id=f"c-{i}", start_at=i * 10) for i in range(2)]
    rig = _make_rig(doc=doc, kb=kb, chunks=chunks, enqueue_accepts=False)
    outcome = await rig.service.run(payload=_payload())
    assert outcome.enqueued_graph_count == 0
    assert outcome.enqueued_graph is False
    # expected: summary(1) + graph(2) = 3; actual owned 0 → 3 releases.
    assert rig.finalizer.finalize_subtask.await_count == 3


async def test_run_wiki_seeds_finalizing_and_dispatches_trigger() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"wiki_enabled": True, "vector_enabled": True},
    )
    rig = _make_rig(doc=doc, kb=kb, chunks=[_chunk(id="c-0")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.wiki_slot_owned is True
    assert outcome.enqueued_wiki is True
    rig.finalizer.seed_finalizing_with_wiki.assert_called_once_with(
        knowledge_id="kn-1", expected_subtasks=2
    )
    rig.wiki.dispatch_ingest.assert_awaited_once()


async def test_run_retries_wiki_trigger_when_finalizing() -> None:
    doc = _doc(parse_status=PARSE_STATUS_FINALIZING)
    kb = _kb(indexing_strategy={"wiki_enabled": True})
    rig = _make_rig(doc=doc, kb=kb, chunks=[_chunk(id="c-0")])
    outcome = await rig.service.run(payload=_payload())
    assert outcome.wiki_slot_owned is True
    assert outcome.enqueued_wiki is True
    assert outcome.reason == ""
    rig.wiki.dispatch_ingest.assert_awaited_once()
    rig.finalizer.seed_finalizing_with_wiki.assert_not_called()
    rig.tracker.finalize_attempt.assert_called_once()


async def test_run_raises_when_wiki_trigger_rejected_on_retry() -> None:
    doc = _doc(parse_status=PARSE_STATUS_FINALIZING)
    kb = _kb(indexing_strategy={"wiki_enabled": True})
    rig = _make_rig(
        doc=doc,
        kb=kb,
        chunks=[_chunk(id="c-0")],
        dispatch_accepts=False,
    )
    with pytest.raises(PostProcessError):
        await rig.service.run(payload=_payload())
    rig.tracker.fail_span.assert_called_once()


async def test_run_skips_fanout_when_promotion_lost() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": True},
        question_generation_config={"enabled": True},
    )
    rig = _make_rig(
        doc=doc,
        kb=kb,
        chunks=[_chunk(id="c-0")],
        promoted=False,
    )
    outcome = await rig.service.run(payload=_payload())
    assert outcome.skipped is True
    assert outcome.reason == "knowledge_no_longer_processing"
    rig.enqueuer.enqueue.assert_not_called()
    rig.tracker.finalize_attempt.assert_called_once()


async def test_run_closes_running_multimodal_stage() -> None:
    doc = _doc()
    rig = _make_rig(
        doc=doc,
        kb=_kb(),
        chunks=[],
        multimodal=_span("multimodal"),
        latest=1,
    )
    await rig.service.run(payload=_payload())
    # The running multimodal stage was closed, not skipped.
    rig.tracker.lookup_stage.assert_awaited_once_with(
        knowledge_id="kn-1", attempt=1, stage="multimodal"
    )
    closed = [
        call.kwargs["span"]
        for call in rig.tracker.end_span.call_args_list
        if call.kwargs.get("span") is not None and call.kwargs["span"].name == "multimodal"
    ]
    assert len(closed) == 1


async def test_run_uses_payload_attempt_when_provided() -> None:
    doc = _doc()
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[])
    await rig.service.run(payload=_payload(attempt=4))
    rig.tracker.latest_attempt.assert_not_called()
    # begin_stage received the payload attempt.
    begin_kwargs = rig.tracker.begin_stage.call_args.kwargs
    assert begin_kwargs["attempt"] == 4


async def test_run_uses_latest_attempt_fallback() -> None:
    doc = _doc()
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[], latest=7)
    await rig.service.run(payload=_payload(attempt=0))
    rig.tracker.latest_attempt.assert_awaited_once_with("kn-1")
    begin_kwargs = rig.tracker.begin_stage.call_args.kwargs
    assert begin_kwargs["attempt"] == 7


async def test_run_raises_when_core_seams_unwired() -> None:
    service = PostProcessService()
    with pytest.raises(PostProcessError):
        await service.run(payload=_payload())


async def test_run_includes_ocr_and_caption_chunks_in_text_count() -> None:
    doc = _doc()
    kb = _kb(
        indexing_strategy={"vector_enabled": True},
        question_generation_config={"enabled": True, "question_count": 3},
    )
    chunks = [
        _chunk(id="c-text", chunk_type=CHUNK_TYPE_TEXT),
        _chunk(id="c-ocr", chunk_type=CHUNK_TYPE_IMAGE_OCR, start_at=10),
        _chunk(id="c-cap", chunk_type=CHUNK_TYPE_IMAGE_CAPTION, start_at=20),
    ]
    rig = _make_rig(doc=doc, kb=kb, chunks=chunks)
    outcome = await rig.service.run(payload=_payload())
    assert outcome.chunks_total == 3
    # Only the plain text chunk feeds question generation.
    question_payloads = rig.enqueued_payloads("question:generation")
    assert len(question_payloads) == 1
    assert question_payloads[0]["chunk_ids"] == ["c-text"]


async def test_run_does_not_touch_summary_status_when_promotion_skips() -> None:
    """The non-processing path never writes summary_status."""
    doc = _doc(parse_status=PARSE_STATUS_FAILED)
    rig = _make_rig(doc=doc, kb=_kb(), chunks=[_chunk(id="c-0")])
    await rig.service.run(payload=_payload())
    rig.knowledge_repo.update_columns.assert_not_called()
