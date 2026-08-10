"""Unit tests for session search, stop, and stream continue.

Covers the three session operations added in this change with tiny
in-memory fakes for the pipeline engine, the knowledge / model
resolvers, and the message reader. The real ``StreamManager`` memory
backend is exercised directly for stop and continue; a raising fake
covers the error path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.pipeline import (
    ERR_SEARCH_NOTHING,
    EventType,
    PluginError,
    SearchResult,
    SearchTarget,
    SearchTargetType,
)
from src.core.chat.service import TagScope
from src.core.chat.sessions.continue_stream import continue_stream
from src.core.chat.sessions.search_knowledge import SearchKnowledgeService
from src.core.chat.sessions.stop import StopStreamService
from src.core.chat.stream.manager import MemoryStreamManager
from src.core.chat.stream.types import Event
from src.core.contracts.infra import ModelParameters
from src.core.contracts.knowledge import Knowledge
from src.core.infra.models.types import ModelInfo
from src.core.knowledge.knowledge_bases.hybrid_search import RetrievalConfig
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)
from src.db.models.message import Message

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

#: Every retrieval stage triggered by one search run.
_FULL_SEARCH_EVENTS = [
    str(EventType.CHUNK_SEARCH),
    str(EventType.CHUNK_RERANK),
    str(EventType.CHUNK_MERGE),
    str(EventType.FILTER_TOP_K),
]


# ── Fakes ────────────────────────────────────────────────────────────


class _FakePipeline:
    """Simulates the pipeline ``EventManager`` for a search run.

    ``trigger`` records the event types it saw and, unless configured to
    fail or report ``search_nothing``, writes ``merge_result`` onto the
    carrier so the service can return it.
    """

    def __init__(
        self,
        *,
        merge_result: list[SearchResult] | None = None,
        fail_event: str | None = None,
        fail_error: PluginError | None = None,
        search_nothing_at: str | None = None,
    ) -> None:
        self.triggered: list[str] = []
        self.merge_result = merge_result or []
        self.fail_event = fail_event
        self.fail_error = fail_error
        self.search_nothing_at = search_nothing_at
        self.last_ctx = None

    async def trigger(self, ctx, event_type, pipeline_ctx):
        self.triggered.append(str(event_type))
        self.last_ctx = pipeline_ctx
        if self.search_nothing_at and str(event_type) == self.search_nothing_at:
            return ERR_SEARCH_NOTHING
        if self.fail_event and str(event_type) == self.fail_event:
            return self.fail_error or PluginError(
                description="Failed to search knowledge base",
                error_type="search_failed",
            )
        pipeline_ctx.merge_result = self.merge_result
        return None


class _FakeKBResolver:
    def __init__(self, kbs: list[KnowledgeBaseInfo] | None = None) -> None:
        self.kbs = kbs or []
        self.calls: list[list[str]] = []

    async def get_knowledge_bases_by_ids(self, *, ids: list[str]) -> list[KnowledgeBaseInfo]:
        self.calls.append(list(ids))
        by_id = {kb.id: kb for kb in self.kbs}
        return [by_id[kid] for kid in ids if kid in by_id]


class _FakeKnowledgeResolver:
    def __init__(
        self,
        documents: list[Knowledge] | None = None,
        tag_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.documents = documents or []
        self.tag_map = tag_map or {}
        self.tag_calls: list[tuple[int, str, list[str]]] = []

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[Knowledge]:
        by_id = {doc.id: doc for doc in self.documents}
        return [by_id[kid] for kid in ids if kid in by_id]

    async def list_knowledge_ids_by_tag_ids(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_ids: list[str],
    ) -> list[str]:
        self.tag_calls.append((tenant_id, knowledge_base_id, list(tag_ids)))
        return list(self.tag_map.get(knowledge_base_id, []))


class _FakeModelResolver:
    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self.models = models or []
        self.calls: list[tuple[int, str | None]] = []

    async def list_models(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
    ) -> list[ModelInfo]:
        self.calls.append((tenant_id, model_type))
        return list(self.models)


class _FakePermissionChecker:
    def __init__(self, allowed: bool = False) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, int, str]] = []

    async def has_tenant_kb_permission(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        org_role: str,
    ) -> bool:
        self.calls.append((knowledge_base_id, tenant_id, org_role))
        return self.allowed


class _FakeMessageReader:
    def __init__(self, messages: dict[str, Message] | None = None) -> None:
        self.messages = messages or {}

    async def get_by_id_and_session(
        self,
        *,
        session_id: str,
        message_id: str,
    ) -> Message | None:
        return self.messages.get((session_id, message_id))


class _RaisingStreamManager:
    """Stream manager whose reads always fail (error path)."""

    async def get_events(self, session_id: str, message_id: str, offset: int, limit=None):
        raise RuntimeError("stream backend unavailable")


# ── Fixtures / builders ──────────────────────────────────────────────


def _kb(
    *,
    kb_id: str,
    tenant_id: int = 1,
    kb_type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT,
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=kb_id,
        name=f"kb-{kb_id}",
        type=kb_type,
        tenant_id=tenant_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _doc(
    *,
    doc_id: str,
    kb_id: str,
    tenant_id: int = 1,
) -> Knowledge:
    return Knowledge(
        id=doc_id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="document",
        title=f"doc-{doc_id}",
        source="manual",
        parse_status="parsed",
        enable_status="enabled",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _model(*, model_id: str, model_type: str) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        tenant_id=1,
        name=model_id,
        type=model_type,
        source="builtin",
        parameters=ModelParameters(),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _hit(*, hit_id: str, content: str = "hit") -> SearchResult:
    return SearchResult(id=hit_id, content=content)


def _search_service(
    *,
    pipeline: _FakePipeline,
    kb_resolver: _FakeKBResolver | None = None,
    knowledge_resolver: _FakeKnowledgeResolver | None = None,
    model_resolver: _FakeModelResolver | None = None,
    permission_checker: _FakePermissionChecker | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> SearchKnowledgeService:
    return SearchKnowledgeService(
        tenant_id=1,
        user_id="u1",
        event_manager=pipeline,  # type: ignore[arg-type]
        knowledge_base_resolver=kb_resolver,
        knowledge_resolver=knowledge_resolver,
        model_resolver=model_resolver,
        kb_permission_checker=permission_checker,
        retrieval_config=retrieval_config,
    )


# ── Search: happy path ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_merged_results() -> None:
    # Arrange
    pipeline = _FakePipeline(merge_result=[_hit(hit_id="r1"), _hit(hit_id="r2")])
    service = _search_service(pipeline=pipeline)

    # Act
    results = await service.search(
        query="what is rrf",
        knowledge_base_ids=["kb1"],
    )

    # Assert
    assert [r.id for r in results] == ["r1", "r2"]
    assert pipeline.triggered == [
        str(EventType.CHUNK_SEARCH),
        str(EventType.CHUNK_RERANK),
        str(EventType.CHUNK_MERGE),
        str(EventType.FILTER_TOP_K),
    ]


@pytest.mark.asyncio
async def test_search_builds_whole_kb_target() -> None:
    # Arrange
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb1")])
    service = _search_service(pipeline=pipeline, kb_resolver=kb_resolver)

    # Act
    await service.search(query="q", knowledge_base_ids=["kb1"])

    # Assert
    assert kb_resolver.calls == [["kb1"]]
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb1",
            tenant_id=1,
        )
    ]
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_empty_targets_returns_empty() -> None:
    # Arrange
    pipeline = _FakePipeline()
    service = _search_service(pipeline=pipeline)

    # Act
    results = await service.search(query="q")

    # Assert
    assert results == []
    assert pipeline.triggered == []


@pytest.mark.asyncio
async def test_search_nothing_returns_empty() -> None:
    # Arrange
    pipeline = _FakePipeline(search_nothing_at=str(EventType.CHUNK_SEARCH))
    service = _search_service(pipeline=pipeline)

    # Act
    results = await service.search(query="q", knowledge_base_ids=["kb1"])

    # Assert — the run stops at the first stage reporting nothing.
    assert results == []
    assert pipeline.triggered == [str(EventType.CHUNK_SEARCH)]


@pytest.mark.asyncio
async def test_search_requires_query() -> None:
    # Arrange
    service = _search_service(pipeline=_FakePipeline())

    # Act / Assert
    with pytest.raises(ValidationError):
        await service.search(query="   ", knowledge_base_ids=["kb1"])


@pytest.mark.asyncio
async def test_search_propagates_pipeline_error() -> None:
    # Arrange
    pipeline = _FakePipeline(fail_event=str(EventType.CHUNK_RERANK))
    service = _search_service(pipeline=pipeline)

    # Act / Assert
    with pytest.raises(RuntimeError):
        await service.search(query="q", knowledge_base_ids=["kb1"])


@pytest.mark.asyncio
async def test_search_propagates_underlying_exception() -> None:
    # Arrange — the plugin error wraps a concrete exception that must
    # surface unchanged (type + message) instead of a generic error.
    underlying = TimeoutError("rerank timed out")
    pipeline = _FakePipeline(
        fail_event=str(EventType.CHUNK_RERANK),
        fail_error=PluginError(
            description="Reranking failed",
            error_type="rerank_failed",
            err=underlying,
        ),
    )
    service = _search_service(pipeline=pipeline)

    # Act / Assert
    with pytest.raises(TimeoutError, match="rerank timed out"):
        await service.search(query="q", knowledge_base_ids=["kb1"])


# ── Search: target building ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_groups_knowledge_ids_by_kb() -> None:
    # Arrange
    pipeline = _FakePipeline()
    knowledge_resolver = _FakeKnowledgeResolver(
        documents=[
            _doc(doc_id="k1", kb_id="kb1"),
            _doc(doc_id="k2", kb_id="kb1"),
            _doc(doc_id="k3", kb_id="kb2"),
        ]
    )
    service = _search_service(pipeline=pipeline, knowledge_resolver=knowledge_resolver)

    # Act
    await service.search(query="q", knowledge_ids=["k1", "k2", "k3"])

    # Assert — one knowledge target per KB, files grouped.
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb1",
            tenant_id=1,
            knowledge_ids=["k1", "k2"],
            disable_recall_thresholds=True,
        ),
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb2",
            tenant_id=1,
            knowledge_ids=["k3"],
            disable_recall_thresholds=True,
        ),
    ]
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_builds_document_tag_scope_target() -> None:
    # Arrange
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb1")])
    knowledge_resolver = _FakeKnowledgeResolver(tag_map={"kb1": ["k1", "k2"]})
    service = _search_service(
        pipeline=pipeline,
        kb_resolver=kb_resolver,
        knowledge_resolver=knowledge_resolver,
    )

    # Act
    await service.search(
        query="q",
        tag_scopes=[TagScope(knowledge_base_id="kb1", tag_ids=("t1", "t2"))],
    )

    # Assert — document KB tag scope resolves to a per-file target.
    assert knowledge_resolver.tag_calls == [(1, "kb1", ["t1", "t2"])]
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb1",
            tenant_id=1,
            knowledge_ids=["k1", "k2"],
            scope_tag_ids=["t1", "t2"],
            disable_recall_thresholds=True,
        )
    ]
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_faq_tag_scope_keeps_kb_target() -> None:
    # Arrange
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb1", kb_type=KNOWLEDGE_BASE_TYPE_FAQ)])
    service = _search_service(pipeline=pipeline, kb_resolver=kb_resolver)

    # Act
    await service.search(
        query="q",
        tag_scopes=[TagScope(knowledge_base_id="kb1", tag_ids=("t1",))],
    )

    # Assert — FAQ KB tag scope stays a whole-KB target carrying the tags.
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb1",
            tenant_id=1,
            tag_ids=["t1"],
            scope_tag_ids=["t1"],
            disable_recall_thresholds=True,
        )
    ]
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_resolves_shared_kb_tenant() -> None:
    # Arrange
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb9", tenant_id=9)])
    permission_checker = _FakePermissionChecker(allowed=True)
    service = _search_service(
        pipeline=pipeline,
        kb_resolver=kb_resolver,
        permission_checker=permission_checker,
    )

    # Act
    await service.search(query="q", knowledge_base_ids=["kb9"])

    # Assert — the shared KB resolves to its owning tenant for the target.
    assert permission_checker.calls == [("kb9", 1, "viewer")]
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb9",
            tenant_id=9,
        )
    ]
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_skips_knowledge_covered_by_full_kb() -> None:
    # Arrange — kb1 is fully searched, so its explicit files are redundant.
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb1")])
    knowledge_resolver = _FakeKnowledgeResolver(
        documents=[_doc(doc_id="k1", kb_id="kb1"), _doc(doc_id="k2", kb_id="kb2")]
    )
    service = _search_service(
        pipeline=pipeline,
        kb_resolver=kb_resolver,
        knowledge_resolver=knowledge_resolver,
    )

    # Act
    await service.search(query="q", knowledge_base_ids=["kb1"], knowledge_ids=["k1", "k2"])

    # Assert — only the whole-KB target and the kb2 file target remain.
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE_BASE,
            knowledge_base_id="kb1",
            tenant_id=1,
        ),
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb2",
            tenant_id=1,
            knowledge_ids=["k2"],
            disable_recall_thresholds=True,
        ),
    ]


@pytest.mark.asyncio
async def test_search_tag_scope_intersects_explicit_files() -> None:
    # Arrange — the tag scope resolves k1/k2/k3, but the request only
    # names k1 and k2; the intersection wins.
    pipeline = _FakePipeline()
    kb_resolver = _FakeKBResolver([_kb(kb_id="kb1")])
    knowledge_resolver = _FakeKnowledgeResolver(
        documents=[_doc(doc_id="k1", kb_id="kb1"), _doc(doc_id="k2", kb_id="kb1")],
        tag_map={"kb1": ["k1", "k2", "k3"]},
    )
    service = _search_service(
        pipeline=pipeline,
        kb_resolver=kb_resolver,
        knowledge_resolver=knowledge_resolver,
    )

    # Act
    await service.search(
        query="q",
        knowledge_ids=["k1", "k2"],
        tag_scopes=[TagScope(knowledge_base_id="kb1", tag_ids=("t1",))],
    )

    # Assert
    assert pipeline.last_ctx.search_targets == [
        SearchTarget(
            type=SearchTargetType.KNOWLEDGE,
            knowledge_base_id="kb1",
            tenant_id=1,
            knowledge_ids=["k1", "k2"],
            scope_tag_ids=["t1"],
            disable_recall_thresholds=True,
        )
    ]


# ── Search: config ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_picks_rerank_model() -> None:
    # Arrange
    pipeline = _FakePipeline()
    model_resolver = _FakeModelResolver([_model(model_id="m-rerank", model_type="rerank")])
    service = _search_service(pipeline=pipeline, model_resolver=model_resolver)

    # Act
    await service.search(query="q", knowledge_base_ids=["kb1"])

    # Assert — the rerank model lands on the carrier before the stages run.
    assert model_resolver.calls == [(1, "rerank")]
    assert pipeline.last_ctx.rerank_model_id == "m-rerank"
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


@pytest.mark.asyncio
async def test_search_applies_retrieval_config() -> None:
    # Arrange
    pipeline = _FakePipeline()
    config = RetrievalConfig(embedding_top_k=7, rerank_top_k=3)
    service = _search_service(pipeline=pipeline, retrieval_config=config)

    # Act
    await service.search(query="q", knowledge_base_ids=["kb1"])

    # Assert — the effective values override the carrier defaults.
    assert pipeline.last_ctx.embedding_top_k == 7
    assert pipeline.last_ctx.rerank_top_k == 3
    assert pipeline.triggered == _FULL_SEARCH_EVENTS


# ── Stop ─────────────────────────────────────────────────────────────


def _message(*, message_id: str, session_id: str, is_completed: bool = False) -> Message:
    return Message(
        id=message_id,
        session_id=session_id,
        role="assistant",
        content="",
        is_completed=is_completed,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_stop_active_stream_appends_event_and_cancels() -> None:
    # Arrange
    manager = MemoryStreamManager()
    reader = _FakeMessageReader({("s1", "m1"): _message(message_id="m1", session_id="s1")})
    service = StopStreamService(stream_manager=manager, message_reader=reader)

    # Act
    result = await service.stop("s1", "m1")

    # Assert
    assert result.stopped is True
    assert result.session_id == "s1"
    assert result.message_id == "m1"
    assert manager.is_cancelled("s1", "m1") is True
    events, _ = await manager.get_events("s1", "m1", 0)
    assert len(events) == 1
    assert events[0].type == "stop"
    assert events[0].done is True
    assert events[0].data == {
        "session_id": "s1",
        "message_id": "m1",
        "reason": "user_requested",
    }


@pytest.mark.asyncio
async def test_stop_already_completed_message_is_noop() -> None:
    # Arrange
    manager = MemoryStreamManager()
    reader = _FakeMessageReader(
        {("s1", "m1"): _message(message_id="m1", session_id="s1", is_completed=True)}
    )
    service = StopStreamService(stream_manager=manager, message_reader=reader)

    # Act
    result = await service.stop("s1", "m1")

    # Assert
    assert result.stopped is False
    assert manager.is_cancelled("s1", "m1") is False
    events, _ = await manager.get_events("s1", "m1", 0)
    assert events == []


@pytest.mark.asyncio
async def test_stop_missing_message_raises() -> None:
    # Arrange
    manager = MemoryStreamManager()
    service = StopStreamService(stream_manager=manager, message_reader=_FakeMessageReader())

    # Act / Assert
    with pytest.raises(NotFoundError):
        await service.stop("s1", "m1")


@pytest.mark.asyncio
async def test_stop_requires_ids() -> None:
    # Arrange
    service = StopStreamService(stream_manager=MemoryStreamManager())

    # Act / Assert
    with pytest.raises(ValidationError):
        await service.stop("", "m1")
    with pytest.raises(ValidationError):
        await service.stop("s1", "")


@pytest.mark.asyncio
async def test_stop_without_message_reader_still_stops() -> None:
    # Arrange
    manager = MemoryStreamManager()
    service = StopStreamService(stream_manager=manager)

    # Act
    result = await service.stop("s1", "m1")

    # Assert
    assert result.stopped is True
    assert manager.is_cancelled("s1", "m1") is True


# ── Continue ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_continue_replays_existing_events() -> None:
    # Arrange
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", Event(id="e1", type="answer", content="a"))
    await manager.append_event("s1", "m1", Event(id="e2", type="answer", content="b"))

    # Act — collect the replay, then stop before the generator polls.
    seen: list[str] = []
    async for event in continue_stream(manager, "s1", "m1"):
        seen.append(event.id)
        if len(seen) == 2:
            break

    # Assert
    assert seen == ["e1", "e2"]


@pytest.mark.asyncio
async def test_continue_no_stream_raises() -> None:
    # Arrange
    manager = MemoryStreamManager()

    # Act / Assert
    with pytest.raises(NotFoundError):
        async for _ in continue_stream(manager, "s1", "m1"):
            pass


@pytest.mark.asyncio
async def test_continue_stops_at_complete_event() -> None:
    # Arrange
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", Event(id="e1", type="answer", content="a"))
    await manager.append_event("s1", "m1", Event(id="e2", type="complete"))

    # Act
    seen = [event async for event in continue_stream(manager, "s1", "m1")]

    # Assert
    assert [e.id for e in seen] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_continue_polls_for_new_events() -> None:
    # Arrange
    manager = MemoryStreamManager()
    await manager.append_event("s1", "m1", Event(id="e1", type="answer", content="a"))

    async def _collect():
        seen = []
        async for event in continue_stream(manager, "s1", "m1", poll_interval_seconds=0.01):
            seen.append(event.id)
            if event.id == "e1":
                # Simulate the producer appending more events mid-poll.
                await manager.append_event("s1", "m1", Event(id="e2", type="complete"))
        return seen

    # Act
    seen = await _collect()

    # Assert
    assert seen == ["e1", "e2"]


@pytest.mark.asyncio
async def test_continue_requires_ids() -> None:
    # Arrange
    manager = MemoryStreamManager()

    # Act / Assert
    with pytest.raises(ValidationError):
        async for _ in continue_stream(manager, "", "m1"):
            pass
    with pytest.raises(ValidationError):
        async for _ in continue_stream(manager, "s1", ""):
            pass


@pytest.mark.asyncio
async def test_continue_propagates_manager_error() -> None:
    # Arrange
    manager = _RaisingStreamManager()

    # Act / Assert
    with pytest.raises(RuntimeError):
        async for _ in continue_stream(manager, "s1", "m1"):
            pass
