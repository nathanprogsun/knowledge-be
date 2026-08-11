"""Unit tests for the ``kb:clone``, ``kb:delete`` and ``index:delete`` worker tasks.

Covers the worker-side surface of all three knowledge-base task
handlers: each handler is registered under its upstream task name, each
payload parses cleanly into the contract model (with tracing /
initiator fields ignored and the model frozen), each handler delegates
to its core seam with the parsed arguments, and the un-injected /
unimplemented seams raise so a miswired worker fails loudly. The core
dispatch is exercised through patched core seams, so no real database,
session or vector store is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.knowledge_bases.delete import KBDeleteResult
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import index_delete as index_delete_module
from src.workers.tasks import kb_clone as kb_clone_module
from src.workers.tasks import kb_delete as kb_delete_module
from src.workers.tasks.index_delete import (
    IndexDeletePayload,
    process_index_delete,
    task_index_delete,
)
from src.workers.tasks.index_delete import (
    RetrieverEngineParams as IndexRetrieverEngineParams,
)
from src.workers.tasks.kb_clone import (
    KBClonePayload,
    process_kb_clone,
    task_kb_clone,
)
from src.workers.tasks.kb_clone import (
    parse_payload as parse_kb_clone_payload,
)
from src.workers.tasks.kb_delete import (
    ChunkDeleteRepo,
    KBDeletePayload,
    KnowledgeDeleteRepo,
    process_kb_delete,
    task_kb_delete,
)
from src.workers.tasks.kb_delete import (
    parse_payload as parse_kb_delete_payload,
)

KB_CLONE_TASK = "kb:clone"
KB_DELETE_TASK = "kb:delete"
INDEX_DELETE_TASK = "index:delete"


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture
def ctx() -> WorkerContext:
    """Worker context for ad-hoc task invocations."""
    return make_ctx()


def _kb_info(kb_id: str) -> KnowledgeBaseInfo:
    """A minimal knowledge-base projection for the core clone return."""
    return KnowledgeBaseInfo(
        id=kb_id,
        name="kb",
        tenant_id=7,
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        updated_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


@pytest.fixture
def kb_clone_payload() -> dict[str, object]:
    """A representative JSON payload for the kb-clone task."""
    return {
        "tenant_id": 42,
        "task_id": "clone-1",
        "source_id": "kb-src",
        "target_id": "kb-dst",
    }


@pytest.fixture
def kb_delete_payload() -> dict[str, object]:
    """A representative JSON payload for the kb-delete task."""
    return {
        "tenant_id": 42,
        "knowledge_base_id": "kb-1",
        "data_source_ids": ["ds-1", "ds-2"],
        "effective_engines": [
            {"retriever_engine_type": "elasticsearch", "retriever_type": "vector"},
            {"retriever_engine_type": "elasticsearch", "retriever_type": "keywords"},
        ],
        "vector_store_id": "store-1",
    }


@pytest.fixture
def index_delete_payload() -> dict[str, object]:
    """A representative JSON payload for the index-delete task."""
    return {
        "tenant_id": 42,
        "knowledge_base_id": "kb-1",
        "embedding_model_id": "model-1",
        "kb_type": "document",
        "chunk_ids": ["c-1", "c-2", "c-3"],
        "effective_engines": [
            {"retriever_engine_type": "milvus", "retriever_type": "vector"},
        ],
        "vector_store_id": "store-1",
    }


# ══ kb:clone ═════════════════════════════════════════════════════════


def test_kb_clone_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task(KB_CLONE_TASK) is task_kb_clone


def test_kb_clone_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("kb:clon") is None


def test_kb_clone_payload_parses_full(kb_clone_payload: dict[str, object]) -> None:
    """A complete payload round-trips through the contract model."""
    payload = parse_kb_clone_payload(cast("dict[str, Any]", kb_clone_payload))
    assert payload.tenant_id == 42
    assert payload.task_id == "clone-1"
    assert payload.source_id == "kb-src"
    assert payload.target_id == "kb-dst"


def test_kb_clone_payload_ignores_initiator_and_tracing_fields() -> None:
    """Initiator / tracing fields are accepted but not modelled."""
    payload = KBClonePayload.model_validate(
        {
            "tenant_id": 1,
            "task_id": "t-1",
            "source_id": "s-1",
            "target_id": "d-1",
            "initiator": {"user_id": "u-1", "role": "admin"},
            "lf_traceparent": "00-abc-def-01",
        }
    )
    assert payload.tenant_id == 1
    assert payload.task_id == "t-1"


def test_kb_clone_payload_rejects_missing_task_id() -> None:
    """The task id is mandatory."""
    with pytest.raises(ValidationError):
        KBClonePayload.model_validate(
            {"tenant_id": 1, "source_id": "s-1", "target_id": "d-1"}
        )


def test_kb_clone_payload_rejects_missing_source_id() -> None:
    """The source id is mandatory."""
    with pytest.raises(ValidationError):
        KBClonePayload.model_validate(
            {"tenant_id": 1, "task_id": "t-1", "target_id": "d-1"}
        )


def test_kb_clone_payload_rejects_missing_target_id() -> None:
    """The target id is mandatory."""
    with pytest.raises(ValidationError):
        KBClonePayload.model_validate(
            {"tenant_id": 1, "task_id": "t-1", "source_id": "s-1"}
        )


def test_kb_clone_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = KBClonePayload.model_validate(
        {"tenant_id": 1, "task_id": "t-1", "source_id": "s-1", "target_id": "d-1"}
    )
    with pytest.raises(ValidationError):
        payload.task_id = "tampered"


async def test_task_kb_clone_delegates_to_core_seam(
    ctx: WorkerContext,
    kb_clone_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        kb_clone_module,
        "process_kb_clone",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        result = await task_kb_clone(ctx, **kb_clone_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        task_id="clone-1",
        source_kb_id="kb-src",
        target_kb_id="kb-dst",
        service=None,
        session=None,
    )
    assert result == {"status": "completed"}


async def test_task_kb_clone_rejects_invalid_payload(ctx: WorkerContext) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_kb_clone(ctx, tenant_id=1)  # type: ignore[arg-type]


async def test_task_kb_clone_propagates_core_errors(
    ctx: WorkerContext,
    kb_clone_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            kb_clone_module,
            "process_kb_clone",
            new_callable=AsyncMock,
            side_effect=RuntimeError("clone exploded"),
        ),
        pytest.raises(RuntimeError, match="clone exploded"),
    ):
        await task_kb_clone(ctx, **kb_clone_payload)  # type: ignore[arg-type]


async def test_process_kb_clone_raises_without_service() -> None:
    """An uninjected service raises so a miswired clone is never silent."""
    with pytest.raises(NotImplementedError, match="KBService"):
        await process_kb_clone(
            tenant_id=1,
            task_id="t-1",
            source_kb_id="s-1",
            target_kb_id="d-1",
            service=None,
            session=cast(AsyncSession, AsyncMock()),
        )


async def test_process_kb_clone_raises_without_session() -> None:
    """An uninjected session raises so a miswired clone is never silent."""
    with pytest.raises(NotImplementedError, match="KBService"):
        await process_kb_clone(
            tenant_id=1,
            task_id="t-1",
            source_kb_id="s-1",
            target_kb_id="d-1",
            service=cast(KBService, AsyncMock(spec=KBService)),
            session=None,
        )


async def test_process_kb_clone_delegates_to_core_copy() -> None:
    """An injected service + session runs the clone and its pair is shaped."""
    mock_service = AsyncMock(spec=KBService)
    mock_session = AsyncMock(spec=AsyncSession)
    source = _kb_info("kb-src")
    target = _kb_info("kb-dst")
    with patch.object(
        kb_clone_module,
        "copy_kb",
        new_callable=AsyncMock,
        return_value=(source, target),
    ) as mock:
        result = await process_kb_clone(
            tenant_id=7,
            task_id="clone-9",
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            service=cast(KBService, mock_service),
            session=cast(AsyncSession, mock_session),
        )

    mock.assert_awaited_once_with(
        service=cast(KBService, mock_service),
        session=cast(AsyncSession, mock_session),
        tenant_id=7,
        source_kb_id="kb-src",
        target_kb_id="kb-dst",
    )
    assert result == {
        "task_id": "clone-9",
        "source_id": "kb-src",
        "target_id": "kb-dst",
        "status": "completed",
    }


# ══ kb:delete ════════════════════════════════════════════════════════


def test_kb_delete_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task(KB_DELETE_TASK) is task_kb_delete


def test_kb_delete_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("kb:delet") is None


def test_kb_delete_payload_parses_full(kb_delete_payload: dict[str, object]) -> None:
    """A complete payload round-trips through the contract model."""
    payload = parse_kb_delete_payload(cast("dict[str, Any]", kb_delete_payload))
    assert payload.tenant_id == 42
    assert payload.knowledge_base_id == "kb-1"
    assert payload.data_source_ids == ["ds-1", "ds-2"]
    assert payload.vector_store_id == "store-1"
    assert len(payload.effective_engines) == 2
    first = payload.effective_engines[0]
    assert first.retriever_engine_type == "elasticsearch"
    assert first.retriever_type == "vector"


def test_kb_delete_payload_defaults_optional_fields() -> None:
    """Omitted optional fields fall back to their no-op values."""
    payload = KBDeletePayload.model_validate(
        {"tenant_id": 1, "knowledge_base_id": "kb-1"}
    )
    assert payload.data_source_ids == []
    assert payload.effective_engines == []
    assert payload.vector_store_id is None


def test_kb_delete_payload_accepts_unknown_engine_type() -> None:
    """An engine type the core enum has not caught up with still parses."""
    payload = KBDeletePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "effective_engines": [
                {"retriever_engine_type": "future_engine", "retriever_type": "vector"},
            ],
        }
    )
    assert payload.effective_engines[0].retriever_engine_type == "future_engine"


def test_kb_delete_payload_ignores_tracing_fields() -> None:
    """Tracing fields are accepted but not modelled."""
    payload = KBDeletePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
        }
    )
    assert payload.tenant_id == 1
    assert payload.knowledge_base_id == "kb-1"


def test_kb_delete_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge-base id is mandatory."""
    with pytest.raises(ValidationError):
        KBDeletePayload.model_validate({"tenant_id": 1})


def test_kb_delete_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = KBDeletePayload.model_validate(
        {"tenant_id": 1, "knowledge_base_id": "kb-1"}
    )
    with pytest.raises(ValidationError):
        payload.knowledge_base_id = "tampered"


async def test_task_kb_delete_delegates_to_core_seam(
    ctx: WorkerContext,
    kb_delete_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        kb_delete_module,
        "process_kb_delete",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        result = await task_kb_delete(ctx, **kb_delete_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_base_id="kb-1",
        data_source_ids=["ds-1", "ds-2"],
        vector_store_id="store-1",
        knowledge_repo=None,
        chunk_repo=None,
    )
    assert result == {"status": "completed"}


async def test_task_kb_delete_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted optional fields fall through to the core seam as no-ops."""
    with patch.object(
        kb_delete_module,
        "process_kb_delete",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        await task_kb_delete(  # type: ignore[call-arg]
            ctx,
            tenant_id=1,
            knowledge_base_id="kb-1",
        )

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_base_id="kb-1",
        data_source_ids=[],
        vector_store_id=None,
        knowledge_repo=None,
        chunk_repo=None,
    )


async def test_task_kb_delete_rejects_invalid_payload(ctx: WorkerContext) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_kb_delete(ctx, tenant_id=1)  # type: ignore[arg-type]


async def test_task_kb_delete_propagates_core_errors(
    ctx: WorkerContext,
    kb_delete_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            kb_delete_module,
            "process_kb_delete",
            new_callable=AsyncMock,
            side_effect=RuntimeError("delete exploded"),
        ),
        pytest.raises(RuntimeError, match="delete exploded"),
    ):
        await task_kb_delete(ctx, **kb_delete_payload)  # type: ignore[arg-type]


async def test_process_kb_delete_raises_without_repos() -> None:
    """An uninjected repository pair raises so a miswired delete is never silent."""
    with pytest.raises(NotImplementedError, match="repositories"):
        await process_kb_delete(
            tenant_id=1,
            knowledge_base_id="kb-1",
            data_source_ids=(),
            vector_store_id=None,
            knowledge_repo=None,
            chunk_repo=None,
        )


async def test_process_kb_delete_delegates_to_core_cascade() -> None:
    """An injected repo pair runs the cascade and its summary is shaped."""
    knowledge_repo = AsyncMock(spec=KnowledgeRepository)
    chunk_repo = AsyncMock(spec=ChunkRepository)
    with patch.object(
        kb_delete_module,
        "_core_process_kb_delete",
        new_callable=AsyncMock,
        return_value=KBDeleteResult(
            knowledge_ids=("doc-1", "doc-2"),
            deleted_chunks=4,
            deleted_knowledge=2,
            vector_store_id="store-1",
        ),
    ) as mock:
        result = await process_kb_delete(
            tenant_id=7,
            knowledge_base_id="kb-1",
            data_source_ids=["ds-1"],
            vector_store_id="store-1",
            knowledge_repo=cast(KnowledgeDeleteRepo, knowledge_repo),
            chunk_repo=cast(ChunkDeleteRepo, chunk_repo),
        )

    mock.assert_awaited_once_with(
        tenant_id=7,
        knowledge_base_id="kb-1",
        knowledge_repo=cast(KnowledgeDeleteRepo, knowledge_repo),
        chunk_repo=cast(ChunkDeleteRepo, chunk_repo),
        vector_store_id="store-1",
    )
    assert result == {
        "knowledge_base_id": "kb-1",
        "knowledge_ids": ["doc-1", "doc-2"],
        "deleted_chunks": 4,
        "deleted_knowledge": 2,
        "vector_store_id": "store-1",
        "status": "completed",
    }


# ══ index:delete ═════════════════════════════════════════════════════


def test_index_delete_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task(INDEX_DELETE_TASK) is task_index_delete


def test_index_delete_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("index:delet") is None


def test_index_delete_payload_parses_full(
    index_delete_payload: dict[str, object],
) -> None:
    """A complete payload round-trips through the contract model."""
    payload = IndexDeletePayload.model_validate(index_delete_payload)
    assert payload.tenant_id == 42
    assert payload.knowledge_base_id == "kb-1"
    assert payload.embedding_model_id == "model-1"
    assert payload.kb_type == "document"
    assert payload.chunk_ids == ["c-1", "c-2", "c-3"]
    assert payload.vector_store_id == "store-1"
    assert payload.effective_engines[0].retriever_engine_type == "milvus"
    assert payload.effective_engines[0].retriever_type == "vector"


def test_index_delete_payload_defaults_optional_fields() -> None:
    """Omitted optional fields fall back to their no-op values."""
    payload = IndexDeletePayload.model_validate(
        {"tenant_id": 1, "knowledge_base_id": "kb-1", "embedding_model_id": "m-1"}
    )
    assert payload.kb_type == ""
    assert payload.chunk_ids == []
    assert payload.effective_engines == []
    assert payload.vector_store_id is None


def test_index_delete_payload_ignores_tracing_fields() -> None:
    """Tracing fields are accepted but not modelled."""
    payload = IndexDeletePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "embedding_model_id": "m-1",
            "lf_traceparent": "00-abc-def-01",
        }
    )
    assert payload.tenant_id == 1
    assert payload.embedding_model_id == "m-1"


def test_index_delete_payload_rejects_missing_embedding_model_id() -> None:
    """The embedding-model id is mandatory."""
    with pytest.raises(ValidationError):
        IndexDeletePayload.model_validate(
            {"tenant_id": 1, "knowledge_base_id": "kb-1"}
        )


def test_index_delete_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = IndexDeletePayload.model_validate(
        {"tenant_id": 1, "knowledge_base_id": "kb-1", "embedding_model_id": "m-1"}
    )
    with pytest.raises(ValidationError):
        payload.chunk_ids = ["tampered"]


async def test_task_index_delete_delegates_to_core_seam(
    ctx: WorkerContext,
    index_delete_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        index_delete_module,
        "process_index_delete",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        result = await task_index_delete(ctx, **index_delete_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_base_id="kb-1",
        embedding_model_id="model-1",
        kb_type="document",
        chunk_ids=["c-1", "c-2", "c-3"],
        effective_engines=[
            IndexRetrieverEngineParams(
                retriever_engine_type="milvus",
                retriever_type="vector",
            ),
        ],
        vector_store_id="store-1",
    )
    assert result == {"status": "completed"}


async def test_task_index_delete_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted optional fields fall through to the core seam as no-ops."""
    with patch.object(
        index_delete_module,
        "process_index_delete",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        await task_index_delete(  # type: ignore[call-arg]
            ctx,
            tenant_id=1,
            knowledge_base_id="kb-1",
            embedding_model_id="m-1",
        )

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_base_id="kb-1",
        embedding_model_id="m-1",
        kb_type="",
        chunk_ids=[],
        effective_engines=[],
        vector_store_id=None,
    )


async def test_task_index_delete_rejects_invalid_payload(ctx: WorkerContext) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_index_delete(ctx, tenant_id=1)  # type: ignore[arg-type]


async def test_task_index_delete_propagates_core_errors(
    ctx: WorkerContext,
    index_delete_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            index_delete_module,
            "process_index_delete",
            new_callable=AsyncMock,
            side_effect=RuntimeError("index cleanup exploded"),
        ),
        pytest.raises(RuntimeError, match="index cleanup exploded"),
    ):
        await task_index_delete(ctx, **index_delete_payload)  # type: ignore[arg-type]


async def test_process_index_delete_raises_not_implemented() -> None:
    """The core seam is a placeholder until the core composition lands."""
    with pytest.raises(NotImplementedError):
        await process_index_delete(
            tenant_id=1,
            knowledge_base_id="kb-1",
            embedding_model_id="m-1",
            kb_type="document",
            chunk_ids=["c-1"],
            effective_engines=[],
            vector_store_id=None,
        )


# ── Re-registration guard ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the modules leave the registrations in place, but a
    future test that re-registers under the same names would silently
    overwrite them. The fixture is defensive — a no-op today.
    """
    from src.workers import registry as registry_module

    snapshot = dict(registry_module.all_tasks())
    yield
    # Restore the snapshot — drop any entries added during the test.
    current = registry_module.all_tasks()
    for name in current.keys() - snapshot.keys():
        current.pop(name, None)
    for name, handler in snapshot.items():
        current[name] = handler
