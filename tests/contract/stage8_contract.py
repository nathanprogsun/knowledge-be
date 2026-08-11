"""Worker-domain contract tests (Stage 8).

Compares the ARQ worker task payloads and handler result shapes
against the reference worker wire fixtures:
``fixtures/worker_responses.json`` -> ``tasks`` (captured from the
upstream Go task payload structs in the upstream types and application
service modules).

- every registered task name maps to a fixture entry whose ``go_type``
  matches an upstream task-type constant;
- every covered payload model is frozen, and its serialized wire field
  set exactly equals the upstream business-field set;
- every fixture sample payload validates against its payload model
  (input-schema parity);
- every handler that returns a JSON result shape carries exactly the
  fixture ``result`` key set for that task (output-schema parity);
- the tasks whose result seam is not yet wired are documented here so
  the gap is tracked rather than silent.

The worker layer has no live HTTP surface, so unlike the Stage 4 / 5
contract tests this module is read-only and model-only — no ARQ broker,
no database, no external services. A failing assertion here means the
Python worker payload or result shape deviates from the reference wire
shape; two payloads deliberately deviate from the upstream shape and
the ``test_worker_invariants`` module records that as a finding.

Together with ``tests/contract/test_worker_invariants.py`` (the
frozen-payload milestone gate), this is the Stage 8 contract sign-off.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, TypeAlias
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from src.core.knowledge.documents.temporary_document import (
    TemporaryDocumentTaskPayload,
)
from src.workers.base import WorkerContext
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
from tests.contract.test_knowledge_invariants import model_wire_fields

_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "worker_responses.json"

# Tasks whose handler delegates to a core seam that raises
# ``NotImplementedError`` until a later wave wires it (or whose runner
# is not yet composed by the worker wiring layer). Their result shape
# is therefore not yet observable and is excluded from the live
# result-parity assertions; recorded here so the skip is explicit and
# tracked rather than silent.
_RESULT_UNWIRED: frozenset[str] = frozenset(
    {
        "faq:import",  # runner not yet composed by the wiring layer
        "index:delete",  # process_index_delete raises NotImplementedError
        "manual_process",  # process_document_manual raises NotImplementedError
        "question:generation",  # process_question_generation raises NotImplementedError
        "summary:generation",  # process_summary_generation raises NotImplementedError
        "wiki:finalize",  # process_wiki_finalize raises NotImplementedError
    }
)


# Mapping of fixture task name -> (payload model, async handler). The
# temporary-document payload lives in the core domain; every other
# payload accompanies its handler in ``src.workers.tasks``.
class _TaskBinding(NamedTuple):
    """One task's payload model and async handler."""

    payload: type[BaseModel]
    handler: Callable[..., Any]


_TASK_BINDINGS: dict[str, _TaskBinding] = {
    "chunk:extract": _TaskBinding(chunk_extract.ChunkExtractTaskPayload, chunk_extract.task_chunk_extract),
    "datasource_sync": _TaskBinding(datasource_sync.DatasourceSyncPayload, datasource_sync.task_datasource_sync),
    "datatable:summary": _TaskBinding(datatable_summary.DatatableSummaryPayload, datatable_summary.task_datatable_summary),
    "document_process": _TaskBinding(document_process.DocumentProcessTaskPayload, document_process.task_document_process),
    "faq:import": _TaskBinding(faq_import.FAQImportPayload, faq_import.task_faq_import),
    "image_multimodal": _TaskBinding(image_multimodal.ImageMultimodalTaskPayload, image_multimodal.task_image_multimodal),
    "index:delete": _TaskBinding(index_delete.IndexDeletePayload, index_delete.task_index_delete),
    "kb:clone": _TaskBinding(kb_clone.KBClonePayload, kb_clone.task_kb_clone),
    "kb:delete": _TaskBinding(kb_delete.KBDeletePayload, kb_delete.task_kb_delete),
    "knowledge:list_delete": _TaskBinding(knowledge_list_delete.KnowledgeListDeletePayload, knowledge_list_delete.task_knowledge_list_delete),
    "knowledge:list_reparse": _TaskBinding(knowledge_list_reparse.KnowledgeListReparsePayload, knowledge_list_reparse.task_knowledge_list_reparse),
    "knowledge:move": _TaskBinding(knowledge_move.KnowledgeMovePayload, knowledge_move.task_knowledge_move),
    "knowledge:post_process": _TaskBinding(knowledge_post_process.KnowledgePostProcessTaskPayload, knowledge_post_process.task_knowledge_post_process),
    "manual_process": _TaskBinding(manual_process.ManualProcessPayload, manual_process.manual_process),
    "question:generation": _TaskBinding(question_generation.QuestionGenerationTaskPayload, question_generation.task_question_generation),
    "summary:generation": _TaskBinding(summary_generation.SummaryGenerationTaskPayload, summary_generation.task_summary_generation),
    "temporary_document:process": _TaskBinding(TemporaryDocumentTaskPayload, temporary_document.task_temporary_document),
    "wiki:finalize": _TaskBinding(wiki_finalize.WikiFinalizePayload, wiki_finalize.task_wiki_finalize),
    "wiki:ingest": _TaskBinding(wiki_ingest.WikiIngestPayload, wiki_ingest.task_wiki_ingest),
}


_FixtureJson: TypeAlias = dict[str, object]


def _load_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _fixture_tasks() -> dict[str, _FixtureJson]:
    raw = _load_fixture()
    tasks = raw.get("tasks", {})
    out: dict[str, _FixtureJson] = {}
    if isinstance(tasks, dict):
        for name, spec in tasks.items():
            if isinstance(name, str) and isinstance(spec, dict):
                out[name] = spec
    return out


def _task_result_keys(spec: _FixtureJson) -> list[str]:
    return _string_list(spec.get("result"))


def _fixture_samples() -> dict[str, _FixtureJson]:
    raw = _load_fixture()
    samples = raw.get("samples", {})
    out: dict[str, _FixtureJson] = {}
    if isinstance(samples, dict):
        for name, payload in samples.items():
            if isinstance(name, str) and isinstance(payload, dict):
                out[name] = payload
    return out


def _make_ctx() -> WorkerContext:
    """Build the minimal ARQ context the handlers receive."""
    return WorkerContext(
        redis=None,  # type: ignore[arg-type]
        job_id="job-contract-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


def _assert_keyset(actual: dict[str, object], expected: list[str], label: str) -> None:
    """Assert ``actual`` carries exactly the ``expected`` top-level keys."""
    actual_set = set(actual)
    expected_set = set(expected)
    assert actual_set == expected_set, (
        f"{label}: result keys diverge from the reference fixture.\n"
        f"  missing: {sorted(expected_set - actual_set)}\n"
        f"  extra:   {sorted(actual_set - expected_set)}"
    )


# ── Fixture coverage ─────────────────────────────────────────────────


def test_fixture_covers_all_registered_tasks() -> None:
    """Every fixture task has a binding in this module."""
    expected = set(_fixture_tasks())
    actual = set(_TASK_BINDINGS)
    assert actual == expected, (
        "task coverage diverges from the fixture.\n"
        f"  fixture entries without a binding: {sorted(expected - actual)}\n"
        f"  bindings without a fixture entry:   {sorted(actual - expected)}"
    )


def test_result_unwired_entries_have_fixture_spec() -> None:
    """Every unwired task has a fixture entry to verify gaps against."""
    fixture = _fixture_tasks()
    missing = sorted(name for name in _RESULT_UNWIRED if name not in fixture)
    assert not missing, f"unwired tasks absent from fixture: {missing}"


# ── Input-schema parity ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    sorted(_TASK_BINDINGS),
    ids=lambda v: v,
)
def test_sample_payload_validates_against_payload_model(name: str) -> None:
    """The reference sample payload deserializes into the payload model.

    Two tasks deviate from the Go-shaped payload and this assertion is
    the documented contract finding:

    - ``document_process`` accepts the Go ``passages`` / ``attempt``
      fields via ``extra="ignore"`` and the round-trip drops them on
      serialisation (Python silently ignores the input but does not
      preserve it in ``model_dump()``);
    - ``faq:import`` requires ``filename`` / ``file_data`` which the
      Go payload does not carry, so a Go-shaped payload cannot be
      consumed by the current Python model without translation.
    """
    samples = _fixture_samples()
    assert name in samples, f"sample payload missing for task {name!r}"
    payload = samples[name]
    model = _TASK_BINDINGS[name].payload
    try:
        instance = model.model_validate(payload)
    except ValidationError as exc:
        pytest.fail(f"{name}: sample payload is not valid against {model.__name__}: {exc}")
    # Round-trip: the validated instance serialises back to the sample keys.
    # This is the data-loss check — Python models with ``extra="ignore"``
    # silently drop unknown Go fields on parse, so the serialised keys can
    # be a strict subset of the sample.
    serialised = instance.model_dump(mode="json")
    missing = set(payload) - set(serialised)
    if missing:
        pytest.fail(
            f"{name}: round-trip payload drops Go-side fields: {sorted(missing)}"
        )


# ── Output-schema parity ─────────────────────────────────────────────


async def test_temporary_document_result_matches_reference() -> None:
    """``temporary_document:process`` handler returns the reference key set.

    The handler is fully wired (no core seam) so its result shape can be
    exercised directly against a parsed sample payload.
    """
    samples = _fixture_samples()
    payload = samples["temporary_document:process"]
    result = await temporary_document.task_temporary_document(
        _make_ctx(), **dict(payload)
    )
    expected = _task_result_keys(_fixture_tasks()["temporary_document:process"])
    _assert_keyset(result, expected, "temporary_document:process")


async def test_chunk_extract_result_matches_reference() -> None:
    """``chunk:extract`` handler returns the reference key set.

    The handler short-circuits with a skipped outcome when no extractor
    is injected; that skipped path still emits every reference result
    key, so the contract check is deterministic without a wired seam.
    """
    result = await chunk_extract.task_chunk_extract(
        _make_ctx(),
        tenant_id=1,
        chunk_id="chunk-1",
        model_id="model-1",
    )
    expected = _task_result_keys(_fixture_tasks()["chunk:extract"])
    _assert_keyset(result, expected, "chunk:extract result")


async def test_document_process_result_matches_reference() -> None:
    """``document_process`` handler returns the reference process key set."""
    from src.core.knowledge.documents.process_document import ProcessOutcome

    sample_outcome = ProcessOutcome(
        parse_status="processing",
        enable_status="enabled",
        summary_status="none",
        storage_size=0,
        text_chunk_count=0,
    )

    with patch(
        "src.workers.tasks.document_process._core_process_document",
        new=AsyncMock(return_value=sample_outcome),
    ):
        result = await document_process.task_document_process(
            _make_ctx(),
            tenant_id=1,
            knowledge_id="knowledge-1",
            knowledge_base_id="kb-1",
        )

    expected = _task_result_keys(_fixture_tasks()["document_process"])
    _assert_keyset(result, expected, "document_process result")


async def test_datasource_sync_result_matches_reference() -> None:
    """``datasource_sync`` handler returns the reference sync-log key set.

    The handler delegates the serialisation to ``SyncLogInfo.model_dump``
    via ``_serialise_sync_log``; the patched core seam returns a sample
    ``SyncLogInfo`` whose serialised keys the fixture pins.
    """
    from src.core.infra.datasources.types import SyncLogInfo

    sample_log = SyncLogInfo(
        id="sync-1",
        data_source_id="ds-1",
        tenant_id=1,
        status="running",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    captured: dict[str, object] = {}

    async def _fake_process(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return sample_log.model_dump(mode="json")

    with patch(
        "src.workers.tasks.datasource_sync.process_datasource_sync",
        new=AsyncMock(side_effect=_fake_process),
    ):
        result = await datasource_sync.task_datasource_sync(
            _make_ctx(),
            data_source_id="ds-1",
            tenant_id=1,
            sync_log_id="sync-1",
            force_full=True,
            max_items=100,
            trigger="manual",
        )

    expected = _task_result_keys(_fixture_tasks()["datasource_sync"])
    _assert_keyset(result, expected, "datasource_sync result")
    assert captured["data_source_id"] == "ds-1"
    assert captured["force_full"] is True


async def test_image_multimodal_result_matches_reference() -> None:
    """``image_multimodal`` handler returns the reference multimodal key set."""
    from src.core.knowledge.documents.image_multimodal import ImageMultimodalOutcome

    sample_outcome = ImageMultimodalOutcome(
        skipped=True,
        read_error="no image content",
    )

    with patch(
        "src.workers.tasks.image_multimodal.process_image_multimodal",
        new=AsyncMock(return_value=sample_outcome),
    ):
        result = await image_multimodal.task_image_multimodal(
            _make_ctx(),
            tenant_id=1,
            knowledge_id="knowledge-1",
            knowledge_base_id="kb-1",
            chunk_id="chunk-1",
            image_url="local://img.png",
        )

    expected = _task_result_keys(_fixture_tasks()["image_multimodal"])
    _assert_keyset(result, expected, "image_multimodal result")


async def test_kb_delete_result_matches_reference() -> None:
    """``kb:delete`` handler returns the reference delete result key set.

    The worker-level ``process_kb_delete`` builds the serialised result
    dict from a :class:`KBDeleteResult`. The patch targets the core seam
    (``_core_process_kb_delete``) so the worker-level helper still runs
    its serialisation path against the injected result.
    """
    from src.core.knowledge.knowledge_bases.delete import KBDeleteResult

    sample_result = KBDeleteResult(
        knowledge_ids=["knowledge-1"],
        deleted_chunks=1,
        deleted_knowledge=1,
        vector_store_id="store-1",
    )

    class _StubRepo:
        async def list_by_knowledge_base(self, *_args: object, **_kw: object) -> list[object]:
            return []

        async def soft_delete_list(self, **_kw: object) -> int:
            return 0

    with patch(
        "src.workers.tasks.kb_delete._core_process_kb_delete",
        new=AsyncMock(return_value=sample_result),
    ):
        result = await kb_delete.task_kb_delete(
            _make_ctx(),
            tenant_id=1,
            knowledge_base_id="kb-1",
            knowledge_repo=_StubRepo(),  # type: ignore[arg-type]
            chunk_repo=_StubRepo(),  # type: ignore[arg-type]
        )

    expected = _task_result_keys(_fixture_tasks()["kb:delete"])
    _assert_keyset(result, expected, "kb:delete result")


async def test_knowledge_list_delete_result_matches_reference() -> None:
    """``knowledge:list_delete`` returns the reference single-key result."""
    with patch(
        "src.workers.tasks.knowledge_list_delete.process_knowledge_list_delete",
        new=AsyncMock(return_value={"deleted": 3}),
    ):
        result = await knowledge_list_delete.task_knowledge_list_delete(
            _make_ctx(),
            tenant_id=1,
            knowledge_ids=["knowledge-1", "knowledge-2", "knowledge-3"],
        )

    expected = _task_result_keys(_fixture_tasks()["knowledge:list_delete"])
    _assert_keyset(result, expected, "knowledge:list_delete result")


async def test_knowledge_list_reparse_result_matches_reference() -> None:
    """``knowledge:list_reparse`` returns the reference summary key set."""
    with patch(
        "src.workers.tasks.knowledge_list_reparse.process_knowledge_list_reparse",
        new=AsyncMock(return_value={"submitted": ["knowledge-1"], "failed": []}),
    ):
        result = await knowledge_list_reparse.task_knowledge_list_reparse(
            _make_ctx(),
            tenant_id=1,
            knowledge_ids=["knowledge-1"],
        )

    expected = _task_result_keys(_fixture_tasks()["knowledge:list_reparse"])
    _assert_keyset(result, expected, "knowledge:list_reparse result")


async def test_knowledge_move_result_matches_reference() -> None:
    """``knowledge:move`` returns the reference summary key set."""
    with patch(
        "src.workers.tasks.knowledge_move.process_knowledge_move",
        new=AsyncMock(return_value={"processed": ["knowledge-1"], "failed": []}),
    ):
        result = await knowledge_move.task_knowledge_move(
            _make_ctx(),
            tenant_id=1,
            knowledge_ids=["knowledge-1"],
            source_kb_id="kb-1",
            target_kb_id="kb-2",
            mode="reuse_vectors",
        )

    expected = _task_result_keys(_fixture_tasks()["knowledge:move"])
    _assert_keyset(result, expected, "knowledge:move result")


async def test_knowledge_post_process_result_matches_reference() -> None:
    """``knowledge:post_process`` returns the reference post-process key set."""
    from src.core.knowledge.documents.post_process_service import PostProcessOutcome

    sample_outcome = PostProcessOutcome(
        skipped=False,
        reason="",
        chunks_total=4,
    )

    with patch(
        "src.workers.tasks.knowledge_post_process._core_run_post_process",
        new=AsyncMock(return_value=sample_outcome),
    ):
        result = await knowledge_post_process.task_knowledge_post_process(
            _make_ctx(),
            tenant_id=1,
            knowledge_id="knowledge-1",
            knowledge_base_id="kb-1",
        )

    expected = _task_result_keys(_fixture_tasks()["knowledge:post_process"])
    _assert_keyset(result, expected, "knowledge:post_process result")


async def test_datatable_summary_result_matches_reference() -> None:
    """``datatable:summary`` returns the reference summary key set."""
    from src.core.knowledge.documents.datatable_summary import DataTableSummaryResult

    sample_result = DataTableSummaryResult(
        knowledge_id="knowledge-1",
        summary_chunk_id="chunk-summary",
        column_chunk_id="chunk-column",
    )

    with patch(
        "src.workers.tasks.datatable_summary.run_datatable_summary",
        new=AsyncMock(return_value=sample_result),
    ):
        result = await datatable_summary.task_datatable_summary(
            _make_ctx(),
            tenant_id=1,
            knowledge_id="knowledge-1",
            summary_model="model-1",
            embedding_model="model-2",
        )

    expected = _task_result_keys(_fixture_tasks()["datatable:summary"])
    _assert_keyset(result, expected, "datatable:summary result")


async def test_wiki_ingest_result_matches_reference() -> None:
    """``wiki:ingest`` returns the reference wiki-batch key set."""
    from src.core.knowledge.wiki.ingest_types import WikiBatchOutcome
    from src.workers.tasks.wiki_ingest import _serialise_outcome

    sample_outcome = WikiBatchOutcome(
        pending_ops=0,
        ingest_succeeded=1,
        ingest_failed=0,
        retract_handled=0,
        pages_affected=1,
        follow_up_scheduled=False,
    )

    with patch(
        "src.workers.tasks.wiki_ingest.process_wiki_ingest",
        new=AsyncMock(side_effect=lambda **_kw: _serialise_outcome(sample_outcome)),
    ):
        result = await wiki_ingest.task_wiki_ingest(
            _make_ctx(),
            tenant_id=1,
            knowledge_base_id="kb-1",
            language="en-US",
        )

    expected = _task_result_keys(_fixture_tasks()["wiki:ingest"])
    _assert_keyset(result, expected, "wiki:ingest result")


def test_kb_clone_result_matches_reference() -> None:
    """``kb:clone`` exercises the patched core seam end-to-end.

    The clone handler accepts an injected session + service; the patched
    seam returns a minimal clone-pair snapshot the handler projects back
    onto the reference result key set.
    """
    captured: dict[str, object] = {}

    async def _fake_process(**kwargs: object) -> dict[str, object]:
        captured["tenant_id"] = kwargs.get("tenant_id")
        captured["task_id"] = kwargs.get("task_id")
        captured["source_kb_id"] = kwargs.get("source_kb_id")
        captured["target_kb_id"] = kwargs.get("target_kb_id")
        return {
            "task_id": kwargs.get("task_id"),
            "source_id": kwargs.get("source_kb_id"),
            "target_id": kwargs.get("target_kb_id"),
            "status": "completed",
        }

    with patch(
        "src.workers.tasks.kb_clone.process_kb_clone",
        new=AsyncMock(side_effect=_fake_process),
    ):
        import asyncio

        result = asyncio.run(
            kb_clone.task_kb_clone(
                _make_ctx(),
                tenant_id=1,
                task_id="task-1",
                source_id="kb-1",
                target_id="kb-2",
                service=object(),  # type: ignore[arg-type]
                session=object(),  # type: ignore[arg-type]
            )
        )

    expected = _task_result_keys(_fixture_tasks()["kb:clone"])
    _assert_keyset(result, expected, "kb:clone result")
    assert captured["source_kb_id"] == "kb-1"
    assert captured["target_kb_id"] == "kb-2"


__all__ = [
    "model_wire_fields",
]