"""Unit tests for :mod:`src.db.dao.evaluation_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) so the tenant scoping, the soft-delete filters,
and the cross-table query shapes stay pinned without a database. The
real SQL round-trip is covered by the integration suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.sql.expression import TextClause

from src.db.dao.evaluation_repository import (
    EvaluationDatasetRepository,
    EvaluationMetricRepository,
    EvaluationRepository,
    EvaluationRunRepository,
)
from src.db.models.evaluation import (
    EVAL_STATUS_PENDING,
    EVAL_STATUS_RUNNING,
    Evaluation,
    EvaluationDataset,
    EvaluationMetric,
    EvaluationRun,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Row factories ────────────────────────────────────────────────────


def _evaluation_row(
    *,
    id: str = "eval-1",
    tenant_id: int = 1,
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "tenant_id": tenant_id,
        "dataset_id": "ds-1",
        "knowledge_base_id": "kb-1",
        "chat_model_id": "chat-1",
        "rerank_model_id": "rerank-1",
        "status": EVAL_STATUS_PENDING,
        "total": 0,
        "finished": 0,
        "error_msg": "",
        "params": {},
        "started_at": None,
        "finished_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


def _dataset_row(
    *,
    id: str = "ds-row-1",
    evaluation_id: str = "eval-1",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "evaluation_id": evaluation_id,
        "name": "default",
        "description": "",
        "qa_pairs": {},
        "item_count": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


def _run_row(
    *,
    id: str = "run-1",
    evaluation_id: str = "eval-1",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "evaluation_id": evaluation_id,
        "status": EVAL_STATUS_PENDING,
        "total": 0,
        "finished": 0,
        "error_msg": "",
        "started_at": None,
        "finished_at": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


def _metric_row(
    *,
    id: str = "metric-1",
    run_id: str = "run-1",
    **overrides: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": id,
        "run_id": run_id,
        "precision": 0.0,
        "recall": 0.0,
        "ndcg3": 0.0,
        "ndcg10": 0.0,
        "mrr": 0.0,
        "map": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu4": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougel": 0.0,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


# ── Fake session ─────────────────────────────────────────────────────


class _FakeMappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None


class _FakeScalarResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar_one(self) -> int:
        return self._value


class _FakeResult:
    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
        *,
        rowcount: int = 1,
        scalar: int | None = None,
    ) -> None:
        self._rows = rows or []
        self.rowcount = rowcount
        self._scalar = scalar

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)

    def scalar_one(self) -> int:
        assert self._scalar is not None
        return self._scalar


class _FakeSession:
    """Records executed SQL and serves canned rows keyed by SQL prefix."""

    def __init__(self, rows_by_prefix: dict[str, list[dict[str, object]]] | None = None) -> None:
        self.executed: list[str] = []
        self._rows_by_prefix = rows_by_prefix or {}
        self._scalar_by_prefix: dict[str, int] = {}

    def register_scalar(self, prefix: str, value: int) -> None:
        self._scalar_by_prefix[prefix] = value

    async def execute(self, stmt: TextClause) -> _FakeResult:
        sql = stmt.text
        self.executed.append(sql)
        stripped = sql.lstrip()
        for prefix, rows in self._rows_by_prefix.items():
            if stripped.startswith(prefix):
                return _FakeResult(rows)
        for prefix, value in self._scalar_by_prefix.items():
            if stripped.startswith(prefix):
                return _FakeResult(scalar=value)
        return _FakeResult([])


def _evaluation_repo(session: _FakeSession) -> EvaluationRepository:
    return EvaluationRepository(session)  # type: ignore[arg-type]


def _dataset_repo(session: _FakeSession) -> EvaluationDatasetRepository:
    return EvaluationDatasetRepository(session)  # type: ignore[arg-type]


def _run_repo(session: _FakeSession) -> EvaluationRunRepository:
    return EvaluationRunRepository(session)  # type: ignore[arg-type]


def _metric_repo(session: _FakeSession) -> EvaluationMetricRepository:
    return EvaluationMetricRepository(session)  # type: ignore[arg-type]


def _sample_evaluation(*, id: str = "eval-1") -> Evaluation:
    return Evaluation.model_validate(_evaluation_row(id=id))


def _sample_dataset(*, id: str = "ds-row-1") -> EvaluationDataset:
    return EvaluationDataset.model_validate(_dataset_row(id=id))


def _sample_run(*, id: str = "run-1") -> EvaluationRun:
    return EvaluationRun.model_validate(_run_row(id=id))


def _sample_metric(*, id: str = "metric-1") -> EvaluationMetric:
    return EvaluationMetric.model_validate(_metric_row(id=id))


# ── insert column list ───────────────────────────────────────────────


def test_evaluation_insert_column_list_includes_caller_assigned_id() -> None:
    cols = Evaluation.insert_sql_column_list()
    assert "id" in cols
    assert "params" in cols
    assert "tenant_id" in cols


def test_evaluation_metric_insert_column_list_includes_metric_fields() -> None:
    cols = EvaluationMetric.insert_sql_column_list()
    for col in ("precision", "recall", "ndcg3", "ndcg10", "mrr", "map"):
        assert col in cols
    for col in ("bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougel"):
        assert col in cols


# ── EvaluationRepository ─────────────────────────────────────────────


async def test_evaluation_create_inserts_row() -> None:
    session = _FakeSession({"insert into evaluations": [_evaluation_row()]})
    repo = _evaluation_repo(session)

    result = await repo.create(_sample_evaluation())

    assert result.id == "eval-1"
    assert session.executed[0].lstrip().startswith("insert into evaluations")


async def test_evaluation_get_by_id_uses_pk_filter() -> None:
    session = _FakeSession({"select * from evaluations": [_evaluation_row()]})
    repo = _evaluation_repo(session)

    result = await repo.get_by_id("eval-1")

    assert result is not None
    assert result.id == "eval-1"
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert "deleted_at is null" in sql


async def test_evaluation_get_by_id_for_tenant_scopes_by_tenant() -> None:
    session = _FakeSession({"select * from evaluations": [_evaluation_row()]})
    repo = _evaluation_repo(session)

    result = await repo.get_by_id_for_tenant(id="eval-1", tenant_id=1)

    assert result is not None
    sql = session.executed[0]
    assert '"id" = :id' in sql
    assert '"tenant_id" = :tenant_id' in sql
    assert "deleted_at is null" in sql


async def test_evaluation_list_by_tenant_orders_newest_first() -> None:
    session = _FakeSession({"select * from evaluations": [_evaluation_row()]})
    repo = _evaluation_repo(session)

    result = await repo.list_by_tenant(tenant_id=1)

    assert len(result) == 1
    sql = session.executed[0]
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc, id desc" in sql
    assert "limit :limit" in sql
    assert "offset :offset" in sql


async def test_evaluation_list_by_tenant_with_status_filter() -> None:
    session = _FakeSession(
        {"select * from evaluations where tenant_id = :tenant_id and deleted_at is null and status = :status": [_evaluation_row()]}
    )
    repo = _evaluation_repo(session)

    result = await repo.list_by_tenant(tenant_id=1, status=EVAL_STATUS_RUNNING)

    assert len(result) == 1
    sql = session.executed[0]
    assert "status = :status" in sql


async def test_evaluation_count_by_tenant_emits_count() -> None:
    session = _FakeSession()
    session.register_scalar("select count(*) from evaluations", 7)
    repo = _evaluation_repo(session)

    count = await repo.count_by_tenant(tenant_id=1)

    assert count == 7
    sql = session.executed[0]
    assert "select count(*) from evaluations" in sql
    assert "tenant_id = :tenant_id" in sql
    assert "deleted_at is null" in sql


async def test_evaluation_soft_delete_marks_row_deleted() -> None:
    session = _FakeSession()
    repo = _evaluation_repo(session)

    affected = await repo.soft_delete(id="eval-1", now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "set deleted_at = :now, updated_at = :now" in sql
    assert "id = :id" in sql
    assert "deleted_at is null" in sql


async def test_evaluation_update_overwrites_mutable_columns() -> None:
    session = _FakeSession({"update evaluations": [_evaluation_row(status=EVAL_STATUS_RUNNING)]})
    repo = _evaluation_repo(session)

    result = await repo.update(_sample_evaluation())

    assert result.status == EVAL_STATUS_RUNNING
    sql = session.executed[0]
    assert '"status" = :u_status' in sql
    assert 'where "id" = :id' in sql
    assert "deleted_at is null" in sql


# ── EvaluationDatasetRepository ─────────────────────────────────────


async def test_dataset_create_inserts_row() -> None:
    session = _FakeSession({"insert into evaluation_datasets": [_dataset_row()]})
    repo = _dataset_repo(session)

    result = await repo.create(_sample_dataset())

    assert result.id == "ds-row-1"
    assert session.executed[0].lstrip().startswith("insert into evaluation_datasets")


async def test_dataset_list_by_evaluation_filters_and_orders() -> None:
    session = _FakeSession({"select * from evaluation_datasets": [_dataset_row()]})
    repo = _dataset_repo(session)

    result = await repo.list_by_evaluation("eval-1")

    assert len(result) == 1
    sql = session.executed[0]
    assert "evaluation_id = :evaluation_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc, id desc" in sql


async def test_dataset_soft_delete_marks_row_deleted() -> None:
    session = _FakeSession()
    repo = _dataset_repo(session)

    affected = await repo.soft_delete(id="ds-row-1", now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "deleted_at is null" in sql


# ── EvaluationRunRepository ─────────────────────────────────────────


async def test_run_create_inserts_row() -> None:
    session = _FakeSession({"insert into evaluation_runs": [_run_row()]})
    repo = _run_repo(session)

    result = await repo.create(_sample_run())

    assert result.id == "run-1"
    assert session.executed[0].lstrip().startswith("insert into evaluation_runs")


async def test_run_list_by_evaluation_filters_and_orders() -> None:
    session = _FakeSession({"select * from evaluation_runs": [_run_row()]})
    repo = _run_repo(session)

    result = await repo.list_by_evaluation("eval-1")

    assert len(result) == 1
    sql = session.executed[0]
    assert "evaluation_id = :evaluation_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc, id desc" in sql


async def test_run_delete_by_evaluation_returns_rowcount() -> None:
    session = _FakeSession()
    repo = _run_repo(session)

    affected = await repo.delete_by_evaluation(evaluation_id="eval-1", now=_NOW)

    assert affected == 1
    sql = session.executed[0]
    assert "evaluation_id = :evaluation_id" in sql
    assert "deleted_at is null" in sql


async def test_run_soft_delete_marks_row_deleted() -> None:
    session = _FakeSession()
    repo = _run_repo(session)

    affected = await repo.soft_delete(id="run-1", now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "id = :id" in sql


# ── EvaluationMetricRepository ─────────────────────────────────────


async def test_metric_create_inserts_row() -> None:
    session = _FakeSession({"insert into evaluation_metrics": [_metric_row()]})
    repo = _metric_repo(session)

    result = await repo.create(_sample_metric())

    assert result.id == "metric-1"
    assert session.executed[0].lstrip().startswith("insert into evaluation_metrics")


async def test_metric_list_by_run_filters_and_orders() -> None:
    session = _FakeSession({"select * from evaluation_metrics": [_metric_row()]})
    repo = _metric_repo(session)

    result = await repo.list_by_run("run-1")

    assert len(result) == 1
    sql = session.executed[0]
    assert "run_id = :run_id" in sql
    assert "deleted_at is null" in sql
    assert "order by created_at desc, id desc" in sql


async def test_metric_latest_by_run_limits_to_one() -> None:
    session = _FakeSession({"select * from evaluation_metrics": [_metric_row()]})
    repo = _metric_repo(session)

    result = await repo.latest_by_run("run-1")

    assert result is not None
    sql = session.executed[0]
    assert "run_id = :run_id" in sql
    assert "limit 1" in sql


async def test_metric_soft_delete_marks_row_deleted() -> None:
    session = _FakeSession()
    repo = _metric_repo(session)

    affected = await repo.soft_delete(id="metric-1", now=_NOW)

    assert affected is True
    sql = session.executed[0]
    assert "deleted_at is null" in sql
