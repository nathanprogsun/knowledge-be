"""Storage rows for the evaluation pipeline tables.

Four tables back the offline evaluation domain, mapping the upstream
``EvaluationTask`` / ``EvaluationDetail`` / ``MetricResult`` contracts to
relational rows so a task survives process restarts and can be queried
after the fact.

- ``evaluations`` — one row per evaluation task. Holds the tenant /
  dataset / model bindings and the task-level status / counters.
  ``params`` is JSONB so the run configuration (vector threshold,
  rerank model, summary config, ...) lives in the row without
  requiring a wide column.
- ``evaluation_datasets`` — one row per dataset attached to an
  evaluation. ``qa_pairs`` is JSONB so the per-row QA ground truth
  ships verbatim with the row.
- ``evaluation_runs`` — one row per single execution of an evaluation
  task. Lets a task be re-run while keeping every run's outcome.
- ``evaluation_metrics`` — one row per run's metric bundle. Retrieval
  metrics (``precision`` / ``recall`` / ``ndcg3`` / ``ndcg10`` /
  ``mrr`` / ``map``) and generation metrics (``bleu1`` / ``bleu2`` /
  ``bleu4`` / ``rouge1`` / ``rouge2`` / ``rougel``) are flattened to
  scalars so the dashboard query does not need JSONB projection.

All four tables use the same soft-delete convention as the rest of
the storage layer: reads filter ``deleted_at is null``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel

# ── Task / run status vocabulary ──────────────────────────────────────

EVAL_STATUS_PENDING = "pending"
EVAL_STATUS_RUNNING = "running"
EVAL_STATUS_SUCCESS = "success"
EVAL_STATUS_FAILED = "failed"

EVAL_STATUSES: frozenset[str] = frozenset(
    {
        EVAL_STATUS_PENDING,
        EVAL_STATUS_RUNNING,
        EVAL_STATUS_SUCCESS,
        EVAL_STATUS_FAILED,
    }
)


class Evaluation(TableModel):
    """One row of the ``evaluations`` table."""

    table: ClassVar[str] = "evaluations"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("params",)
    # ``id`` is application-assigned (a caller-minted UUID), so it takes
    # part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    dataset_id: str
    knowledge_base_id: str = ""
    chat_model_id: str = ""
    rerank_model_id: str = ""
    status: str = EVAL_STATUS_PENDING
    total: int = 0
    finished: int = 0
    error_msg: str = ""
    params: JsonObject = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class EvaluationDataset(TableModel):
    """One row of the ``evaluation_datasets`` table."""

    table: ClassVar[str] = "evaluation_datasets"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("qa_pairs",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    evaluation_id: str
    name: str = ""
    description: str = ""
    qa_pairs: JsonObject = Field(default_factory=dict)
    item_count: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class EvaluationRun(TableModel):
    """One row of the ``evaluation_runs`` table."""

    table: ClassVar[str] = "evaluation_runs"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    evaluation_id: str
    status: str = EVAL_STATUS_PENDING
    total: int = 0
    finished: int = 0
    error_msg: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class EvaluationMetric(TableModel):
    """One row of the ``evaluation_metrics`` table."""

    table: ClassVar[str] = "evaluation_metrics"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    run_id: str
    precision: float = 0.0
    recall: float = 0.0
    ndcg3: float = 0.0
    ndcg10: float = 0.0
    mrr: float = 0.0
    map: float = 0.0
    bleu1: float = 0.0
    bleu2: float = 0.0
    bleu4: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougel: float = 0.0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = [
    "EVAL_STATUSES",
    "EVAL_STATUS_FAILED",
    "EVAL_STATUS_PENDING",
    "EVAL_STATUS_RUNNING",
    "EVAL_STATUS_SUCCESS",
    "Evaluation",
    "EvaluationDataset",
    "EvaluationMetric",
    "EvaluationRun",
]
