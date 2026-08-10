"""Create the evaluation pipeline tables.

Four tables back the offline evaluation domain, mirroring the upstream
``EvaluationTask`` / ``EvaluationDetail`` / ``MetricResult`` contracts:

- ``evaluations`` — one row per evaluation task. ``id`` is a caller-
  assigned UUID; ``tenant_id`` scopes the row to one workspace;
  ``params`` is JSONB so the run configuration (vector threshold,
  rerank model, summary config, ...) lives in the row without a wide
  column. ``dataset_id`` is the upstream dataset identifier.
- ``evaluation_datasets`` — one row per dataset attached to an
  evaluation. ``qa_pairs`` is JSONB so the per-row QA ground truth
  ships verbatim with the row. ``evaluation_id`` references the
  parent evaluation.
- ``evaluation_runs`` — one row per single execution of an evaluation
  task. A task may be re-run, and each run keeps its own
  outcome. ``evaluation_id`` references the parent evaluation.
- ``evaluation_metrics`` — one row per run's metric bundle. Retrieval
  metrics (``precision`` / ``recall`` / ``ndcg3`` / ``ndcg10`` / ``mrr``
  / ``map``) and generation metrics (``bleu1`` / ``bleu2`` / ``bleu4``
  / ``rouge1`` / ``rouge2`` / ``rougel``) are flattened to scalars so
  the dashboard query does not need JSONB projection. ``run_id``
  references the parent run.

Indexes mirror the query shapes: the per-tenant task list, the
per-evaluation dataset / run lists, and the per-run metrics read.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0033_evaluation"
down_revision: str | None = "0032_embed_channels"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_params_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_qa_pairs_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)

_LIVE_ROW = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "evaluations",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column(
            "knowledge_base_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "chat_model_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "rerank_model_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "total",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "finished",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "error_msg",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "params",
            _params_json,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_evaluations_tenant",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_evaluations_tenant_id", "evaluations", ["tenant_id"])
    op.create_index("idx_evaluations_dataset_id", "evaluations", ["dataset_id"])
    op.create_index("idx_evaluations_status", "evaluations", ["status"])
    op.create_index(
        "idx_evaluations_deleted_at",
        "evaluations",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )

    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("evaluation_id", sa.String(length=36), nullable=False),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "qa_pairs",
            _qa_pairs_json,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "item_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_id"],
            ["evaluations.id"],
            name="fk_evaluation_datasets_evaluation",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_evaluation_datasets_evaluation_id",
        "evaluation_datasets",
        ["evaluation_id"],
    )
    op.create_index(
        "idx_evaluation_datasets_deleted_at",
        "evaluation_datasets",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("evaluation_id", sa.String(length=36), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "total",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "finished",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "error_msg",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_id"],
            ["evaluations.id"],
            name="fk_evaluation_runs_evaluation",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_evaluation_runs_evaluation_id",
        "evaluation_runs",
        ["evaluation_id"],
    )
    op.create_index("idx_evaluation_runs_status", "evaluation_runs", ["status"])
    op.create_index(
        "idx_evaluation_runs_deleted_at",
        "evaluation_runs",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )

    op.create_table(
        "evaluation_metrics",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column(
            "precision",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "recall",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "ndcg3",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "ndcg10",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "mrr",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "map",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "bleu1",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "bleu2",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "bleu4",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "rouge1",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "rouge2",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "rougel",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["evaluation_runs.id"],
            name="fk_evaluation_metrics_run",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_evaluation_metrics_run_id",
        "evaluation_metrics",
        ["run_id"],
    )
    op.create_index(
        "idx_evaluation_metrics_deleted_at",
        "evaluation_metrics",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("idx_evaluation_metrics_deleted_at", table_name="evaluation_metrics")
    op.drop_index("idx_evaluation_metrics_run_id", table_name="evaluation_metrics")
    op.drop_table("evaluation_metrics")
    op.drop_index("idx_evaluation_runs_deleted_at", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_status", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_evaluation_id", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index("idx_evaluation_datasets_deleted_at", table_name="evaluation_datasets")
    op.drop_index(
        "idx_evaluation_datasets_evaluation_id",
        table_name="evaluation_datasets",
    )
    op.drop_table("evaluation_datasets")
    op.drop_index("idx_evaluations_deleted_at", table_name="evaluations")
    op.drop_index("idx_evaluations_status", table_name="evaluations")
    op.drop_index("idx_evaluations_dataset_id", "evaluations")
    op.drop_index("idx_evaluations_tenant_id", table_name="evaluations")
    op.drop_table("evaluations")
