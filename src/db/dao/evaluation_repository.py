"""Evaluation pipeline persistence — raw SQL only, no ORM.

Covers the four tables that back the offline evaluation domain:

- ``evaluations`` — task-level CRUD + tenant-scoped listing.
- ``evaluation_datasets`` — dataset-level CRUD + per-evaluation listing.
- ``evaluation_runs`` — run-level CRUD + per-evaluation listing.
- ``evaluation_metrics`` — metric-level CRUD + per-run listing.

Every read filters ``deleted_at is null`` (via the base
``GenericRepository`` helpers) so a soft-deleted row behaves as if it
no longer exists. Every write is a thin wrapper over the base CRUD or
a small conditional UPDATE / DELETE that returns a row count.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.evaluation import (
    Evaluation,
    EvaluationDataset,
    EvaluationMetric,
    EvaluationRun,
)

_LIVE = "deleted_at is null"

# Newest first: the task / dataset / run lists read as
# reverse-chronological feeds.
_LIST_ORDER = "created_at desc, id desc"


class EvaluationRepository(
    GenericRepository[Evaluation],
):
    """`evaluations`-table SQL — task CRUD + tenant-scoped listing."""

    model_class = Evaluation

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Evaluation) -> Evaluation:
        """Insert a task row and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: Evaluation) -> Evaluation:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``created_at`` are immutable by
        contract, so they stay out of the SET clause.
        """
        immutable = {"id", "tenant_id", "created_at"}
        updates: BindParams = {
            k: v for k, v in row.model_dump().items() if k not in immutable
        }
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="evaluation.update_no_row",
                message=f"evaluation {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark a live task deleted. Returns whether a row was affected."""
        stmt = text(
            "update evaluations set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_or_none(self, id: str) -> Evaluation | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_id(self, id: str) -> Evaluation:
        """Return the live row for ``id``; raise when absent."""
        return await self.find_by_primary_key_or_fail({"id": id})

    async def get_by_id_for_tenant(
        self,
        *,
        id: str,
        tenant_id: int,
    ) -> Evaluation | None:
        """Return the live row for ``id`` scoped to ``tenant_id``."""
        return await self.find_unique_by_column_values(
            {"id": id, "tenant_id": tenant_id},
        )

    async def list_by_tenant(
        self,
        tenant_id: int,
        *,
        status: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Evaluation]:
        """Every live task of one tenant, newest first.

        ``status`` is an optional exact-match filter (pending / running
        / success / failed). ``limit`` / ``offset`` cap and paginate the
        result set.
        """
        params: BindParams = {"tenant_id": tenant_id, "limit": limit, "offset": offset}
        where = "tenant_id = :tenant_id and deleted_at is null"
        if status:
            where += " and status = :status"
            params["status"] = status
        stmt = text(
            f"select * from evaluations where {where} "
            f"order by {_LIST_ORDER} limit :limit offset :offset"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_by_tenant(
        self,
        tenant_id: int,
        *,
        status: str = "",
    ) -> int:
        """Count live tasks of one tenant, optionally filtered by status."""
        params: BindParams = {"tenant_id": tenant_id}
        where = "tenant_id = :tenant_id and deleted_at is null"
        if status:
            where += " and status = :status"
            params["status"] = status
        stmt = text(f"select count(*) from evaluations where {where}").bindparams(**params)
        return int((await self._session.execute(stmt)).scalar_one())


class EvaluationDatasetRepository(
    GenericRepository[EvaluationDataset],
):
    """`evaluation_datasets`-table SQL — dataset CRUD + per-evaluation listing."""

    model_class = EvaluationDataset

    async def create(self, row: EvaluationDataset) -> EvaluationDataset:
        """Insert a dataset row and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: EvaluationDataset) -> EvaluationDataset:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``evaluation_id`` / ``created_at`` are immutable by
        contract.
        """
        immutable = {"id", "evaluation_id", "created_at"}
        updates: BindParams = {
            k: v for k, v in row.model_dump().items() if k not in immutable
        }
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="evaluation_dataset.update_no_row",
                message=f"evaluation dataset {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark a live dataset deleted. Returns whether a row was affected."""
        stmt = text(
            "update evaluation_datasets set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def get_by_id_or_none(self, id: str) -> EvaluationDataset | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def list_by_evaluation(self, evaluation_id: str) -> list[EvaluationDataset]:
        """Every live dataset attached to one evaluation, newest first."""
        stmt = text(
            "select * from evaluation_datasets "
            "where evaluation_id = :evaluation_id and deleted_at is null "
            f"order by {_LIST_ORDER}"
        ).bindparams(evaluation_id=evaluation_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


class EvaluationRunRepository(
    GenericRepository[EvaluationRun],
):
    """`evaluation_runs`-table SQL — run CRUD + per-evaluation listing."""

    model_class = EvaluationRun

    async def create(self, row: EvaluationRun) -> EvaluationRun:
        """Insert a run row and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: EvaluationRun) -> EvaluationRun:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``evaluation_id`` / ``created_at`` are immutable by
        contract.
        """
        immutable = {"id", "evaluation_id", "created_at"}
        updates: BindParams = {
            k: v for k, v in row.model_dump().items() if k not in immutable
        }
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="evaluation_run.update_no_row",
                message=f"evaluation run {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark a live run deleted. Returns whether a row was affected."""
        stmt = text(
            "update evaluation_runs set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def get_by_id_or_none(self, id: str) -> EvaluationRun | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def get_by_id(self, id: str) -> EvaluationRun:
        """Return the live row for ``id``; raise when absent."""
        return await self.find_by_primary_key_or_fail({"id": id})

    async def list_by_evaluation(self, evaluation_id: str) -> list[EvaluationRun]:
        """Every live run of one evaluation, newest first."""
        stmt = text(
            "select * from evaluation_runs "
            "where evaluation_id = :evaluation_id and deleted_at is null "
            f"order by {_LIST_ORDER}"
        ).bindparams(evaluation_id=evaluation_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def delete_by_evaluation(
        self,
        *,
        evaluation_id: str,
        now: datetime,
    ) -> int:
        """Soft-delete every live run of an evaluation. Returns rows affected."""
        stmt = text(
            "update evaluation_runs set deleted_at = :now, updated_at = :now "
            "where evaluation_id = :evaluation_id and deleted_at is null"
        ).bindparams(evaluation_id=evaluation_id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return int(result.rowcount or 0)


class EvaluationMetricRepository(
    GenericRepository[EvaluationMetric],
):
    """`evaluation_metrics`-table SQL — metric CRUD + per-run listing."""

    model_class = EvaluationMetric

    async def create(self, row: EvaluationMetric) -> EvaluationMetric:
        """Insert a metric row and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: EvaluationMetric) -> EvaluationMetric:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``run_id`` / ``created_at`` are immutable by contract.
        """
        immutable = {"id", "run_id", "created_at"}
        updates: BindParams = {
            k: v for k, v in row.model_dump().items() if k not in immutable
        }
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="evaluation_metric.update_no_row",
                message=f"evaluation metric {row.id} not found for update",
            )
        return persisted

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        """Mark a live metric deleted. Returns whether a row was affected."""
        stmt = text(
            "update evaluation_metrics set deleted_at = :now, updated_at = :now "
            "where id = :id and deleted_at is null"
        ).bindparams(id=id, now=now)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def get_by_id_or_none(self, id: str) -> EvaluationMetric | None:
        """Return the live row for ``id``, or ``None`` when absent."""
        return await self.find_by_primary_key({"id": id})

    async def list_by_run(self, run_id: str) -> list[EvaluationMetric]:
        """Every live metric record attached to one run, newest first."""
        stmt = text(
            "select * from evaluation_metrics "
            "where run_id = :run_id and deleted_at is null "
            f"order by {_LIST_ORDER}"
        ).bindparams(run_id=run_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def latest_by_run(self, run_id: str) -> EvaluationMetric | None:
        """Return the most recent live metric record for ``run_id``.

        Used by the result endpoint to surface the latest metric bundle
        without scanning the whole list.
        """
        stmt = text(
            "select * from evaluation_metrics "
            "where run_id = :run_id and deleted_at is null "
            f"order by {_LIST_ORDER} limit 1"
        ).bindparams(run_id=run_id)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())


__all__ = [
    "EvaluationDatasetRepository",
    "EvaluationMetricRepository",
    "EvaluationRepository",
    "EvaluationRunRepository",
]
