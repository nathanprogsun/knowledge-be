"""Async task-progress tenant guard.

Maps the upstream ``requireTaskProgressTenant`` helper: generated task
ids embed the owning workspace id
(``<type>_<tenant>_<millis>_<uuid>[_<business>]``), so a progress
endpoint can verify the caller owns the task before serving it.
Cross-workspace probes are hidden as not-found so the task-id space is
not enumerable; malformed ids fail as invalid input.

FAQ and knowledge-base progress endpoints reuse this guard.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.common.exception import NotFoundError, UnauthorizedError, ValidationError
from src.web.deps.context import get_tenant_id_dep

_INVALID_TASK_ID_CODE = "task_progress.invalid_task_id"
_NOT_FOUND_CODE = "task_progress.not_found"

# A millisecond timestamp is at least 13 digits (year 2001+).
_MIN_TIMESTAMP = 1_000_000_000_000


def task_tenant_id(task_id: str) -> int | None:
    """Extract the embedded workspace id from a generated task id.

    Mirrors the upstream parse: the workspace is the first numeric
    segment that is followed by a millisecond timestamp, so task types
    containing underscores (``faq_import``, ``kg_move``) still resolve.
    Returns ``None`` when the id carries no valid tenant.
    """
    parts = task_id.split("_")
    for index in range(1, len(parts) - 2):
        tenant_raw = parts[index]
        stamp_raw = parts[index + 1]
        if not tenant_raw.isdigit() or int(tenant_raw) <= 0:
            continue
        if not stamp_raw.isdigit() or int(stamp_raw) < _MIN_TIMESTAMP:
            continue
        return int(tenant_raw)
    return None


def require_task_progress_tenant(task_id: str, tenant_id: int) -> None:
    """Ensure the task's owning workspace matches the caller's.

    Raises ``ValidationError`` for a malformed task id, ``UnauthorizedError``
    when the caller has no workspace context, and ``NotFoundError`` for a
    cross-workspace probe (hidden as not-found).
    """
    task_tenant = task_tenant_id(task_id)
    if task_tenant is None:
        raise ValidationError(
            code=_INVALID_TASK_ID_CODE,
            message="invalid task ID",
        )
    if tenant_id <= 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized",
        )
    if task_tenant != tenant_id:
        raise NotFoundError(
            code=_NOT_FOUND_CODE,
            message="task not found",
        )


def get_task_progress_tenant_dep(
    task_id: str,
    tenant_id: Annotated[int, Depends(get_tenant_id_dep)],
) -> None:
    """FastAPI dependency gate for ``GET /.../progress/{task_id}`` routes.

    Declaring this dependency on a progress endpoint ensures the task id
    belongs to the caller's workspace before the handler reads any
    progress record.
    """
    require_task_progress_tenant(task_id, tenant_id)


TaskProgressTenantDep = Annotated[None, Depends(get_task_progress_tenant_dep)]


__all__ = [
    "TaskProgressTenantDep",
    "get_task_progress_tenant_dep",
    "require_task_progress_tenant",
    "task_tenant_id",
]
