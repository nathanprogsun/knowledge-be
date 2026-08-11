"""Evaluation HTTP endpoints - run and retrieve evaluation tasks.

Maps the upstream evaluation handler:

===========================================  ====
Route                                        Action
===========================================  ====
``POST   /evaluation``                        Run an evaluation over a dataset
``GET    /evaluation?task_id=...``            Retrieve a task snapshot (status + metrics)
===========================================  ====

RBAC mirrors the upstream route guard: running an evaluation drives
LLM calls and reads tenant knowledge bases, so it is Admin-gated;
retrieving the read-only snapshot is Viewer+. Every endpoint reads the
caller's workspace id from the request context; a missing context fails
closed with 401.

The request body mirrors the upstream request struct exactly:
``dataset_id`` / ``knowledge_base_id`` / ``chat_id`` / ``rerank_id``.
An empty ``chat_id`` / ``rerank_id`` falls back to the tenant's default
model resolution in the service layer.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.common.exception import UnauthorizedError
from src.core.contracts.evaluation import (
    EvaluationCreateRequest,
    EvaluationGetQuery,
)
from src.core.evaluation.service.seams import EvaluationCreateQuery
from src.web.api.evaluation.views import (
    EvaluationEnvelope,
    evaluation_envelope,
)
from src.web.deps import AuthDep, EvaluationServiceDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep

# Function-arg-style principal dep. ``_PrincipalTenant`` is an int
# (auth-middleware-populated ``request.state.tenant_id``); a request
# without a workspace context reads as ``0`` and the router rejects it
# explicitly because evaluation tasks are workspace-scoped.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail closed.

    Evaluation tasks are workspace-scoped; without a workspace context
    there is no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="evaluation.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


@router.post("", response_model=EvaluationEnvelope)
async def run_evaluation(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: EvaluationCreateRequest,
    service: EvaluationServiceDep,
    tenant_id: _PrincipalTenant,
) -> EvaluationEnvelope:
    """Start an evaluation over a dataset; returns the task snapshot.

    The task runs in the background; the caller polls ``GET /evaluation``
    with the returned ``task.id`` to observe progress and metrics.
    """
    _require_tenant(tenant_id)
    query = EvaluationCreateQuery(
        dataset_id=body.dataset_id,
        knowledge_base_id=body.knowledge_base_id,
        chat_model_id=body.chat_id,
        rerank_model_id=body.rerank_id,
    )
    data = await service.create(query)
    return evaluation_envelope(data)


@router.get("", response_model=EvaluationEnvelope)
async def get_evaluation_result(
    _auth: AuthDep,
    _role: RoleViewerDep,
    query: Annotated[EvaluationGetQuery, Query()],
    service: EvaluationServiceDep,
    tenant_id: _PrincipalTenant,
) -> EvaluationEnvelope:
    """Return the latest snapshot for a task id.

    ``task_id`` is required and must belong to the caller's workspace;
    an unknown id reads as ``404``.
    """
    _require_tenant(tenant_id)
    data = await service.get(query.task_id)
    return evaluation_envelope(data)


__all__ = ["router"]
