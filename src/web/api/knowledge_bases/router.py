"""Knowledge-base HTTP endpoints - CRUD, search, copy, counts and move targets.

Registered by ``RegisterKnowledgeBaseRoutes``.

Knowledge bases are tenant-scoped: every service call takes the caller's
``tenant_id`` from the request context, and a cross-workspace id reads
as 404 rather than 403 so the id space is not enumerable.

============================================  ========
Route                                          Role
============================================  ========
``POST   /knowledge-bases``                    Admin
``GET    /knowledge-bases``                    Viewer
``GET    /knowledge-bases/{id}``               Viewer
``PUT    /knowledge-bases/{id}``               Admin
``DELETE /knowledge-bases/{id}``               Admin
``POST   /knowledge-bases/{id}/duplicate``     Admin
``POST   /knowledge-bases/copy``               Admin
``GET    /knowledge-bases/{id}/move-targets``  Viewer
``POST   /knowledge-bases/{id}/hybrid-search`` Viewer
``GET    /knowledge-bases/{id}/hybrid-search`` Viewer
============================================  ========

Route order matters: the static ``/copy`` path is declared before the
``/{id}``-shaped routes so a literal segment is never captured as an id.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream swagger annotations).
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src.common.exception import UnauthorizedError
from src.common.json import JsonObject
from src.core.contracts.knowledge import (
    CreateKnowledgeBaseRequest,
    HybridSearchRequest,
    KnowledgeCopyRequest,
    KnowledgeCopyResponse,
    KnowledgeDuplicateResponse,
    UpdateKnowledgeBaseRequest,
)
from src.core.knowledge.knowledge_bases.copy import copy_kb
from src.core.knowledge.knowledge_bases.duplicate import duplicate_kb
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)
from src.web.api.knowledge_bases.views import (
    DeleteKnowledgeBaseResponse,
    HybridSearchEnvelope,
    KnowledgeBaseEnvelope,
    KnowledgeBaseListEnvelope,
    KnowledgeCopyEnvelope,
    KnowledgeDuplicateEnvelope,
    knowledge_base_envelope,
    knowledge_base_list_envelope,
    knowledge_base_to_contract,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep
from src.web.deps.knowledge_bases import KBServiceDep
from src.web.deps.session import SessionDep

# Function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

# The handler answers create/duplicate/delete/copy with message strings
# that the UI matches verbatim.
_DELETE_MESSAGE = "Knowledge base deleted successfully"
_COPY_MESSAGE = "Knowledge base copy task started"
_DUPLICATE_MESSAGE = "Knowledge base duplicate created"


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A knowledge base is always workspace-scoped; without a tenant
    context there is no safe default (tenant 0 is the system scope,
    which owns no knowledge bases), so this rejects rather than
    guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="knowledge_base.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _config_to_json(value: BaseModel | None) -> JsonObject | None:
    """Dump an optional typed config contract onto the service JSON shape."""
    if value is None:
        return None
    return value.model_dump(mode="json")


def _filter_by_creator(
    infos: list[KnowledgeBaseInfo],
    *,
    creator: str,
    user_id: str | None,
) -> list[KnowledgeBaseInfo]:
    """Apply the optional [All | Mine | Others] creator filter.

    Rows with no creator id are treated as belonging to no one, so they
    fall out of both ``mine`` and ``others``.
    """
    key = creator.strip().lower()
    if key not in ("mine", "others"):
        return infos
    caller = user_id or ""
    kept: list[KnowledgeBaseInfo] = []
    for info in infos:
        if not info.creator_id:
            continue
        if (key == "mine" and info.creator_id == caller) or (
            key == "others" and info.creator_id != caller
        ):
            kept.append(info)
    return kept


async def _fill_counts(
    *,
    service: KBService,
    tenant_id: int,
    info: KnowledgeBaseInfo,
) -> KnowledgeBaseInfo:
    """Fill the aggregate counts carried by the detail response.

    Mirrors the list path: document rows get ``knowledge_count`` and
    FAQ rows get ``chunk_count``. Best-effort — a failing count query
    leaves the field at its zero default.
    """
    updates: dict[str, int] = {}
    try:
        if info.type == KNOWLEDGE_BASE_TYPE_DOCUMENT:
            updates["knowledge_count"] = await service.count_documents(
                tenant_id=tenant_id,
                knowledge_base_id=info.id,
            )
        elif info.type == KNOWLEDGE_BASE_TYPE_FAQ:
            updates["chunk_count"] = await service.count_chunks(
                tenant_id=tenant_id,
                knowledge_base_id=info.id,
            )
    except Exception:
        return info
    return info.model_copy(update=updates)


def _accept_language(request: Request) -> str:
    """Return the caller's language hint for the duplicate-name suffix."""
    return request.headers.get("accept-language", "en")


def _generate_task_id(tenant_id: int, source_id: str) -> str:
    """Build the caller-facing task id for a synchronous copy.

    Mirrors the upstream format so progress polling can key on the same
    shape once the async worker lands.
    """
    return f"kb_clone_{tenant_id}_{source_id}_{int(time.time())}"


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("", response_model=KnowledgeBaseEnvelope, status_code=201)
async def create_knowledge_base(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: CreateKnowledgeBaseRequest,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> KnowledgeBaseEnvelope:
    """Create a knowledge base for the active workspace.

    The domain service stamps id / tenant / timestamps and applies the
    type-specific config defaults before persisting.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.create_knowledge_base(
        tenant_id=tenant_id,
        name=body.name,
        kb_type=body.type or KNOWLEDGE_BASE_TYPE_DOCUMENT,
        description=body.description,
        creator_id=user_id,
        is_temporary=body.is_temporary,
        chunking_config=_config_to_json(body.chunking_config),
        image_processing_config=_config_to_json(body.image_processing_config),
        embedding_model_id=body.embedding_model_id,
        summary_model_id=body.summary_model_id,
        vlm_config=_config_to_json(body.vlm_config),
        asr_config=_config_to_json(body.asr_config),
        storage_provider_config=_config_to_json(body.storage_provider_config),
        storage_config=_config_to_json(body.storage_config),
        extract_config=_config_to_json(body.extract_config),
        faq_config=_config_to_json(body.faq_config),
        question_generation_config=_config_to_json(body.question_generation_config),
        wiki_config=_config_to_json(body.wiki_config),
        indexing_strategy=_config_to_json(body.indexing_strategy),
        vector_store_id=body.vector_store_id,
    )
    return knowledge_base_envelope(info)


@router.get("", response_model=KnowledgeBaseListEnvelope)
async def list_knowledge_bases(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
    creator: str = Query(default="", description="创建者筛选"),
) -> KnowledgeBaseListEnvelope:
    """List the workspace's knowledge bases, newest first.

    The service enriches each row with its aggregate counts; the
    optional ``creator`` query drives the [All | Mine | Others]
    segmented control.
    """
    tenant_id = _require_tenant(tenant_id)
    infos = await service.list_knowledge_bases(tenant_id=tenant_id)
    infos = _filter_by_creator(infos, creator=creator, user_id=user_id)
    return knowledge_base_list_envelope(infos)


@router.get("/{id}", response_model=KnowledgeBaseEnvelope)
async def get_knowledge_base(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeBaseEnvelope:
    """Return one knowledge base, enriched with its aggregate counts.

    Ownership is enforced by the tenant-scoped read, so a cross-workspace
    id reads as not-found.
    """
    tenant_id = _require_tenant(tenant_id)
    info = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    info = await _fill_counts(service=service, tenant_id=tenant_id, info=info)
    return knowledge_base_envelope(info)


@router.put("/{id}", response_model=KnowledgeBaseEnvelope)
async def update_knowledge_base(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpdateKnowledgeBaseRequest,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeBaseEnvelope:
    """Partial-update a knowledge base's mutable fields.

    Every body field is optional (per ``UpdateKnowledgeBaseRequest``);
    the service treats a missing field as "leave the existing value
    alone", so the same request shape works for PUT (full body) and
    PATCH (subset). The tenant-ownership pre-check makes a
    cross-workspace id read as not-found; the vector-store binding is
    immutable by contract and never part of an update.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    info = await service.update_knowledge_base(
        knowledge_base_id=id,
        name=body.name,
        description=body.description,
        config=body.config,
    )
    return knowledge_base_envelope(info)


@router.delete("/{id}", response_model=DeleteKnowledgeBaseResponse)
async def delete_knowledge_base(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> DeleteKnowledgeBaseResponse:
    """Soft-delete a knowledge base.

    The heavy content cleanup (indexes, files, graph) is a deferred
    seam in the domain service; the row itself is removed synchronously.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    await service.delete_knowledge_base(knowledge_base_id=id)
    return DeleteKnowledgeBaseResponse(success=True, message=_DELETE_MESSAGE)


# ── Copy / duplicate ─────────────────────────────────────────────────


@router.post("/copy", response_model=KnowledgeCopyEnvelope)
async def copy_knowledge_base(
    _auth: AuthDep,
    _role: RoleAdminDep,
    body: KnowledgeCopyRequest,
    service: KBServiceDep,
    session: SessionDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> KnowledgeCopyEnvelope:
    """Copy a knowledge base's settings.

    With ``target_id`` the clone is settings-level into the existing
    target; without it a new knowledge base is created mirroring the
    source. The copy defenses (embedding-model, vector-store and
    storage-instance match) are enforced by the domain composition.
    """
    tenant_id = _require_tenant(tenant_id)
    source, target = await copy_kb(
        service=service,
        session=session,
        tenant_id=tenant_id,
        source_kb_id=body.source_id,
        target_kb_id=body.target_id,
        creator_id=user_id,
    )
    return KnowledgeCopyEnvelope(
        success=True,
        data=KnowledgeCopyResponse(
            task_id=body.task_id or _generate_task_id(tenant_id, source.id),
            source_id=source.id,
            target_id=target.id,
            message=_COPY_MESSAGE,
        ),
    )


@router.post("/{id}/duplicate", response_model=KnowledgeDuplicateEnvelope, status_code=201)
async def duplicate_knowledge_base(
    _auth: AuthDep,
    _role: RoleAdminDep,
    request: Request,
    id: str,
    service: KBServiceDep,
    session: SessionDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> KnowledgeDuplicateEnvelope:
    """Create a settings-only copy of a knowledge base.

    The new row carries the source's configuration but none of its
    content; the name gets a locale-aware suffix derived from the
    ``Accept-Language`` header.
    """
    tenant_id = _require_tenant(tenant_id)
    target = await duplicate_kb(
        service=service,
        session=session,
        tenant_id=tenant_id,
        source_kb_id=id,
        creator_id=user_id,
        locale=_accept_language(request),
    )
    return KnowledgeDuplicateEnvelope(
        success=True,
        data=KnowledgeDuplicateResponse(
            source_id=id,
            target_id=target.id,
            message=_DUPLICATE_MESSAGE,
            knowledge_base=knowledge_base_to_contract(target),
        ),
    )


# ── Move targets ─────────────────────────────────────────────────────


@router.get("/{id}/move-targets", response_model=KnowledgeBaseListEnvelope)
async def list_move_targets(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> KnowledgeBaseListEnvelope:
    """List knowledge bases eligible as move targets for the source.

    Filters to same-type, same-embedding-model, non-temporary KBs in the
    workspace, excluding the source itself.
    """
    tenant_id = _require_tenant(tenant_id)
    source = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    candidates = await service.list_knowledge_bases(tenant_id=tenant_id)
    targets = [
        info
        for info in candidates
        if info.id != source.id
        and not info.is_temporary
        and info.type == source.type
        and info.embedding_model_id == source.embedding_model_id
    ]
    return knowledge_base_list_envelope(targets)


# ── Hybrid search ────────────────────────────────────────────────────


@router.post("/{id}/hybrid-search", response_model=HybridSearchEnvelope)
async def hybrid_search(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    body: HybridSearchRequest,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> HybridSearchEnvelope:
    """Hybrid (vector + keyword) search inside one knowledge base.

    The retrieval pipeline is a deferred seam — no search service is
    wired into the web layer yet. The endpoint validates knowledge-base
    access and returns the wire shape with an empty result set so
    callers can build against the contract.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    return HybridSearchEnvelope(success=True, data=[])


@router.get("/{id}/hybrid-search", response_model=HybridSearchEnvelope)
async def hybrid_search_get(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: KBServiceDep,
    tenant_id: _PrincipalTenant,
) -> HybridSearchEnvelope:
    """GET compatibility shim for legacy clients carrying a JSON body.

    The request body is ignored because FastAPI does not bind JSON bodies
    on GET and the retrieval pipeline is not wired; access is validated
    identically to the POST form.
    """
    tenant_id = _require_tenant(tenant_id)
    await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tenant_id,
        knowledge_base_id=id,
    )
    return HybridSearchEnvelope(success=True, data=[])


__all__ = ["router"]
