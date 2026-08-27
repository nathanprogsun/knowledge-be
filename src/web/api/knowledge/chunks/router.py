"""Chunk HTTP endpoints - knowledge chunk CRUD, revisions, and questions.

Registered by ``RegisterChunkRoutes``.

Chunks are the retrieval units split out of a knowledge item. The router
exposes the chunk surface over the request-scoped chunk services:

========================================  ========
Route                                     Role
========================================  ========
``GET    /chunks/by-id/{id}``             Viewer
``PUT    /chunks/by-id/{id}/questions``   Admin
``DELETE /chunks/by-id/{id}/questions``   Admin
``POST   /chunks/by-id/{id}/questions/regenerate`` Admin
``GET    /chunks/{knowledge_id}``         Viewer
``GET    /chunks/{knowledge_id}/{id}/revisions`` Viewer
``DELETE /chunks/{knowledge_id}/{id}``    Admin
``DELETE /chunks/{knowledge_id}``         Admin
``PUT    /chunks/{knowledge_id}/{id}``    Admin
``POST   /chunks/{knowledge_id}/{id}/revert`` Admin
========================================  ========

Static ``/by-id`` routes are declared before ``/{knowledge_id}``-shaped
routes so the literal segment is never captured as a knowledge id. Every
mutation on a ``:knowledge_id`` route first verifies the chunk belongs to
that knowledge (defence in depth on top of the route-level access gate)
and answers a mismatch with 403 rather than leaking the chunk.

Reads are Viewer+; every mutation is Admin+ (the closest role-level
approximation of the upstream KB-owner-or-Admin gate). The LLM-backed
regenerate endpoint is wired but returns an empty result until the
question-generation orchestration lands in the domain.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.common.exception import PermissionDeniedError, UnauthorizedError, ValidationError
from src.core.knowledge.chunks.questions import (
    DocumentChunkMetadata,
    bind_generated_question,
    parse_document_metadata,
    unbind_generated_question,
)
from src.core.knowledge.chunks.service.chunk_service import chunk_to_contract
from src.core.knowledge.chunks.types import CHUNK_TYPE_TEXT
from src.web.api.knowledge.chunks.views import (
    ChunkEnvelope,
    ChunkListEnvelope,
    ChunkMessageEnvelope,
    ChunkRevisionsEnvelope,
    DeleteGeneratedQuestionRequest,
    GeneratedQuestionEnvelope,
    GeneratedQuestionsEnvelope,
    RevertChunkRequest,
    UpdateChunkRequest,
    UpsertGeneratedQuestionRequest,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.chunks import ChunkRevisionServiceDep, ChunkServiceDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep

# Shortcut aliases for the function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]

router = APIRouter(prefix="/chunks", tags=["chunks"])


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A chunk is always workspace-scoped; without a tenant context there is
    no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _actor(user_id: str | None) -> str:
    """Return the acting user's id for ``last_editor_id`` (``""`` absent)."""
    return user_id or ""


# ── by-id routes (declared before /{knowledge_id}) ───────────────────


@router.get("/by-id/{id}", response_model=ChunkEnvelope)
async def get_chunk_by_id_only(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: ChunkServiceDep,
) -> ChunkEnvelope:
    """Get one chunk by its id without a knowledge id.

    No workspace scoping here: the caller's read permission is resolved
    against the chunk's parent knowledge by the access gate before the
    handler runs, so a shared knowledge's chunks stay reachable.
    """
    chunk = await service.get_chunk_by_id_only(id=id)
    return ChunkEnvelope(success=True, data=chunk_to_contract(chunk))


@router.put("/by-id/{id}/questions", response_model=GeneratedQuestionEnvelope)
async def upsert_generated_question(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: UpsertGeneratedQuestionRequest,
    service: ChunkServiceDep,
) -> GeneratedQuestionEnvelope:
    """Insert or replace one generated retrieval question on a chunk.

    An empty ``question_id`` appends a fresh question bound to the
    chunk's current ``content_revision``; a known id replaces the stored
    question text. Persists the updated document metadata back to the
    chunk row.
    """
    chunk = await service.get_chunk_by_id_only(id=id)
    metadata = parse_document_metadata(chunk.metadata) or DocumentChunkMetadata()
    updated, bound = bind_generated_question(
        metadata,
        question_id=body.question_id or None,
        question=body.question,
        content_revision=chunk.content_revision,
    )
    await service.update_chunk(
        chunk=chunk.model_copy(
            update={
                "metadata": updated.to_json(),
                "updated_at": datetime.now(UTC),
            }
        )
    )
    return GeneratedQuestionEnvelope(success=True, data=bound)


@router.delete("/by-id/{id}/questions", response_model=ChunkMessageEnvelope)
async def delete_generated_question(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    body: DeleteGeneratedQuestionRequest,
    service: ChunkServiceDep,
) -> ChunkMessageEnvelope:
    """Delete one generated question from a chunk's metadata."""
    chunk = await service.get_chunk_by_id_only(id=id)
    metadata = parse_document_metadata(chunk.metadata) or DocumentChunkMetadata()
    updated, _removed = unbind_generated_question(metadata, question_id=body.question_id)
    await service.update_chunk(
        chunk=chunk.model_copy(
            update={
                "metadata": updated.to_json(),
                "updated_at": datetime.now(UTC),
            }
        )
    )
    return ChunkMessageEnvelope(success=True, message="Generated question deleted")


@router.post("/by-id/{id}/questions/regenerate", response_model=GeneratedQuestionsEnvelope)
async def regenerate_generated_questions(
    _auth: AuthDep,
    _role: RoleAdminDep,
    id: str,
    service: ChunkServiceDep,
) -> GeneratedQuestionsEnvelope:
    """Regenerate a chunk's generated retrieval questions.

    The chunk is resolved so a missing id surfaces as 404 and a non-text
    chunk is rejected up front. The LLM-backed generation orchestration is
    a deferred domain concern, so the endpoint is wired and returns an
    empty result until that pipeline lands.
    """
    chunk = await service.get_chunk_by_id_only(id=id)
    if chunk.chunk_type != CHUNK_TYPE_TEXT:
        raise ValidationError(
            code="chunk.question_non_text",
            message="questions can only be generated for text chunks",
        )
    return GeneratedQuestionsEnvelope(success=True, data=[])


# ── knowledge-scoped routes ──────────────────────────────────────────


@router.get("/{knowledge_id}", response_model=ChunkListEnvelope)
async def list_knowledge_chunks(
    _auth: AuthDep,
    _role: RoleViewerDep,
    knowledge_id: str,
    service: ChunkServiceDep,
    tenant_id: _PrincipalTenant,
    page: int = Query(default=1, description="页码"),
    page_size: int = Query(default=10, description="每页数量"),
    chunk_type: Annotated[list[str] | None, Query(description="分块类型，可重复")] = None,
) -> ChunkListEnvelope:
    """List a knowledge item's chunks, paginated, in document order.

    Defaults to text chunks, matching the upstream default; a caller may
    request other types via repeated ``?chunk_type=``. The merged listing
    service currently returns text chunks only, so a non-text type
    request answers with an empty page until the paged/filtered listing
    lands. Page bounds are clamped like the upstream handler (min page 1,
    page size 1-100) rather than rejected.
    """
    tenant_id = _require_tenant(tenant_id)
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    chunks = await service.list_chunks_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
    )
    wanted = set(chunk_type) if chunk_type else {CHUNK_TYPE_TEXT}
    filtered = [c for c in chunks if c.chunk_type in wanted]
    offset = (page - 1) * page_size
    data = [chunk_to_contract(c) for c in filtered[offset : offset + page_size]]
    return ChunkListEnvelope(
        success=True,
        data=data,
        total=len(filtered),
        page=page,
        page_size=page_size,
    )


@router.get("/{knowledge_id}/{id}/revisions", response_model=ChunkRevisionsEnvelope)
async def list_chunk_revisions(
    _auth: AuthDep,
    _role: RoleViewerDep,
    knowledge_id: str,
    id: str,
    service: ChunkServiceDep,
    revision_service: ChunkRevisionServiceDep,
    tenant_id: _PrincipalTenant,
) -> ChunkRevisionsEnvelope:
    """List a chunk's edit history, newest revision first."""
    tenant_id = _require_tenant(tenant_id)
    chunk = await service.get_chunk_by_id(tenant_id=tenant_id, id=id)
    # Defence in depth: the route-level access gate already authorised the
    # caller for the knowledge; this stops a same-workspace caller from
    # passing one knowledge_id while addressing a chunk of another.
    if chunk.knowledge_id != knowledge_id:
        raise PermissionDeniedError(
            code="chunk.not_owned",
            message="No permission to access this chunk",
        )
    items = await revision_service.list_revisions(
        tenant_id=tenant_id,
        chunk_id=chunk.id,
    )
    return ChunkRevisionsEnvelope(success=True, data=items)


@router.put("/{knowledge_id}/{id}", response_model=ChunkEnvelope)
async def update_chunk(
    _auth: AuthDep,
    _role: RoleAdminDep,
    knowledge_id: str,
    id: str,
    body: UpdateChunkRequest,
    service: ChunkServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> ChunkEnvelope:
    """Apply an optimistic, revision-guarded edit to a text chunk.

    ``content`` and ``is_enabled`` are optional (absent keeps the current
    value); ``expected_revision`` guards against a stale edit. A
    concurrent edit that advanced the revision answers with 409.
    """
    tenant_id = _require_tenant(tenant_id)
    chunk = await service.get_chunk_by_id(tenant_id=tenant_id, id=id)
    if chunk.knowledge_id != knowledge_id:
        raise PermissionDeniedError(
            code="chunk.not_owned",
            message="No permission to access this chunk",
        )
    updated = await service.update_document_chunk(
        tenant_id=tenant_id,
        chunk_id=id,
        content=body.content,
        is_enabled=body.is_enabled,
        expected_revision=body.expected_revision,
        last_editor_id=_actor(user_id),
    )
    return ChunkEnvelope(success=True, data=chunk_to_contract(updated))


@router.delete("/{knowledge_id}/{id}", response_model=ChunkMessageEnvelope)
async def delete_chunk(
    _auth: AuthDep,
    _role: RoleAdminDep,
    knowledge_id: str,
    id: str,
    service: ChunkServiceDep,
    tenant_id: _PrincipalTenant,
) -> ChunkMessageEnvelope:
    """Soft-delete one chunk."""
    tenant_id = _require_tenant(tenant_id)
    chunk = await service.get_chunk_by_id(tenant_id=tenant_id, id=id)
    if chunk.knowledge_id != knowledge_id:
        raise PermissionDeniedError(
            code="chunk.not_owned",
            message="No permission to access this chunk",
        )
    await service.delete_chunk(tenant_id=tenant_id, id=id)
    return ChunkMessageEnvelope(success=True, message="Chunk deleted")


@router.delete("/{knowledge_id}", response_model=ChunkMessageEnvelope)
async def delete_chunks_by_knowledge_id(
    _auth: AuthDep,
    _role: RoleAdminDep,
    knowledge_id: str,
    service: ChunkServiceDep,
    tenant_id: _PrincipalTenant,
) -> ChunkMessageEnvelope:
    """Soft-delete every chunk under a knowledge item."""
    tenant_id = _require_tenant(tenant_id)
    await service.delete_chunks_by_knowledge_id(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
    )
    return ChunkMessageEnvelope(success=True, message="All chunks under knowledge deleted")


@router.post("/{knowledge_id}/{id}/revert", response_model=ChunkEnvelope)
async def revert_chunk(
    _auth: AuthDep,
    _role: RoleAdminDep,
    knowledge_id: str,
    id: str,
    body: RevertChunkRequest,
    service: ChunkServiceDep,
    revision_service: ChunkRevisionServiceDep,
    tenant_id: _PrincipalTenant,
    user_id: _PrincipalUser,
) -> ChunkEnvelope:
    """Revert a chunk to a historical revision.

    Replays the snapshot's ``content`` and ``is_enabled`` through the
    optimistic edit pipeline, so a concurrent edit still answers with 409.
    """
    tenant_id = _require_tenant(tenant_id)
    if body.revision < 0:
        raise ValidationError(
            code="chunk.invalid_revision",
            message="revision must be a non-negative integer",
        )
    chunk = await service.get_chunk_by_id(tenant_id=tenant_id, id=id)
    if chunk.knowledge_id != knowledge_id:
        raise PermissionDeniedError(
            code="chunk.not_owned",
            message="No permission to access this chunk",
        )
    snapshot = await revision_service.get_revision(
        tenant_id=tenant_id,
        chunk_id=id,
        revision=body.revision,
    )
    updated = await service.update_document_chunk(
        tenant_id=tenant_id,
        chunk_id=id,
        content=snapshot.content,
        is_enabled=snapshot.is_enabled,
        expected_revision=body.expected_revision,
        last_editor_id=_actor(user_id),
    )
    return ChunkEnvelope(success=True, data=chunk_to_contract(updated))


__all__ = ["router"]
