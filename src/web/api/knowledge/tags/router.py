"""Knowledge-tag HTTP endpoints — knowledge-base scoped tag CRUD.

Tags are knowledge-base metadata, so reads are Viewer+ and every
mutation is Contributor+. Per-KB ownership enforcement (the creator-or-
admin matrix) lands with the KB access guard; the tag service validates
tenant-level KB ownership on every write, so a cross-workspace id reads
as 404 rather than 403.

=====================================  ==========
Route                                  Role
=====================================  ==========
``GET    /knowledge-bases/{id}/tags``  Viewer
``POST   /knowledge-bases/{id}/tags``  Contributor
``PUT    /knowledge-bases/{id}/tags/{tag_id}``   Contributor
``DELETE /knowledge-bases/{id}/tags/{tag_id}``   Contributor
=====================================  ==========

``tag_id`` may be either the tag's UUID or its numeric ``seq_id``; the
service resolves a numeric value before the update/delete.

Query-parameter ``description`` strings are intentionally Chinese
(mirrors the upstream Go swagger annotations). RUF001 flags the
full-width punctuation; suppressed file-wide for the same reason as
``src/web/api/system/router.py``.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.common.exception import UnauthorizedError
from src.core.contracts.knowledge import CreateTagRequest, UpdateTagRequest
from src.web.api.knowledge.tags.views import (
    DeleteTagRequest,
    DeleteTagResponse,
    TagEnvelope,
    TagListEnvelope,
    tag_envelope,
    tag_list_envelope,
)
from src.web.deps import AuthDep, RoleContributorDep, RoleViewerDep, TagServiceDep
from src.web.deps.context import get_tenant_id_dep

# Shortcut aliases for the function-arg-style principal deps.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]

router = APIRouter(prefix="/knowledge-bases/{id}/tags", tags=["tags"])


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail.

    A tag is always tenant-scoped; without a tenant context there is no
    safe default (tenant 0 is the system scope, which owns no tags), so
    this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="auth.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


# ── List ─────────────────────────────────────────────────────────────


@router.get("", response_model=TagListEnvelope)
async def list_tags(
    _auth: AuthDep,
    _role: RoleViewerDep,
    id: str,
    service: TagServiceDep,
    tenant_id: _PrincipalTenant,
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=1000, description="每页数量"),
    keyword: str = Query(default="", description="关键词搜索"),
) -> TagListEnvelope:
    """Return one page of the knowledge base's tags plus usage stats."""
    tenant_id = _require_tenant(tenant_id)
    result = await service.list_tags(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        page=page,
        page_size=page_size,
        keyword=keyword,
    )
    return tag_list_envelope(result)


# ── Create ───────────────────────────────────────────────────────────


@router.post("", response_model=TagEnvelope)
async def create_tag(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    body: CreateTagRequest,
    service: TagServiceDep,
    tenant_id: _PrincipalTenant,
) -> TagEnvelope:
    """Create a tag under the knowledge base."""
    tenant_id = _require_tenant(tenant_id)
    info = await service.create_tag(
        tenant_id=tenant_id,
        knowledge_base_id=id,
        name=body.name,
        color=body.color,
        sort_order=body.sort_order if body.sort_order is not None else 0,
    )
    return tag_envelope(info)


# ── Update ───────────────────────────────────────────────────────────


@router.put("/{tag_id}", response_model=TagEnvelope)
async def update_tag(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    tag_id: str,
    body: UpdateTagRequest,
    service: TagServiceDep,
    tenant_id: _PrincipalTenant,
) -> TagEnvelope:
    """Patch a tag's mutable fields; ``tag_id`` may be a UUID or seq id."""
    tenant_id = _require_tenant(tenant_id)
    resolved = await service.resolve_tag_id(tenant_id=tenant_id, tag_id=tag_id)
    info = await service.update_tag(
        tenant_id=tenant_id,
        tag_id=resolved,
        name=body.name,
        color=body.color,
        sort_order=body.sort_order,
    )
    return tag_envelope(info)


# ── Delete ───────────────────────────────────────────────────────────


@router.delete("/{tag_id}", response_model=DeleteTagResponse)
async def delete_tag(
    _auth: AuthDep,
    _role: RoleContributorDep,
    id: str,
    tag_id: str,
    service: TagServiceDep,
    tenant_id: _PrincipalTenant,
    force: bool = Query(default=False, description="强制删除"),
    content_only: bool = Query(default=False, description="仅删除内容，保留标签"),
    body: DeleteTagRequest | None = None,
) -> DeleteTagResponse:
    """Delete a tag; ``tag_id`` may be a UUID or seq id.

    ``force`` bypasses the reference guard, ``content_only`` clears only
    the tag's content, and a non-empty ``exclude_ids`` keeps the tag
    because excluded content still references it.
    """
    tenant_id = _require_tenant(tenant_id)
    resolved = await service.resolve_tag_id(tenant_id=tenant_id, tag_id=tag_id)
    exclude_ids = body.exclude_ids if body is not None else []
    await service.delete_tag(
        tenant_id=tenant_id,
        tag_id=resolved,
        force=force,
        content_only=content_only,
        exclude_ids=[str(seq) for seq in exclude_ids] or None,
    )
    return DeleteTagResponse(success=True)


__all__ = ["router"]
