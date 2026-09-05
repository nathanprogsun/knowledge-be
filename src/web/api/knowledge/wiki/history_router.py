"""Wiki issue and revision HTTP endpoints.

Registered beside the page/folder router. Static paths (``/issues``,
``/revert``) are declared before ``/revisions/{slug:path}``. Every
handler calls ``require_wiki_kb`` so a wiki-off KB stays
``wiki.kb_wiki_not_enabled``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.core.knowledge.wiki.issues import list_issues, update_issue_status
from src.web.api.knowledge.wiki.views import (
    WikiEnvelope,
    WikiIssueStatusRequest,
    WikiIssueView,
    WikiPageView,
    WikiRevertRequest,
    WikiRevisionListData,
    WikiRevisionView,
    issue_to_view,
    page_to_view,
    require_wiki_kb,
    revision_list_to_view,
    revision_to_view,
)
from src.web.deps import AuthDep, RoleAdminDep, RoleViewerDep
from src.web.deps.context import get_user_id_dep
from src.web.deps.knowledge_wiki import (
    KBServiceDep,
    WikiIssueRepoDep,
    WikiPageServiceDep,
)
from src.web.deps.sharing import KBShareServiceDep

_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]

router = APIRouter(prefix="/knowledgebase/{kb_id}/wiki", tags=["wiki"])


def _actor(user_id: str | None) -> str:
    """Return the acting user's id (``""`` when absent)."""
    return user_id or ""


@router.get("/issues", response_model=WikiEnvelope[list[WikiIssueView]])
async def get_issues(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    issue_repo: WikiIssueRepoDep,
    slug: str = Query(default=""),
    status: str = Query(default=""),
) -> WikiEnvelope[list[WikiIssueView]]:
    """List issues for the KB. ``data`` is the list itself (empty is 200)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    issues = await list_issues(
        issue_repo=issue_repo,
        knowledge_base_id=kb_id,
        slug=slug.strip(),
        status=status.strip(),
    )
    return WikiEnvelope(success=True, data=[issue_to_view(issue) for issue in issues])


@router.put("/issues/{issue_id}/status", response_model=WikiEnvelope[WikiIssueView])
async def put_issue_status(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    issue_id: str,
    body: WikiIssueStatusRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    issue_repo: WikiIssueRepoDep,
) -> WikiEnvelope[WikiIssueView]:
    """Set an issue's status (``pending`` / ``ignored`` / ``resolved``)."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    updated = await update_issue_status(
        issue_repo=issue_repo,
        issue_id=issue_id.strip(),
        status=body.status.strip(),
    )
    return WikiEnvelope(success=True, data=issue_to_view(updated))


@router.post("/revert", response_model=WikiEnvelope[WikiPageView])
async def revert_page(
    _auth: AuthDep,
    _role: RoleAdminDep,
    kb_id: str,
    body: WikiRevertRequest,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    user_id: _PrincipalUser,
) -> WikiEnvelope[WikiPageView]:
    """Restore a stored snapshot as a new edit. Missing version is 404."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
        write=True,
    )
    page = await service.revert_page(
        knowledge_base_id=kb_id,
        slug=body.slug.strip(),
        version=body.version,
        editor_id=_actor(user_id),
    )
    return WikiEnvelope(success=True, data=page_to_view(page))


@router.get(
    "/revisions/{slug:path}",
    response_model=WikiEnvelope[WikiRevisionListData | WikiRevisionView],
)
async def get_revisions(
    _auth: AuthDep,
    _role: RoleViewerDep,
    kb_id: str,
    slug: str,
    kb_service: KBServiceDep,
    kb_share_service: KBShareServiceDep,
    service: WikiPageServiceDep,
    version: int | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> WikiEnvelope[WikiRevisionListData | WikiRevisionView]:
    """List snapshots, or return one snapshot with content when ``version`` is set."""
    await require_wiki_kb(
        kb_id=kb_id,
        kb_service=kb_service,
        kb_share_service=kb_share_service,
    )
    slug = slug.strip()
    if version is not None:
        revision = await service.get_revision(knowledge_base_id=kb_id, slug=slug, version=version)
        return WikiEnvelope(success=True, data=revision_to_view(revision))
    listing = await service.list_revisions(
        knowledge_base_id=kb_id,
        slug=slug,
        limit=limit,
        offset=offset,
    )
    return WikiEnvelope(success=True, data=revision_list_to_view(listing))


__all__ = ["router"]
