"""Wiki-domain FastAPI dependency factories.

One-line forwarders to ``src.core.knowledge.wiki.factory`` (plus the
knowledge-base factory for the lint seam): repositories are assembled
in ``core`` on the request-scoped ``AsyncSession`` so the request's
reads and writes share one transactional unit of work. ``web`` never
imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.wiki.factory import (
    build_wiki_folder_service,
    build_wiki_lint_service,
    build_wiki_page_service,
)
from src.core.knowledge.wiki.folders import WikiFolderService
from src.core.knowledge.wiki.lint_service import WikiLintService
from src.core.knowledge.wiki.page_service import WikiPageService
from src.web.deps.session import SessionDep


def get_kb_service(session: SessionDep) -> KBService:
    """Build a per-request ``KBService`` on the shared session."""
    return build_kb_service(session)


def get_wiki_page_service(session: SessionDep) -> WikiPageService:
    """Build a per-request ``WikiPageService`` on the shared session."""
    return build_wiki_page_service(session)


def get_wiki_folder_service(session: SessionDep) -> WikiFolderService:
    """Build a per-request ``WikiFolderService`` on the shared session."""
    return build_wiki_folder_service(session)


def get_wiki_lint_service(session: SessionDep) -> WikiLintService:
    """Build a per-request ``WikiLintService`` on the shared session."""
    return build_wiki_lint_service(session)


KBServiceDep = Annotated[KBService, Depends(get_kb_service)]
WikiPageServiceDep = Annotated[WikiPageService, Depends(get_wiki_page_service)]
WikiFolderServiceDep = Annotated[WikiFolderService, Depends(get_wiki_folder_service)]
WikiLintServiceDep = Annotated[WikiLintService, Depends(get_wiki_lint_service)]

__all__ = [
    "KBServiceDep",
    "WikiFolderServiceDep",
    "WikiLintServiceDep",
    "WikiPageServiceDep",
    "get_kb_service",
    "get_wiki_folder_service",
    "get_wiki_lint_service",
    "get_wiki_page_service",
]
