"""Wiki-domain request-scoped service factories.

See ``src.core.tenants.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
Services are not registered in the lifespan registry — the web layer
consumes them via ``Depends`` on these factories.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.core.knowledge.wiki.folders import WikiFolderService
from src.core.knowledge.wiki.lint_service import WikiLintService
from src.core.knowledge.wiki.page_service import WikiPageService
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository


def build_wiki_page_service(session: AsyncSession) -> WikiPageService:
    """Per-request ``WikiPageService`` with fresh repositories."""
    return WikiPageService(
        page_repo=WikiPageRepository(session),
        folder_repo=WikiFolderRepository(session),
    )


def build_wiki_folder_service(session: AsyncSession) -> WikiFolderService:
    """Per-request ``WikiFolderService`` with fresh repositories."""
    return WikiFolderService(
        folder_repo=WikiFolderRepository(session),
        page_repo=WikiPageRepository(session),
    )


def build_wiki_lint_service(session: AsyncSession) -> WikiLintService:
    """Per-request ``WikiLintService`` composed from request-scoped services."""
    return WikiLintService(
        wiki_service=build_wiki_page_service(session),
        kb_service=build_kb_service(session),
    )


__all__ = [
    "build_wiki_folder_service",
    "build_wiki_lint_service",
    "build_wiki_page_service",
]
