"""Factories assembling the sharing services on a per-request session.

Repositories bind the request-scoped ``AsyncSession``; the web layer
obtains services only through these builders so ``web`` never imports
``db`` itself.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.sharing.kb_share_service import KBShareServiceImpl
from src.db.dao.kb_share_repository import KBShareRepository


def build_kb_share_service(session: AsyncSession) -> KBShareServiceImpl:
    """Build the per-request KB share service on the shared session."""
    return KBShareServiceImpl(kb_share_repo=KBShareRepository(session))


__all__ = ["build_kb_share_service"]
