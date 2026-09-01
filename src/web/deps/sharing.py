"""Dependency forwarders for the sharing services on the web layer."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.sharing.factory import build_kb_share_service
from src.core.sharing.kb_share_service import KBShareServiceImpl
from src.web.deps.session import SessionDep


def get_kb_share_service(session: SessionDep) -> KBShareServiceImpl:
    """Build the per-request KB share service on the shared session."""
    return build_kb_share_service(session)


KBShareServiceDep = Annotated[KBShareServiceImpl, Depends(get_kb_share_service)]

__all__ = [
    "KBShareServiceDep",
    "get_kb_share_service",
]
