"""Web-search-domain FastAPI dependency factories.

One-line forwarders to ``src.core.infra.web_search.factory``: repositories
are assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.web_search.factory import build_web_search_provider_service
from src.core.infra.web_search.provider_service import WebSearchProviderService
from src.web.deps.session import SessionDep


def get_web_search_provider_service(
    session: SessionDep,
) -> WebSearchProviderService:
    """Build a per-request ``WebSearchProviderService`` on the shared session."""
    return build_web_search_provider_service(session)


WebSearchProviderServiceDep = Annotated[
    WebSearchProviderService,
    Depends(get_web_search_provider_service),
]


__all__ = [
    "WebSearchProviderServiceDep",
    "get_web_search_provider_service",
]
