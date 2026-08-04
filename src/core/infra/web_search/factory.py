"""Web-search-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.

The search service additionally takes a ``WebSearchClientRegistry`` —
typically the one built at lifespan time and stashed on
``app.state``. The factory accepts it as a parameter so callers (web
routers, tests) inject the right registry.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.infra.web_search.provider_service import (
    WebSearchClientRegistry,
    WebSearchProviderService,
)
from src.core.infra.web_search.search_service import (
    DEFAULT_TIMEOUT_SECONDS,
    WebSearchSearchService,
)
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository


def build_web_search_provider_service(session: AsyncSession) -> WebSearchProviderService:
    """Per-request ``WebSearchProviderService`` with a fresh repo."""
    return WebSearchProviderService(
        provider_repo=WebSearchProviderRepository(session),
    )


def build_web_search_search_service(
    session: AsyncSession,
    registry: WebSearchClientRegistry,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> WebSearchSearchService:
    """Per-request ``WebSearchSearchService`` with a fresh repo + shared registry."""
    return WebSearchSearchService(
        provider_repo=WebSearchProviderRepository(session),
        registry=registry,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "build_web_search_provider_service",
    "build_web_search_search_service",
]
