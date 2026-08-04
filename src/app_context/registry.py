"""APP-scope singleton registry + DI accessors.

``LifeSpanService`` holds **APP-scope** singletons only: stateless objects
that are expensive to construct (connection pools, TLS state). It is
populated during FastAPI lifespan startup and attached to
``app.state.lifespan_service``. Web routers obtain singletons via the
``get_xxx_from_lifespan`` factories here.

Scope rule (see AGENTS.md §3):

- APP scope (this registry): stateless + expensive — ``DatabaseEngine``,
  ``OidcClient``/``httpx.AsyncClient``, settings.
- REQUEST scope (``web.deps`` per-request construction): anything holding
  an ``AsyncSession`` — repositories and the services binding them.
  Request-scoped services MUST NOT be registered here.

It lives in its own module (rather than ``lifespan.py``) so ``web.deps``
can import the accessors without creating an import cycle with the app
factory (which mounts routers that import ``web.deps``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI

from src.common.oidc_client import OidcClient
from src.db.base import DatabaseEngine


@dataclass
class LifeSpanService:
    """Registry of APP-scope singletons.

    Populated during `lifespan` startup. Access via `get_xxx_from_lifespan`
    factories; never import this class directly from `web/` modules.
    """

    db_engine: DatabaseEngine | None = None
    oidc_client: OidcClient | None = None


def get_lifespan_service(app: FastAPI) -> LifeSpanService:
    """Return the lifespan service attached to the FastAPI app."""
    if not hasattr(app.state, "lifespan_service"):
        raise RuntimeError("LifeSpanService is not initialized — was the lifespan started?")
    return cast(LifeSpanService, app.state.lifespan_service)


def get_db_engine_from_lifespan(app: FastAPI) -> DatabaseEngine:
    """DI factory for the database engine."""
    service = get_lifespan_service(app)
    if service.db_engine is None:
        raise RuntimeError("DatabaseEngine is not initialized.")
    return service.db_engine


def get_oidc_client_from_lifespan(app: FastAPI) -> OidcClient:
    """DI factory for the shared ``OidcClient``.

    The client wraps a pooled ``httpx.AsyncClient``; sharing it across
    requests reuses TCP/TLS connections to the identity provider.
    """
    service = get_lifespan_service(app)
    if service.oidc_client is None:
        raise RuntimeError("OidcClient is not initialized.")
    return service.oidc_client


__all__ = [
    "LifeSpanService",
    "get_db_engine_from_lifespan",
    "get_lifespan_service",
    "get_oidc_client_from_lifespan",
]
