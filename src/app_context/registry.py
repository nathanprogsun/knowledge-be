"""APP-scope singleton registry + DI accessors.

``LifeSpanService`` holds **APP-scope** singletons only: stateless objects
that are expensive to construct (connection pools, TLS state). It is
populated during FastAPI lifespan startup and attached to
``app.state.lifespan_service``. Web routers obtain singletons via the
``get_xxx_from_lifespan`` factories here.

Scope rule (see AGENTS.md §3):

- APP scope (this registry): stateless + expensive — ``DatabaseEngine``,
  ``OidcClient``/``httpx.AsyncClient``, the MCP connection pool.
- REQUEST scope (``web.deps`` per-request construction): anything holding
  an ``AsyncSession`` — repositories and the services binding them.
  Request-scoped services MUST NOT be registered here.

The live MCP singletons the lifespan wires during startup are also
registered here: the connection pool, the OAuth state + secret stores, and a
factory that produces per-service :class:`OAuthManager` instances on
demand. All new fields default to ``None`` so legacy callers and tests
keep working.

It lives in its own module (rather than ``lifespan.py``) so ``web.deps``
can import the accessors without creating an import cycle with the app
factory (which mounts routers that import ``web.deps``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import httpx
from arq.connections import ArqRedis
from fastapi import FastAPI

from src.ai.mcp_transport import MCPConnectionManager
from src.common.oidc_client import OidcClient
from src.core.chat.stream.manager import MemoryStreamManager
from src.core.infra.mcp_services.oauth import (
    InMemorySecretStore,
    OAuthManager,
    OAuthStateStore,
)
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.db.base import DatabaseEngine


@dataclass
class LifeSpanService:
    """Registry of APP-scope singletons.

    Populated during `lifespan` startup. Access via `get_xxx_from_lifespan`
    factories; never import this class directly from `web/` modules.
    """

    db_engine: DatabaseEngine | None = None
    oidc_client: OidcClient | None = None
    # Live MCP transport singletons. Each entry is optional so a slim
    # deployment can ship without the live MCP transport layer.
    mcp_connection_manager: MCPConnectionManager | None = None
    mcp_oauth_state_store: OAuthStateStore | None = None
    mcp_oauth_secret_store: InMemorySecretStore | None = None
    mcp_oauth_http_client: httpx.AsyncClient | None = None
    mcp_oauth_manager_factory: Callable[[MCPServiceInfo], Awaitable[OAuthManager]] | None = None
    arq_redis: ArqRedis | None = None
    arq_queue_name: str = "arq:queue"
    # Process-local cancel flags. Stop and the in-flight QA loop must
    # share one instance or POST /stop cannot interrupt tokens.
    stream_manager: MemoryStreamManager | None = None


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


def get_stream_manager_from_lifespan(app: FastAPI) -> MemoryStreamManager:
    """DI factory for the shared in-process stream manager."""
    service = get_lifespan_service(app)
    if service.stream_manager is None:
        raise RuntimeError("MemoryStreamManager is not initialized.")
    return service.stream_manager


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
    "get_stream_manager_from_lifespan",
]
