"""FastAPI app factory + lifespan + DI registry.

This module owns:
- `create_app()`: builds the FastAPI instance with middleware wired in.
- `lifespan`: async context manager that initializes and tears down
  long-lived resources (DB engine, HTTP client, ARQ pool) and populates
  `app.state.lifespan_service`.
- `LifeSpanService` and the `get_*_from_lifespan` factories live in
  `src.app_context.registry` (a dedicated module so `web.deps` can import
  the accessors without creating an import cycle with the app factory).

The router modules are imported at module top (not inside `create_app`) so
the anti-drift layer check can verify web -> core -> db directionality
without function-level imports. The routers themselves only reference
`web.deps`, which imports the DI accessors from `registry` — breaking the
former `lifespan` <-> `web.deps` cycle.

The lifespan also wires the live MCP transport layer (connection
pool, discovery + connectivity probes, OAuth state + secret stores)
into ``app.state.lifespan_service`` during startup; the matching
teardown in the ``finally`` block releases the connection pool's
cleanup loop and closes the OAuth HTTP client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ai.graph.neo4j_repo import build_graph_repository
from src.ai.mcp_transport import MCPConnectionManager
from src.app_context.registry import LifeSpanService
from src.app_logging import configure_logging, logger
from src.common.oidc_client import OidcClient
from src.core.infra.mcp_services.oauth import (
    InMemorySecretStore,
    OAuthManager,
    OAuthStateStore,
)
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.core.infra.web_search.registry import build_web_search_client_registry
from src.db.base import DatabaseEngine
from src.settings import get_settings
from src.web.api.agents.router import router as agents_router
from src.web.api.agents.skill_views import skill_router as skills_router
from src.web.api.auth.router import router as auth_router
from src.web.api.chat.messages.router import (
    router as messages_router,
)
from src.web.api.chat.messages.router import (
    suggestion_router,
)
from src.web.api.chat.router import router as chat_router
from src.web.api.chat.sessions.router import router as sessions_router
from src.web.api.infra.datasources.router import router as datasources_router
from src.web.api.infra.initialization.router import router as initialization_router
from src.web.api.infra.mcp_services.router import router as mcp_services_router
from src.web.api.infra.models.router import router as models_router
from src.web.api.infra.storage_backends.router import router as storage_backends_router
from src.web.api.infra.vector_stores.router import router as vector_stores_router
from src.web.api.infra.web_search.catalog_router import (
    router as web_search_catalog_router,
)
from src.web.api.infra.web_search.router import router as web_search_router
from src.web.api.knowledge.chunker.router import router as chunker_router
from src.web.api.knowledge.chunks.router import router as chunks_router
from src.web.api.knowledge.documents.router import (
    documents_router,
    kb_documents_router,
)
from src.web.api.knowledge.faq.router import (
    import_progress_router as faq_import_progress_router,
)
from src.web.api.knowledge.faq.router import router as faq_router
from src.web.api.knowledge.tags.router import router as knowledge_tags_router
from src.web.api.knowledge.wiki.router import router as wiki_router
from src.web.api.knowledge_bases.router import router as knowledge_bases_router
from src.web.api.system.router import router as system_router
from src.web.api.tenants.router import router as tenants_router
from src.web.exception_handler import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down app-scoped resources."""
    configure_logging()
    settings = get_settings()
    logger.info("starting {}", settings.app_name)

    db_engine = DatabaseEngine(
        url=settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    if settings.db_conn_prewarm:
        await db_engine.prewarm()

    # APP-scope singleton: pooled httpx.AsyncClient underneath; shared
    # across requests so TCP/TLS connections to the IdP are reused.
    oidc_client = OidcClient()

    # Live MCP singletons. The connection pool runs the
    # background sweep; the discovery + connectivity probes borrow it
    # per-request via ``infra_mcp`` (their DB-session-bound resolver
    # is request-scoped); the OAuth state + secret stores back the
    # per-service OAuthManager that ``infra_mcp`` constructs on demand.
    mcp_connection_manager = MCPConnectionManager()
    mcp_connection_manager.start_cleanup()

    oauth_state_store = OAuthStateStore()
    oauth_secret_store = InMemorySecretStore()
    oauth_http_client = httpx.AsyncClient(timeout=30.0)

    # Bind the per-request ``OAuthManager`` to APP-scope state. The
    # signature matches the ``OAuthManagerFactoryLike`` callable type
    # advertised by ``core.infra.mcp_services.factory``.
    async def _oauth_manager_factory(info: MCPServiceInfo) -> OAuthManager:
        return OAuthManager(
            service=info,
            http_client=oauth_http_client,
            secret_store=oauth_secret_store,
            state_store=oauth_state_store,
        )

    lifespan_service = LifeSpanService(
        db_engine=db_engine,
        oidc_client=oidc_client,
        mcp_connection_manager=mcp_connection_manager,
        mcp_oauth_state_store=oauth_state_store,
        mcp_oauth_secret_store=oauth_secret_store,
        mcp_oauth_http_client=oauth_http_client,
        mcp_oauth_manager_factory=_oauth_manager_factory,
    )
    app.state.lifespan_service = lifespan_service

    # Graph repository (Neo4j). Enabled only when ``NEO4J_ENABLE`` is
    # ``true``; a disabled deployment still registers a no-op repository
    # so callers can rely on ``app.state.graph_repository`` existing.
    graph_repository = await build_graph_repository()
    app.state.graph_repository = graph_repository

    # Web-search provider registry: one app-scope registry mapping every
    # supported provider type to its concrete HTTP client factory. The
    # registry is stateless (it holds factories only); a client is built
    # per search call by the dispatch service, so no shutdown hook is
    # needed here.
    app.state.web_search_client_registry = build_web_search_client_registry()
    logger.info("lifespan ready")
    try:
        yield
    finally:
        logger.info("shutting down {}", settings.app_name)
        await mcp_connection_manager.shutdown()
        await oauth_http_client.aclose()
        await oidc_client.aclose()
        await db_engine.close()
        await graph_repository.close()
        app.state.lifespan_service = None


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(application)

    application.include_router(agents_router)
    application.include_router(auth_router)
    application.include_router(chat_router)
    application.include_router(chunker_router)
    application.include_router(chunks_router)
    application.include_router(datasources_router)
    application.include_router(faq_import_progress_router)
    application.include_router(faq_router)
    application.include_router(initialization_router)
    application.include_router(knowledge_bases_router)
    application.include_router(kb_documents_router)
    application.include_router(knowledge_tags_router)
    application.include_router(mcp_services_router)
    application.include_router(messages_router)
    application.include_router(models_router)
    application.include_router(sessions_router)
    application.include_router(skills_router)
    application.include_router(storage_backends_router)
    application.include_router(suggestion_router)
    application.include_router(system_router)
    application.include_router(tenants_router)
    application.include_router(vector_stores_router)
    application.include_router(web_search_catalog_router)
    application.include_router(web_search_router)
    application.include_router(documents_router)
    application.include_router(wiki_router)

    @application.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


# Module-level instance so `uvicorn src.main:app` and `uvicorn src.app_context.lifespan:app` both work.
app = create_app()


__all__ = [
    "app",
    "create_app",
    "lifespan",
]
