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
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ai.graph.neo4j_repo import build_graph_repository
from src.ai.mcp_transport import MCPConnectionManager
from src.app_context.registry import LifeSpanService
from src.app_logging import configure_logging, logger
from src.common.oidc_client import OidcClient
from src.common.telemetry import instrument_engine, is_file_exporter, is_tracing_enabled, setup_tracing
from src.core.infra.mcp_services.oauth import (
    InMemorySecretStore,
    OAuthManager,
    OAuthStateStore,
)
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.core.infra.web_search.registry import build_web_search_client_registry
from src.db.base import DatabaseEngine
from src.db.dao.audit_log_repository import AuditLogRepository
from src.settings import get_settings
from src.workers.settings import get_worker_settings
from src.web.api.agents.router import router as agents_router
from src.web.api.agents.skill_views import skill_router as skills_router
from src.web.api.auth.router import router as auth_router
from src.web.api.channels.embed.router import (
    agents_router as embed_agents_router,
)
from src.web.api.channels.embed.router import (
    public_router as embed_public_router,
)
from src.web.api.channels.embed.router import router as embed_router
from src.web.api.channels.im.router import (
    agents_router as im_agents_router,
)
from src.web.api.channels.im.router import (
    callback_router as im_callback_router,
)
from src.web.api.channels.im.router import router as im_router
from src.web.api.channels.im.router import wechat_router as im_wechat_router
from src.web.api.chat.messages.router import (
    router as messages_router,
)
from src.web.api.chat.messages.router import (
    suggestion_router,
)
from src.web.api.chat.router import router as chat_router
from src.web.api.chat.sessions.router import router as sessions_router
from src.web.api.cloud.router import router as cloud_router
from src.web.api.evaluation.router import router as evaluation_router
from src.web.api.favorites.router import router as favorites_router
from src.web.api.files.router import bare_files_router, kb_files_router
from src.web.api.infra.datasources.router import router as datasources_router
from src.web.api.infra.initialization.router import router as initialization_router
from src.web.api.infra.mcp_services.oauth_callback_router import (
    router as mcp_oauth_callback_router,
)
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
from src.web.api.me.router import router as me_router
from src.web.api.organizations.router import router as organizations_router
from src.web.api.organizations.shared_router import (
    router as shared_resources_router,
)
from src.web.api.system.admin_views import router as system_admin_router
from src.web.api.system.router import info_router as system_info_router
from src.web.api.system.router import router as system_router
from src.web.api.system.service_views import router as system_service_router
from src.web.api.tenants.router import router as tenants_router
from src.web.exception_handler import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down app-scoped resources."""
    configure_logging()
    settings = get_settings()
    logger.info("starting {}", settings.app_name)

    # Record the boot instant for ``GET /system/info`` (mirrors the
    # upstream ``runtime.ServerStartedAt()`` value); the per-request
    # ``SystemInfoService`` reads it back from ``app.state`` so the
    # uptime calculation reflects process lifetime, not request time.
    app.state.started_at = datetime.now(UTC)

    db_engine = DatabaseEngine(
        url=settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    if settings.db_conn_prewarm:
        await db_engine.prewarm()

    # Tracing instruments the SQLAlchemy engine; the FastAPI/httpx side
    # ran earlier in create_app (instrumenting from the lifespan is too
    # late — uvicorn has already built the middleware stack).
    instrument_engine(db_engine.engine)

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

    # OTel must instrument before the server builds the middleware stack;
    # no-op unless OTEL_ENABLED is set.
    setup_tracing(application)

    # All API routes live under /api/v1, matching the upstream contract
    # (the frontend and nginx proxy send /api/v1/... unchanged). /health
    # stays bare below for infra probes.
    api_v1_prefix = "/api/v1"
    api_routers = [
        agents_router,
        auth_router,
        chat_router,
        chunker_router,
        chunks_router,
        datasources_router,
        embed_agents_router,
        embed_public_router,
        embed_router,
        evaluation_router,
        faq_import_progress_router,
        faq_router,
        favorites_router,
        im_agents_router,
        im_callback_router,
        im_router,
        im_wechat_router,
        initialization_router,
        kb_files_router,
        knowledge_bases_router,
        kb_documents_router,
        knowledge_tags_router,
        mcp_services_router,
        mcp_oauth_callback_router,
        me_router,
        messages_router,
        models_router,
        organizations_router,
        sessions_router,
        shared_resources_router,
        skills_router,
        storage_backends_router,
        suggestion_router,
        system_admin_router,
        system_info_router,
        system_router,
        system_service_router,
        tenants_router,
        vector_stores_router,
        cloud_router,
        web_search_catalog_router,
        web_search_router,
        documents_router,
        wiki_router,
    ]
    for router in api_routers:
        application.include_router(router, prefix=api_v1_prefix)

    # The tenant-scoped storage proxy lives outside /api/v1 so nginx and
    # the dev proxy forward /files unchanged (same as the upstream layout).
    application.include_router(bare_files_router)

    @application.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/v1/meta/capabilities", tags=["meta"])
    async def capabilities() -> dict[str, str | int | bool | list[str]]:
        """Self-describing capability manifest.

        Agents and operators read this endpoint to discover which
        routes exist, which feature flags are on, and which optional
        services (OIDC, OTel, ARQ worker) are wired. Designed to be
        cheap (no DB / no Redis) so it can be polled.
        """
        routes: set[str] = set()

        def _walk(route_list: list, prefix: str = "") -> None:
            for r in route_list:
                path = getattr(r, "path", None)
                methods = getattr(r, "methods", None)
                # FastAPI's ``_IncludedRouter`` wraps every router passed
                # to ``app.include_router`` and exposes its child routes
                # via ``.original_router.routes``. The mount prefix
                # (``include_context.prefix``) is what FastAPI prepends
                # to child paths at serve time; merge it so the manifest
                # reports the fully qualified path agents would call.
                if type(r).__name__ == "_IncludedRouter":
                    inner = getattr(r, "original_router", None)
                    ic_prefix = ""
                    ic = getattr(r, "include_context", None)
                    if ic is not None:
                        ic_prefix = getattr(ic, "prefix", "") or ""
                    if inner is not None:
                        _walk(list(getattr(inner, "routes", [])), prefix + ic_prefix)
                    continue
                child_routes = getattr(r, "routes", None)
                if child_routes and not methods and isinstance(child_routes, list):
                    _walk(child_routes, prefix + (path or ""))
                    continue
                if not methods or not path or not isinstance(methods, set):
                    continue
                cleaned = methods - {"HEAD", "OPTIONS"}
                if not cleaned:
                    continue
                routes.add(f"{','.join(sorted(cleaned))} {prefix}{path}")

        _walk(application.routes)
        return {
            "service": "knowledge-be",
            "version": "0.2.0",
            "api_prefix": api_v1_prefix,
            "total_routes": len(routes),
            "routes": sorted(routes),
            "tracing_enabled": is_tracing_enabled(),
            "file_exporter": is_file_exporter(),
            "worker_configured": get_worker_settings().redis_url != "",
        }

    return application


# Module-level instance so `uvicorn src.main:app` and `uvicorn src.app_context.lifespan:app` both work.
app = create_app()


__all__ = [
    "app",
    "create_app",
    "lifespan",
]
