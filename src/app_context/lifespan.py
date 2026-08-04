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
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app_context.registry import LifeSpanService
from src.app_logging import configure_logging, logger
from src.common.oidc_client import OidcClient
from src.db.base import DatabaseEngine
from src.settings import get_settings
from src.web.api.auth.router import router as auth_router
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

    app.state.lifespan_service = LifeSpanService(db_engine=db_engine, oidc_client=oidc_client)
    logger.info("lifespan ready")
    try:
        yield
    finally:
        logger.info("shutting down {}", settings.app_name)
        await oidc_client.aclose()
        await db_engine.close()
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

    application.include_router(auth_router)
    application.include_router(system_router)
    application.include_router(tenants_router)

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
