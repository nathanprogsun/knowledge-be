"""FastAPI app factory + lifespan + DI registry.

This module owns:
- `create_app()`: builds the FastAPI instance with middleware wired in.
- `lifespan`: async context manager that initializes and tears down
  long-lived resources (DB engine, HTTP client, ARQ pool) and populates
  `app.state.lifespan_service`.
- `LifeSpanService`: dataclass registry of all domain services. Each domain
  service lands here as a singleton; web routers obtain them via the
  `get_xxx_from_lifespan` factories below.

Why a registry: avoids module-level globals while keeping DI explicit and
testable (monkeypatch `app.state.lifespan_service` in tests).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app_logging import configure_logging, logger
from src.db.base import DatabaseEngine
from src.settings import get_settings


@dataclass
class LifeSpanService:
    """Registry of singleton services.

    Populated during `lifespan` startup. Access via `get_xxx_from_lifespan`
    factories; never import this class directly from `web/` modules.
    """

    db_engine: DatabaseEngine | None = None


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

    app.state.lifespan_service = LifeSpanService(db_engine=db_engine)
    logger.info("lifespan ready")
    try:
        yield
    finally:
        logger.info("shutting down {}", settings.app_name)
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

    # Register the ApplicationError -> HTTP status handler.
    from src.web.exception_handler import register_exception_handlers

    register_exception_handlers(application)

    # Mount domain routers.
    from src.web.api.auth.router import router as auth_router
    from src.web.api.tenants.router import router as tenants_router

    application.include_router(auth_router)
    application.include_router(tenants_router)

    @application.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


# Module-level instance so `uvicorn src.main:app` and `uvicorn src.app_context.lifespan:app` both work.
app = create_app()


__all__ = [
    "LifeSpanService",
    "app",
    "create_app",
    "get_db_engine_from_lifespan",
    "get_lifespan_service",
    "lifespan",
]
