"""Database engine wrapper — async SQLAlchemy with explicit pool sizing.

Owns the `AsyncEngine`, an `async_sessionmaker`, and `prewarm()` /
`close()` lifecycle hooks called from the FastAPI lifespan.
"""

from __future__ import annotations

import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import Pool


class DatabaseEngine:
    """Process-wide async database engine.

    One instance lives on `app.state.lifespan_service.db_engine` for the
    lifetime of the FastAPI process. Repositories obtain the session
    factory via DI (`get_db_engine_from_lifespan` → `.session_factory`).
    """

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        poolclass: type[Pool] | None = None,
    ) -> None:
        # ``NullPool`` (no connection reuse) is used by the test suite so a
        # connection checked out in one event loop is never handed back in
        # another (asyncpg binds connections to the loop they were created
        # in). Production keeps the default ``QueuePool`` sizing below.
        if poolclass is not None:
            self._engine = create_async_engine(url, echo=echo, poolclass=poolclass)
        else:
            self._engine: AsyncEngine = create_async_engine(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def prewarm(self) -> None:
        """Acquire one connection so the pool is initialized at startup."""
        async with self._engine.connect() as conn:
            await conn.execute(sqlalchemy.text("SELECT 1"))

    async def close(self) -> None:
        await self._engine.dispose()

    async def healthcheck(self) -> bool:
        try:
            async with self._engine.connect() as conn:
                await conn.execute(sqlalchemy.text("SELECT 1"))
        except Exception:
            return False
        return True


__all__ = ["DatabaseEngine"]
