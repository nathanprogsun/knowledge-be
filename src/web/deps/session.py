"""Per-request database session dependency.

One session per request, committed on clean exit and rolled back on
error. Every repository served by a request shares that session, so
the request's reads and writes form a single unit of work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context.lifespan import get_db_engine_from_lifespan


async def get_async_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a per-request ``AsyncSession`` with commit/rollback semantics."""
    db_engine = get_db_engine_from_lifespan(request.app)
    async with db_engine.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_async_session)]


__all__ = ["SessionDep", "get_async_session"]
