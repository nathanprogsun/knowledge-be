"""Async session context manager with commit/rollback semantics.

DAO functions accept an `AsyncSession` and never commit or rollback
themselves. `session_scope` is the canonical wrapper used by service
code to bound a unit of work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession inside a transaction.

    Commits on clean exit; rolls back on exception; always closes.
    """
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


__all__ = ["session_scope"]
