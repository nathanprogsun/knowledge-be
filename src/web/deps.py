"""Web-layer FastAPI dependency factories.

Per the architecture convention (pr-plan.md §「架构约定」):
- ``get_async_session`` opens a per-request ``AsyncSession`` from the
  lifespan-held ``DatabaseEngine`` and commits/rolls back on exit.
- ``get_auth_service`` constructs fresh ``UserRepository`` +
  ``AuthTokenRepository`` sharing that session, then wraps them in an
  ``AuthService``. The service is request-scoped - never shared across
  requests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context.lifespan import get_db_engine_from_lifespan
from src.core.auth.service import AuthService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository


async def get_async_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a per-request ``AsyncSession`` with commit/rollback semantics.

    The session is created from the lifespan-held ``DatabaseEngine``'s
    session factory. On clean exit the session is committed; on any
    exception it is rolled back. Either way the session is closed.
    """
    db_engine = get_db_engine_from_lifespan(request.app)
    async with db_engine.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_async_session)]


def get_auth_service(session: SessionDep) -> AuthService:
    """Build a per-request ``AuthService`` with fresh repos.

    Both repos share the same ``AsyncSession`` so the service's reads
    and writes participate in one transactional unit of work.
    """
    users_repo = UserRepository(session)
    tokens_repo = AuthTokenRepository(session)
    return AuthService(users_repo=users_repo, tokens_repo=tokens_repo)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


__all__ = ["AuthServiceDep", "SessionDep", "get_async_session", "get_auth_service"]
