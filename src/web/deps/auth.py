"""Auth-domain FastAPI dependency factories.

Builds a per-request ``AuthService`` from fresh ``UserRepository`` and
``AuthTokenRepository`` instances sharing the request-scoped
``AsyncSession``. The service is request-scoped and never shared
across requests.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.auth.service import AuthService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.web.deps.session import SessionDep


def get_auth_service(session: SessionDep) -> AuthService:
    """Build a per-request ``AuthService`` with fresh repos sharing the session."""
    users_repo = UserRepository(session)
    tokens_repo = AuthTokenRepository(session)
    return AuthService(users_repo=users_repo, tokens_repo=tokens_repo)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


__all__ = ["AuthServiceDep", "get_auth_service"]
