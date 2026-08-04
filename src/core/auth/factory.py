"""Auth-domain request-scoped service factories.

Repository construction belongs to ``core`` (AGENTS.md §1: ``web`` never
imports ``db``). Each factory builds fresh repositories on the caller's
per-request ``AsyncSession``, so every read/write in the request shares
one transactional unit of work. ``web/deps`` forwards to these builders
and contributes only the APP-scope singletons from the lifespan registry.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.oidc_client import OidcClient
from src.core.auth.oidc import OidcService
from src.core.auth.service import AuthService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository


def build_auth_service(session: AsyncSession) -> AuthService:
    """Per-request ``AuthService`` on the shared session."""
    return AuthService(
        users_repo=UserRepository(session),
        tokens_repo=AuthTokenRepository(session),
    )


def build_oidc_service(session: AsyncSession, *, oidc_client: OidcClient) -> OidcService:
    """Per-request ``OidcService``.

    ``oidc_client`` must be the APP-scope singleton from the lifespan
    registry (pooled ``httpx.AsyncClient``); the repos are request-scoped.
    """
    return OidcService(
        users_repo=UserRepository(session),
        tokens_repo=AuthTokenRepository(session),
        oidc_client=oidc_client,
    )


__all__ = ["build_auth_service", "build_oidc_service"]
