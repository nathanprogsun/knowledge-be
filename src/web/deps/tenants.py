"""Tenants-domain FastAPI dependency factories.

Builds a request-scoped ``TenantService`` on the shared ``AsyncSession``,
following the same pattern as ``deps/auth.py``: the repository is
constructed per request, so the service's reads and writes join the
same transactional unit of work as everything else in the request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.tenants.service import TenantService
from src.db.dao.tenants_repository import TenantRepository
from src.web.deps.session import SessionDep


def get_tenant_service(session: SessionDep) -> TenantService:
    """Build a per-request ``TenantService`` with a fresh repository."""
    return TenantService(tenants_repo=TenantRepository(session))


TenantServiceDep = Annotated[TenantService, Depends(get_tenant_service)]


__all__ = ["TenantServiceDep", "get_tenant_service"]
