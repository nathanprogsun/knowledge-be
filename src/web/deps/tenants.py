"""Tenants-domain FastAPI dependency factories.

Builds request-scoped ``TenantService``, ``TenantAPIKeyService`` and
``TenantKVService`` on the shared ``AsyncSession``, following the same
pattern as ``deps/auth.py``: repositories are constructed per request, so
each service's reads and writes join the same transactional unit of work
as everything else in the request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.kv_service import TenantKVService
from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.service import TenantService
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.dao.tenant_kv_repository import TenantKVRepository
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.dao.tenants_repository import TenantRepository
from src.web.deps.session import SessionDep


def get_tenant_service(session: SessionDep) -> TenantService:
    """Build a per-request ``TenantService`` with a fresh repository."""
    return TenantService(tenants_repo=TenantRepository(session))


def get_tenant_api_key_service(session: SessionDep) -> TenantAPIKeyService:
    """Build a per-request ``TenantAPIKeyService`` with a fresh repository."""
    return TenantAPIKeyService(api_keys_repo=TenantAPIKeyRepository(session))


def get_tenant_kv_service(session: SessionDep) -> TenantKVService:
    """Build a per-request ``TenantKVService`` with a fresh repository."""
    return TenantKVService(kv_repo=TenantKVRepository(session))


def get_tenant_member_service(session: SessionDep) -> TenantMemberService:
    """Build a per-request ``TenantMemberService`` with a fresh repository."""
    return TenantMemberService(members_repo=TenantMemberRepository(session))


TenantServiceDep = Annotated[TenantService, Depends(get_tenant_service)]
TenantAPIKeyServiceDep = Annotated[TenantAPIKeyService, Depends(get_tenant_api_key_service)]
TenantKVServiceDep = Annotated[TenantKVService, Depends(get_tenant_kv_service)]
TenantMemberServiceDep = Annotated[TenantMemberService, Depends(get_tenant_member_service)]


__all__ = [
    "TenantAPIKeyServiceDep",
    "TenantKVServiceDep",
    "TenantMemberServiceDep",
    "TenantServiceDep",
    "get_tenant_api_key_service",
    "get_tenant_kv_service",
    "get_tenant_member_service",
    "get_tenant_service",
]
