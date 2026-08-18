"""Tenants-domain FastAPI dependency factories.

One-line forwarders to ``src.core.tenants.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.factory import (
    build_tenant_api_key_service,
    build_tenant_invitation_service,
    build_tenant_kv_service,
    build_tenant_member_service,
    build_tenant_service,
)
from src.core.tenants.invitation_service import TenantInvitationService
from src.core.tenants.kv_service import TenantKVService
from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.service import TenantService
from src.web.deps.session import SessionDep


def get_tenant_service(session: SessionDep) -> TenantService:
    """Build a per-request ``TenantService`` on the shared session."""
    return build_tenant_service(session)


def get_tenant_api_key_service(session: SessionDep) -> TenantAPIKeyService:
    """Build a per-request ``TenantAPIKeyService`` on the shared session."""
    return build_tenant_api_key_service(session)


def get_tenant_kv_service(session: SessionDep) -> TenantKVService:
    """Build a per-request ``TenantKVService`` on the shared session."""
    return build_tenant_kv_service(session)


def get_tenant_member_service(session: SessionDep) -> TenantMemberService:
    """Build a per-request ``TenantMemberService`` on the shared session."""
    return build_tenant_member_service(session)


def get_tenant_invitation_service(session: SessionDep) -> TenantInvitationService:
    """Build a per-request ``TenantInvitationService`` on the shared session."""
    return build_tenant_invitation_service(session)


TenantServiceDep = Annotated[TenantService, Depends(get_tenant_service)]
TenantAPIKeyServiceDep = Annotated[TenantAPIKeyService, Depends(get_tenant_api_key_service)]
TenantKVServiceDep = Annotated[TenantKVService, Depends(get_tenant_kv_service)]
TenantMemberServiceDep = Annotated[TenantMemberService, Depends(get_tenant_member_service)]
TenantInvitationServiceDep = Annotated[
    TenantInvitationService, Depends(get_tenant_invitation_service)
]


__all__ = [
    "TenantAPIKeyServiceDep",
    "TenantInvitationServiceDep",
    "TenantKVServiceDep",
    "TenantMemberServiceDep",
    "TenantServiceDep",
    "get_tenant_api_key_service",
    "get_tenant_invitation_service",
    "get_tenant_kv_service",
    "get_tenant_member_service",
    "get_tenant_service",
]
