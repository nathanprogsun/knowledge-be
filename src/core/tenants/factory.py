"""Tenants-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.kv_service import TenantKVService
from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.service import TenantService
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.dao.tenant_kv_repository import TenantKVRepository
from src.db.dao.tenant_members_repository import TenantMemberRepository
from src.db.dao.tenants_repository import TenantRepository


def build_tenant_service(session: AsyncSession) -> TenantService:
    """Per-request ``TenantService`` with a fresh repository."""
    return TenantService(
        tenants_repo=TenantRepository(session),
        members_repo=TenantMemberRepository(session),
    )


def build_tenant_api_key_service(session: AsyncSession) -> TenantAPIKeyService:
    """Per-request ``TenantAPIKeyService`` with a fresh repository."""
    return TenantAPIKeyService(api_keys_repo=TenantAPIKeyRepository(session))


def build_tenant_kv_service(session: AsyncSession) -> TenantKVService:
    """Per-request ``TenantKVService`` with a fresh repository."""
    return TenantKVService(kv_repo=TenantKVRepository(session))


def build_tenant_member_service(session: AsyncSession) -> TenantMemberService:
    """Per-request ``TenantMemberService`` with a fresh repository."""
    return TenantMemberService(members_repo=TenantMemberRepository(session))


__all__ = [
    "build_tenant_api_key_service",
    "build_tenant_kv_service",
    "build_tenant_member_service",
    "build_tenant_service",
]
