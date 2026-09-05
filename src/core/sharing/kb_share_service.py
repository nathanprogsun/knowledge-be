"""Knowledge-base share service — interface and implementation.

The service owns the cross-tenant share lifecycle: sharing a knowledge
base into an organization, updating or revoking the grant, and resolving
whether a tenant may read a shared knowledge base. The stored grant is
``viewer`` / ``editor`` / ``admin``. Effective permission for receivers
is capped later by ``SharedResourceService``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.common.exception import (
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.auth.permissions import TenantRole
from src.core.sharing.types import KnowledgeBaseShareInfo
from src.db.dao.kb_share_repository import KBShareRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.organization_repository import (
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.kb_share import SHARE_PERMISSIONS, KnowledgeBaseShare
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.organization import ORG_ROLE_VIEWER, has_org_permission

_KB_NOT_FOUND_CODE = "knowledge_base.not_found"
_ORG_NOT_FOUND_CODE = "organization.not_found"
_SHARE_NOT_FOUND_CODE = "kb_share.not_found"
_NOT_MEMBER_CODE = "organization.tenant_not_member"


@runtime_checkable
class KBShareService(Protocol):
    """Cross-tenant knowledge-base sharing operations."""

    async def share_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
        permission: str,
    ) -> KnowledgeBaseShareInfo: ...

    async def update_share_permission(
        self,
        *,
        knowledge_base_id: str,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None: ...

    async def remove_share(
        self,
        *,
        knowledge_base_id: str,
        share_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None: ...

    async def get_share(self, *, share_id: str) -> KnowledgeBaseShareInfo: ...

    async def get_share_by_kb_and_org(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShareInfo: ...

    async def list_shares_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> list[KnowledgeBaseShareInfo]: ...

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> list[KnowledgeBaseShareInfo]: ...

    async def can_access_knowledge_base(
        self,
        *,
        tenant_id: int,
        owner_tenant_id: int,
        knowledge_base_id: str,
    ) -> bool: ...


class KBShareServiceImpl:
    """Repository-backed share service."""

    def __init__(
        self,
        *,
        kb_share_repo: KBShareRepository,
        kb_repo: KnowledgeBaseRepository,
        org_repo: OrganizationRepository,
        member_repo: OrganizationMemberRepository,
    ) -> None:
        self._kb_share_repo = kb_share_repo
        self._kb_repo = kb_repo
        self._org_repo = org_repo
        self._member_repo = member_repo

    async def can_access_knowledge_base(
        self,
        *,
        tenant_id: int,
        owner_tenant_id: int,
        knowledge_base_id: str,
    ) -> bool:
        """Whether ``tenant_id`` may read ``knowledge_base_id``.

        The owner tenant always has access. Any other tenant needs a
        live share grant into an organization it belongs to.
        """
        if tenant_id == owner_tenant_id:
            return True
        shares = await self._kb_share_repo.list_shared_for_tenant(tenant_id)
        return any(share.knowledge_base_id == knowledge_base_id for share in shares)

    async def get_share(self, *, share_id: str) -> KnowledgeBaseShareInfo:
        """Return one live share by id, or raise ``NotFoundError``."""
        share = await self._live_share(share_id)
        return KnowledgeBaseShareInfo.map_from_db(share)

    async def get_share_by_kb_and_org(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShareInfo:
        """Return the live grant for a (KB, organization) pair."""
        share = await self._kb_share_repo.get_by_kb_and_org_or_none(
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
        )
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message="share not found for knowledge base and organization",
            )
        return KnowledgeBaseShareInfo.map_from_db(share)

    async def list_shares_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> list[KnowledgeBaseShareInfo]:
        """Live grants of one KB. Caller tenant must own the KB."""
        await self._owned_kb(knowledge_base_id, tenant_id)
        rows = await self._kb_share_repo.list_by_knowledge_base(knowledge_base_id)
        cache: dict[str, str | None] = {}
        out: list[KnowledgeBaseShareInfo] = []
        for row in rows:
            role = await self._member_role(row.organization_id, tenant_id, cache)
            out.append(_project(row, org_role=role, tenant_role=tenant_role))
        return out

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> list[KnowledgeBaseShareInfo]:
        """Live grants held by one organization. Caller must be a member."""
        org_role = await self._require_org_member(organization_id, tenant_id)
        rows = await self._kb_share_repo.list_by_organization(organization_id)
        return [_project(row, org_role=org_role, tenant_role=tenant_role) for row in rows]

    async def share_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
        permission: str,
    ) -> KnowledgeBaseShareInfo:
        """Share an owned knowledge base into an organization.

        Stores the requested grant when it is in ``SHARE_PERMISSIONS``.
        A live duplicate pair is upgraded to the new permission.
        """
        permission = _require_permission(permission)
        kb = await self._owned_kb(knowledge_base_id, tenant_id)
        _authorize_mutate_kb(kb, user_id=user_id, tenant_role=tenant_role)
        org_role = await self._require_source_tenant_in_org(organization_id, tenant_id)
        now = _now()
        row = KnowledgeBaseShare(
            id=str(uuid.uuid4()),
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
            shared_by_user_id=user_id,
            source_tenant_id=tenant_id,
            permission=permission,
            created_at=now,
            updated_at=now,
        )
        created = await self._kb_share_repo.create_or_none(row)
        if created is not None:
            return _project(created, org_role=org_role, tenant_role=tenant_role)
        return await self._upgrade_duplicate(
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
            permission=permission,
            org_role=org_role,
            tenant_role=tenant_role,
            now=now,
        )

    async def update_share_permission(
        self,
        *,
        knowledge_base_id: str,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None:
        """Update a live grant's stored permission."""
        permission = _require_permission(permission)
        share, _kb = await self._manageable_share(
            knowledge_base_id=knowledge_base_id,
            share_id=share_id,
            user_id=user_id,
            tenant_id=tenant_id,
            tenant_role=tenant_role,
        )
        await self._kb_share_repo.update(
            share.model_copy(update={"permission": permission, "updated_at": _now()})
        )

    async def remove_share(
        self,
        *,
        knowledge_base_id: str,
        share_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None:
        """Revoke a live grant."""
        await self._manageable_share(
            knowledge_base_id=knowledge_base_id,
            share_id=share_id,
            user_id=user_id,
            tenant_id=tenant_id,
            tenant_role=tenant_role,
        )
        await self._kb_share_repo.soft_delete(id=share_id, now=_now())

    async def _upgrade_duplicate(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
        permission: str,
        org_role: str,
        tenant_role: str,
        now: datetime,
    ) -> KnowledgeBaseShareInfo:
        existing = await self._kb_share_repo.get_by_kb_and_org_or_none(
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
        )
        if existing is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message="share not found for knowledge base and organization",
            )
        updated = await self._kb_share_repo.update(
            existing.model_copy(update={"permission": permission, "updated_at": now})
        )
        return _project(updated, org_role=org_role, tenant_role=tenant_role)

    async def _manageable_share(
        self,
        *,
        knowledge_base_id: str,
        share_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> tuple[KnowledgeBaseShare, KnowledgeBase]:
        share = await self._live_share(share_id)
        if share.knowledge_base_id != knowledge_base_id:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"share {share_id} not found",
            )
        kb = await self._owned_kb(knowledge_base_id, tenant_id)
        _authorize_manage(share, kb, user_id=user_id, tenant_role=tenant_role)
        return share, kb

    async def _live_share(self, share_id: str) -> KnowledgeBaseShare:
        share = await self._kb_share_repo.get_by_id_or_none(share_id)
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"share {share_id} not found",
            )
        return share

    async def _owned_kb(self, knowledge_base_id: str, tenant_id: int) -> KnowledgeBase:
        kb = await self._kb_repo.get_by_id_and_tenant(knowledge_base_id, tenant_id)
        if kb is None:
            raise NotFoundError(
                code=_KB_NOT_FOUND_CODE,
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return kb

    async def _require_source_tenant_in_org(self, organization_id: str, tenant_id: int) -> str:
        org = await self._org_repo.get_by_id_or_none(organization_id)
        if org is None:
            raise NotFoundError(
                code=_ORG_NOT_FOUND_CODE,
                message=f"organization {organization_id} not found",
            )
        member = await self._member_repo.get_member(
            organization_id=organization_id,
            tenant_id=tenant_id,
        )
        if member is None:
            raise PermissionDeniedError(
                code="kb_share.tenant_not_in_org",
                message="caller's tenant is not a member of the organization",
            )
        return member.role

    async def _require_org_member(self, organization_id: str, tenant_id: int) -> str:
        org = await self._org_repo.get_by_id_or_none(organization_id)
        if org is None:
            raise NotFoundError(
                code=_ORG_NOT_FOUND_CODE,
                message=f"organization {organization_id} not found",
            )
        member = await self._member_repo.get_member(
            organization_id=organization_id,
            tenant_id=tenant_id,
        )
        if member is None:
            raise NotFoundError(
                code=_NOT_MEMBER_CODE,
                message=f"tenant {tenant_id} is not a member of this organization",
            )
        return member.role

    async def _member_role(
        self,
        organization_id: str,
        tenant_id: int,
        cache: dict[str, str | None],
    ) -> str | None:
        if organization_id not in cache:
            member = await self._member_repo.get_member(
                organization_id=organization_id,
                tenant_id=tenant_id,
            )
            cache[organization_id] = member.role if member is not None else None
        return cache[organization_id]


def _require_permission(permission: str) -> str:
    if permission not in SHARE_PERMISSIONS:
        raise ValidationError(
            code="kb_share.invalid_permission",
            message=f"permission {permission!r} is not a knowledge-base share grant",
        )
    return permission


def _authorize_mutate_kb(kb: KnowledgeBase, *, user_id: str, tenant_role: str) -> None:
    if kb.creator_id and kb.creator_id == user_id:
        return
    if TenantRole.has_permission(tenant_role, TenantRole.ADMIN):
        return
    raise PermissionDeniedError(
        code="kb_share.permission_denied",
        message="only the knowledge-base creator or a tenant admin can manage shares",
    )


def _authorize_manage(
    share: KnowledgeBaseShare,
    kb: KnowledgeBase,
    *,
    user_id: str,
    tenant_role: str,
) -> None:
    if share.shared_by_user_id == user_id:
        return
    _authorize_mutate_kb(kb, user_id=user_id, tenant_role=tenant_role)


def _min_grant(left: str, right: str) -> str:
    if has_org_permission(left, right):
        return right
    return left


def _project(
    row: KnowledgeBaseShare,
    *,
    org_role: str | None,
    tenant_role: str,
) -> KnowledgeBaseShareInfo:
    info = KnowledgeBaseShareInfo.map_from_db(row)
    if not org_role:
        return info
    effective = _min_grant(row.permission, org_role)
    if tenant_role == TenantRole.VIEWER:
        effective = _min_grant(effective, ORG_ROLE_VIEWER)
    return info.model_copy(update={"my_role_in_org": org_role, "my_permission": effective})


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = ["KBShareService", "KBShareServiceImpl"]
