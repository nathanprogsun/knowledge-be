"""Agent share service — cross-tenant agent sharing operations.

Request-scoped: constructed per request by the organizations factory
with fresh repositories on the shared ``AsyncSession``; the web layer
never imports ``db`` directly.

The service orchestrates the cross-tenant agent share lifecycle:
sharing a custom agent into an organization, revoking the grant, and
resolving which shared agents a tenant can reach through the
organizations it participates in. Agent shares are read-only grants:
the permission is forced to ``viewer`` regardless of what the caller
requests, mirroring the upstream contract.
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
from src.core.sharing.types import AgentShareInfo
from src.db.dao.agent_share_repository import AgentShareRepository
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.dao.organization_repository import (
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.agent_share import (
    SHARE_PERMISSION_VIEWER,
    AgentShare,
)
from src.db.models.organization import (
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    has_org_permission,
)

_AGENT_NOT_FOUND_CODE = "agent.not_found"
_ORG_NOT_FOUND_CODE = "organization.not_found"
_SHARE_NOT_FOUND_CODE = "agent_share.not_found"


@runtime_checkable
class AgentShareService(Protocol):
    """Cross-tenant agent sharing operations."""

    async def share_agent(
        self,
        *,
        agent_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        permission: str,
    ) -> AgentShareInfo: ...

    async def update_share_permission(
        self,
        *,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None: ...

    async def remove_share(
        self,
        *,
        share_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None: ...

    async def get_share(self, *, share_id: str) -> AgentShareInfo: ...

    async def get_share_by_agent_and_org(
        self,
        *,
        agent_id: str,
        organization_id: str,
    ) -> AgentShareInfo: ...

    async def list_shares_by_agent(self, *, agent_id: str) -> list[AgentShareInfo]: ...

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[AgentShareInfo]: ...


class AgentShareServiceImpl:
    """Concrete agent share service over the storage repositories."""

    def __init__(
        self,
        *,
        agent_repo: CustomAgentRepository,
        org_repo: OrganizationRepository,
        member_repo: OrganizationMemberRepository,
        share_repo: AgentShareRepository,
    ) -> None:
        self._agent_repo = agent_repo
        self._org_repo = org_repo
        self._member_repo = member_repo
        self._share_repo = share_repo

    # ── Share ──────────────────────────────────────────────────────

    async def share_agent(
        self,
        *,
        agent_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        permission: str,
    ) -> AgentShareInfo:
        """Share an owned agent into an organization as a read-only grant.

        The agent must exist and belong to the caller's tenant, be
        configured with a chat model, and the caller's tenant must hold
        an editor-or-higher role inside the target organization. The
        grant is forced to ``viewer``; a duplicate share is upgraded to
        ``viewer`` instead of raising.
        """
        agent = await self._agent_repo.get_by_id_and_tenant(id=agent_id, tenant_id=tenant_id)
        if agent is None:
            raise NotFoundError(
                code=_AGENT_NOT_FOUND_CODE,
                message=f"agent {agent_id} not found",
            )
        if agent.tenant_id != tenant_id:
            raise PermissionDeniedError(
                code="agent_share.not_owner",
                message="only the agent owner can share it",
            )
        if not agent.config.get("model_id"):
            raise ValidationError(
                code="agent_share.agent_not_configured",
                message="agent is not fully configured: set the chat model before sharing",
            )

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
                code="agent_share.tenant_not_in_org",
                message="caller's tenant is not a member of the organization",
            )
        if not has_org_permission(member.role, ORG_ROLE_EDITOR):
            raise PermissionDeniedError(
                code="agent_share.org_role_cannot_share",
                message="only editors and admins can share agents to this organization",
            )

        now = _now()
        share = AgentShare(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            organization_id=organization_id,
            shared_by_user_id=user_id,
            source_tenant_id=tenant_id,
            permission=SHARE_PERMISSION_VIEWER,
            created_at=now,
            updated_at=now,
        )
        created = await self._share_repo.create_or_none(share)
        if created is not None:
            return AgentShareInfo.map_from_db(created)

        # Duplicate live share: upgrade the existing row to viewer.
        existing = await self._share_repo.get_by_agent_and_org_or_none(
            agent_id=agent_id,
            organization_id=organization_id,
        )
        if existing is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message="agent share not found",
            )
        upgraded = existing.model_copy(
            update={"permission": SHARE_PERMISSION_VIEWER, "updated_at": now}
        )
        updated = await self._share_repo.update(upgraded)
        return AgentShareInfo.map_from_db(updated)

    # ── Update ──────────────────────────────────────────────────────

    async def update_share_permission(
        self,
        *,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None:
        """Update a share's permission.

        Agent shares are read-only grants, so the permission is forced
        to ``viewer``. The caller must be the original sharer, a
        source-tenant admin, or an admin of the target organization.
        """
        share = await self._share_repo.get_by_id_or_none(share_id)
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"agent share {share_id} not found",
            )
        await self._authorize_manage(
            share, user_id=user_id, tenant_id=tenant_id, tenant_role=tenant_role
        )
        updated = share.model_copy(
            update={"permission": SHARE_PERMISSION_VIEWER, "updated_at": _now()}
        )
        await self._share_repo.update(updated)

    # ── Remove ──────────────────────────────────────────────────────

    async def remove_share(
        self,
        *,
        share_id: str,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None:
        """Revoke a share.

        The caller must be the original sharer, a source-tenant admin,
        or an admin of the target organization.
        """
        share = await self._share_repo.get_by_id_or_none(share_id)
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"agent share {share_id} not found",
            )
        await self._authorize_manage(
            share, user_id=user_id, tenant_id=tenant_id, tenant_role=tenant_role
        )
        await self._share_repo.soft_delete(id=share_id, now=_now())

    # ── Reads ───────────────────────────────────────────────────────

    async def get_share(self, *, share_id: str) -> AgentShareInfo:
        """Return one live share row, or raise ``NotFoundError``."""
        share = await self._share_repo.get_by_id_or_none(share_id)
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"agent share {share_id} not found",
            )
        return AgentShareInfo.map_from_db(share)

    async def get_share_by_agent_and_org(
        self,
        *,
        agent_id: str,
        organization_id: str,
    ) -> AgentShareInfo:
        """Return the live share for the (agent, org) pair, or raise."""
        share = await self._share_repo.get_by_agent_and_org_or_none(
            agent_id=agent_id,
            organization_id=organization_id,
        )
        if share is None:
            raise NotFoundError(
                code=_SHARE_NOT_FOUND_CODE,
                message=f"agent {agent_id} is not shared into {organization_id}",
            )
        return AgentShareInfo.map_from_db(share)

    async def list_shares_by_agent(self, *, agent_id: str) -> list[AgentShareInfo]:
        """Return every live share of one agent, newest first."""
        rows = await self._share_repo.list_by_agent(agent_id)
        return [AgentShareInfo.map_from_db(row) for row in rows]

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[AgentShareInfo]:
        """Return every live share into one organization, newest first."""
        rows = await self._share_repo.list_by_organization(organization_id)
        return [AgentShareInfo.map_from_db(row) for row in rows]

    # ── Authorization ───────────────────────────────────────────────

    async def _authorize_manage(
        self,
        share: AgentShare,
        *,
        user_id: str,
        tenant_id: int,
        tenant_role: str,
    ) -> None:
        """Reject a caller who cannot manage ``share``.

        Allowed: the original sharer, a source-tenant admin, or an
        admin of the target organization.
        """
        if share.shared_by_user_id == user_id:
            return
        if (
            tenant_id != 0
            and tenant_id == share.source_tenant_id
            and tenant_role == TenantRole.ADMIN
        ):
            return
        member = await self._member_repo.get_member(
            organization_id=share.organization_id,
            tenant_id=tenant_id,
        )
        if member is not None and member.role == ORG_ROLE_ADMIN:
            return
        raise PermissionDeniedError(
            code="agent_share.permission_denied",
            message="only the sharer, a source-tenant admin, or an org admin can manage this share",
        )


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


__all__ = ["AgentShareService", "AgentShareServiceImpl"]
