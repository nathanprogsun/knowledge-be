"""Agent share service — interface and skeleton.

The full implementation lands in a later PR. This module pins the
service contract (the ``AgentShareService`` protocol) and carries an
empty concrete skeleton so the web layer can depend on the seam without
fabricating business logic.

The service orchestrates the cross-tenant agent share lifecycle:
sharing a custom agent into an organization, revoking the grant, and
resolving which shared agents a tenant can reach through the
organizations it participates in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.db.models.agent_share import AgentShare


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
    ) -> AgentShare: ...

    async def remove_share(self, *, share_id: str, user_id: str, tenant_id: int) -> None: ...

    async def get_share(self, *, share_id: str) -> AgentShare: ...

    async def get_share_by_agent_and_org(
        self,
        *,
        agent_id: str,
        organization_id: str,
    ) -> AgentShare: ...

    async def list_shares_by_agent(self, *, agent_id: str) -> list[AgentShare]: ...

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[AgentShare]: ...


class AgentShareServiceImpl:
    """Concrete agent share service — implemented in a later PR."""

    async def share_agent(
        self,
        *,
        agent_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        permission: str,
    ) -> AgentShare:
        raise NotImplementedError

    async def remove_share(self, *, share_id: str, user_id: str, tenant_id: int) -> None:
        raise NotImplementedError

    async def get_share(self, *, share_id: str) -> AgentShare:
        raise NotImplementedError

    async def get_share_by_agent_and_org(
        self,
        *,
        agent_id: str,
        organization_id: str,
    ) -> AgentShare:
        raise NotImplementedError

    async def list_shares_by_agent(self, *, agent_id: str) -> list[AgentShare]:
        raise NotImplementedError

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[AgentShare]:
        raise NotImplementedError


__all__ = ["AgentShareService", "AgentShareServiceImpl"]
