"""Knowledge-base share service — interface and skeleton.

The full implementation lands in a later PR. This module pins the
service contract (the ``KBShareService`` protocol) and carries an empty
concrete skeleton so the web layer can depend on the seam without
fabricating business logic.

The service orchestrates the cross-tenant share lifecycle: sharing a
knowledge base into an organization, updating / revoking the grant, and
resolving the effective permission a tenant holds on a shared knowledge
base (capped by the receiver's own role inside the organization).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.db.models.kb_share import KnowledgeBaseShare


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
        permission: str,
    ) -> KnowledgeBaseShare: ...

    async def update_share_permission(
        self,
        *,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
    ) -> None: ...

    async def remove_share(self, *, share_id: str, user_id: str, tenant_id: int) -> None: ...

    async def get_share(self, *, share_id: str) -> KnowledgeBaseShare: ...

    async def get_share_by_kb_and_org(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShare: ...

    async def list_shares_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> list[KnowledgeBaseShare]: ...

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[KnowledgeBaseShare]: ...


class KBShareServiceImpl:
    """Concrete share service — implemented in a later PR."""

    async def share_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
        user_id: str,
        tenant_id: int,
        permission: str,
    ) -> KnowledgeBaseShare:
        raise NotImplementedError

    async def update_share_permission(
        self,
        *,
        share_id: str,
        permission: str,
        user_id: str,
        tenant_id: int,
    ) -> None:
        raise NotImplementedError

    async def remove_share(self, *, share_id: str, user_id: str, tenant_id: int) -> None:
        raise NotImplementedError

    async def get_share(self, *, share_id: str) -> KnowledgeBaseShare:
        raise NotImplementedError

    async def get_share_by_kb_and_org(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShare:
        raise NotImplementedError

    async def list_shares_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> list[KnowledgeBaseShare]:
        raise NotImplementedError

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[KnowledgeBaseShare]:
        raise NotImplementedError


__all__ = ["KBShareService", "KBShareServiceImpl"]
