"""Knowledge-base share service — interface and implementation.

The service orchestrates the cross-tenant share lifecycle: sharing a
knowledge base into an organization, updating / revoking the grant, and
resolving the effective permission a tenant holds on a shared knowledge
base (capped by the receiver's own role inside the organization).

Read-side operations (share lookups and the access predicate the web
gates rely on) are implemented against ``KBShareRepository``. The
mutating lifecycle (create / update / revoke a grant) still lands in a
later change and raises ``NotImplementedError`` until then.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.common.exception import NotFoundError
from src.db.dao.kb_share_repository import KBShareRepository
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

    async def can_access_knowledge_base(
        self,
        *,
        tenant_id: int,
        owner_tenant_id: int,
        knowledge_base_id: str,
    ) -> bool: ...


class KBShareServiceImpl:
    """Repository-backed share service (read side implemented)."""

    def __init__(self, *, kb_share_repo: KBShareRepository) -> None:
        self._kb_share_repo = kb_share_repo

    # ── Access predicate ──────────────────────────────────────────

    async def can_access_knowledge_base(
        self,
        *,
        tenant_id: int,
        owner_tenant_id: int,
        knowledge_base_id: str,
    ) -> bool:
        """Whether ``tenant_id`` may read ``knowledge_base_id``.

        The owner tenant always has access. Any other tenant needs a
        live share grant into an organization it belongs to — the
        repository's ``list_shared_for_tenant`` already joins the
        share over organization membership and drops grants whose
        organization or knowledge base was soft-deleted.
        """
        if tenant_id == owner_tenant_id:
            return True
        shares = await self._kb_share_repo.list_shared_for_tenant(tenant_id)
        return any(share.knowledge_base_id == knowledge_base_id for share in shares)

    # ── Read side ─────────────────────────────────────────────────

    async def get_share(self, *, share_id: str) -> KnowledgeBaseShare:
        """Return one live share by id, or raise ``NotFoundError``."""
        share = await self._kb_share_repo.get_by_id_or_none(share_id)
        if share is None:
            raise NotFoundError(
                code="share.not_found",
                message=f"share {share_id} not found",
            )
        return share

    async def get_share_by_kb_and_org(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShare:
        """Return the live grant for a (KB, organization) pair."""
        share = await self._kb_share_repo.get_by_kb_and_org_or_none(
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
        )
        if share is None:
            raise NotFoundError(
                code="share.not_found",
                message="share not found for knowledge base and organization",
            )
        return share

    async def list_shares_by_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> list[KnowledgeBaseShare]:
        """Live grants of one KB; caller must own the KB."""
        return await self._kb_share_repo.list_by_knowledge_base(knowledge_base_id)

    async def list_shares_by_organization(
        self,
        *,
        organization_id: str,
    ) -> list[KnowledgeBaseShare]:
        """Live grants held by one organization."""
        return await self._kb_share_repo.list_by_organization(organization_id)

    # ── Mutating lifecycle — lands with the share-management change ──

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


__all__ = ["KBShareService", "KBShareServiceImpl"]
