"""Unit tests for ``KBShareServiceImpl`` against an in-memory repository.

The DB-backed repository surface is covered by ``test_kb_share.py``;
here the access predicate and read-side lookups are exercised with a
recording fake only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import NotFoundError
from src.core.sharing.kb_share_service import KBShareServiceImpl
from src.db.dao.kb_share_repository import KBShareRepository
from src.db.models.kb_share import KnowledgeBaseShare

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _share(
    *,
    id: str,
    knowledge_base_id: str,
    organization_id: str = "org-1",
) -> KnowledgeBaseShare:
    return KnowledgeBaseShare(
        id=id,
        knowledge_base_id=knowledge_base_id,
        organization_id=organization_id,
        shared_by_user_id="usr-1",
        source_tenant_id=1,
        permission="viewer",
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeKBShareRepo(KBShareRepository):
    """In-memory stand-in for the share repository."""

    def __init__(self, shares: list[KnowledgeBaseShare]) -> None:
        self._shares = {s.id: s for s in shares}
        self._shared_for_tenant: dict[int, list[KnowledgeBaseShare]] = {}

    def grant(self, tenant_id: int, share: KnowledgeBaseShare) -> None:
        self._shared_for_tenant.setdefault(tenant_id, []).append(share)

    async def get_by_id_or_none(self, id: str) -> KnowledgeBaseShare | None:
        return self._shares.get(id)

    async def get_by_kb_and_org_or_none(
        self,
        *,
        knowledge_base_id: str,
        organization_id: str,
    ) -> KnowledgeBaseShare | None:
        for share in self._shares.values():
            if (
                share.knowledge_base_id == knowledge_base_id
                and share.organization_id == organization_id
            ):
                return share
        return None

    async def list_by_knowledge_base(
        self, knowledge_base_id: str
    ) -> list[KnowledgeBaseShare]:
        return [
            s for s in self._shares.values() if s.knowledge_base_id == knowledge_base_id
        ]

    async def list_by_organization(
        self, organization_id: str
    ) -> list[KnowledgeBaseShare]:
        return [
            s for s in self._shares.values() if s.organization_id == organization_id
        ]

    async def list_shared_for_tenant(self, tenant_id: int) -> list[KnowledgeBaseShare]:
        return list(self._shared_for_tenant.get(tenant_id, []))


async def test_owner_tenant_always_has_access() -> None:
    service = KBShareServiceImpl(kb_share_repo=_FakeKBShareRepo([]))
    assert await service.can_access_knowledge_base(
        tenant_id=7,
        owner_tenant_id=7,
        knowledge_base_id="kb-1",
    )


async def test_share_grant_allows_foreign_tenant_read() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-1")
    repo = _FakeKBShareRepo([share])
    repo.grant(9, share)
    service = KBShareServiceImpl(kb_share_repo=repo)

    assert await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_tenant_without_grant_is_denied() -> None:
    service = KBShareServiceImpl(kb_share_repo=_FakeKBShareRepo([]))
    assert not await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_grant_for_other_kb_does_not_leak() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-other")
    repo = _FakeKBShareRepo([share])
    repo.grant(9, share)
    service = KBShareServiceImpl(kb_share_repo=repo)

    assert not await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_get_share_missing_raises_not_found() -> None:
    service = KBShareServiceImpl(kb_share_repo=_FakeKBShareRepo([]))
    with pytest.raises(NotFoundError):
        await service.get_share(share_id="ghost")


async def test_get_share_by_kb_and_org_roundtrip() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-1", organization_id="org-2")
    service = KBShareServiceImpl(kb_share_repo=_FakeKBShareRepo([share]))

    found = await service.get_share_by_kb_and_org(
        knowledge_base_id="kb-1",
        organization_id="org-2",
    )
    assert found.id == "s-1"

    with pytest.raises(NotFoundError):
        await service.get_share_by_kb_and_org(
            knowledge_base_id="kb-1",
            organization_id="org-9",
        )
