"""Unit tests for ``KBShareServiceImpl`` against in-memory repositories."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, PermissionDeniedError, ValidationError
from src.core.sharing.kb_share_service import KBShareServiceImpl
from src.db.dao.kb_share_repository import KBShareRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.organization_repository import (
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.kb_share import KnowledgeBaseShare
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.organization import Organization, OrganizationTenantMember

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = 1
_OTHER_TENANT = 9
_USER = "usr-1"
_CREATOR = "usr-creator"


def _share(
    *,
    id: str,
    knowledge_base_id: str,
    organization_id: str = "org-1",
    permission: str = "viewer",
    shared_by_user_id: str = _USER,
    source_tenant_id: int = _TENANT,
) -> KnowledgeBaseShare:
    return KnowledgeBaseShare(
        id=id,
        knowledge_base_id=knowledge_base_id,
        organization_id=organization_id,
        shared_by_user_id=shared_by_user_id,
        source_tenant_id=source_tenant_id,
        permission=permission,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _kb(
    *,
    id: str = "kb-1",
    tenant_id: int = _TENANT,
    creator_id: str | None = _CREATOR,
) -> KnowledgeBase:
    return KnowledgeBase(
        id=id,
        name="KB",
        tenant_id=tenant_id,
        creator_id=creator_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _org(org_id: str = "org-1") -> Organization:
    return Organization(
        id=org_id,
        name="Org",
        owner_id=_USER,
        owner_tenant_id=_TENANT,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _member(*, org_id: str, tenant_id: int, role: str) -> OrganizationTenantMember:
    return OrganizationTenantMember(
        id=str(uuid.uuid4()),
        organization_id=org_id,
        tenant_id=tenant_id,
        role=role,
        representative_user_id=_USER,
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
        row = self._shares.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

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
                and share.deleted_at is None
            ):
                return share
        return None

    async def list_by_knowledge_base(self, knowledge_base_id: str) -> list[KnowledgeBaseShare]:
        return [
            s
            for s in self._shares.values()
            if s.knowledge_base_id == knowledge_base_id and s.deleted_at is None
        ]

    async def list_by_organization(self, organization_id: str) -> list[KnowledgeBaseShare]:
        return [
            s
            for s in self._shares.values()
            if s.organization_id == organization_id and s.deleted_at is None
        ]

    async def list_shared_for_tenant(self, tenant_id: int) -> list[KnowledgeBaseShare]:
        return list(self._shared_for_tenant.get(tenant_id, []))

    async def create_or_none(self, row: KnowledgeBaseShare) -> KnowledgeBaseShare | None:
        for share in self._shares.values():
            if (
                share.knowledge_base_id == row.knowledge_base_id
                and share.organization_id == row.organization_id
                and share.deleted_at is None
            ):
                return None
        self._shares[row.id] = row
        return row

    async def update(self, row: KnowledgeBaseShare) -> KnowledgeBaseShare:
        self._shares[row.id] = row
        return row

    async def soft_delete(self, *, id: str, now: datetime) -> bool:
        row = self._shares.get(id)
        if row is None or row.deleted_at is not None:
            return False
        self._shares[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
        return True


def _read_service(repo: _FakeKBShareRepo) -> KBShareServiceImpl:
    return KBShareServiceImpl(
        kb_share_repo=repo,
        kb_repo=AsyncMock(spec=KnowledgeBaseRepository),
        org_repo=AsyncMock(spec=OrganizationRepository),
        member_repo=AsyncMock(spec=OrganizationMemberRepository),
    )


def _make_service() -> tuple[KBShareServiceImpl, SimpleNamespace]:
    state = SimpleNamespace(
        kbs={},
        orgs={},
        members={},
        shares={},
    )
    share_repo = _FakeKBShareRepo([])
    state.shares = share_repo._shares

    kb_repo = AsyncMock(spec=KnowledgeBaseRepository)
    org_repo = AsyncMock(spec=OrganizationRepository)
    member_repo = AsyncMock(spec=OrganizationMemberRepository)

    async def _get_kb(id: str, tenant_id: int) -> KnowledgeBase | None:
        row = state.kbs.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def _get_org(id: str) -> Organization | None:
        row = state.orgs.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _get_member(organization_id: str, tenant_id: int) -> OrganizationTenantMember | None:
        return state.members.get((organization_id, tenant_id))

    kb_repo.get_by_id_and_tenant.side_effect = _get_kb
    org_repo.get_by_id_or_none.side_effect = _get_org
    member_repo.get_member.side_effect = _get_member

    service = KBShareServiceImpl(
        kb_share_repo=share_repo,
        kb_repo=kb_repo,
        org_repo=org_repo,
        member_repo=member_repo,
    )
    return service, state


async def test_owner_tenant_always_has_access() -> None:
    service = _read_service(_FakeKBShareRepo([]))
    assert await service.can_access_knowledge_base(
        tenant_id=7,
        owner_tenant_id=7,
        knowledge_base_id="kb-1",
    )


async def test_share_grant_allows_foreign_tenant_read() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-1")
    repo = _FakeKBShareRepo([share])
    repo.grant(9, share)
    service = _read_service(repo)

    assert await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_tenant_without_grant_is_denied() -> None:
    service = _read_service(_FakeKBShareRepo([]))
    assert not await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_grant_for_other_kb_does_not_leak() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-other")
    repo = _FakeKBShareRepo([share])
    repo.grant(9, share)
    service = _read_service(repo)

    assert not await service.can_access_knowledge_base(
        tenant_id=9,
        owner_tenant_id=1,
        knowledge_base_id="kb-1",
    )


async def test_get_share_missing_raises_not_found() -> None:
    service = _read_service(_FakeKBShareRepo([]))
    with pytest.raises(NotFoundError):
        await service.get_share(share_id="ghost")


async def test_get_share_by_kb_and_org_roundtrip() -> None:
    share = _share(id="s-1", knowledge_base_id="kb-1", organization_id="org-2")
    service = _read_service(_FakeKBShareRepo([share]))

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


async def test_share_stores_requested_permission() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    state.orgs["org-1"] = _org()
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="viewer")

    share = await service.share_knowledge_base(
        knowledge_base_id="kb-1",
        organization_id="org-1",
        user_id=_CREATOR,
        tenant_id=_TENANT,
        tenant_role="contributor",
        permission="editor",
    )
    assert share.permission == "editor"
    assert share.knowledge_base_id == "kb-1"
    assert share.my_role_in_org == "viewer"
    assert share.my_permission == "viewer"


async def test_share_rejects_unknown_permission() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    with pytest.raises(ValidationError):
        await service.share_knowledge_base(
            knowledge_base_id="kb-1",
            organization_id="org-1",
            user_id=_CREATOR,
            tenant_id=_TENANT,
            tenant_role="admin",
            permission="owner",
        )


async def test_share_other_tenant_kb_is_not_found() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb(tenant_id=_OTHER_TENANT)
    with pytest.raises(NotFoundError):
        await service.share_knowledge_base(
            knowledge_base_id="kb-1",
            organization_id="org-1",
            user_id=_CREATOR,
            tenant_id=_TENANT,
            tenant_role="admin",
            permission="viewer",
        )


async def test_share_contributor_not_creator_is_denied() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    state.orgs["org-1"] = _org()
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="editor")
    with pytest.raises(PermissionDeniedError):
        await service.share_knowledge_base(
            knowledge_base_id="kb-1",
            organization_id="org-1",
            user_id="usr-other",
            tenant_id=_TENANT,
            tenant_role="contributor",
            permission="viewer",
        )


async def test_share_tenant_not_in_org_is_denied() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    state.orgs["org-1"] = _org()
    with pytest.raises(PermissionDeniedError):
        await service.share_knowledge_base(
            knowledge_base_id="kb-1",
            organization_id="org-1",
            user_id=_CREATOR,
            tenant_id=_TENANT,
            tenant_role="admin",
            permission="viewer",
        )


async def test_share_does_not_require_org_editor() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    state.orgs["org-1"] = _org()
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="viewer")

    share = await service.share_knowledge_base(
        knowledge_base_id="kb-1",
        organization_id="org-1",
        user_id=_CREATOR,
        tenant_id=_TENANT,
        tenant_role="admin",
        permission="editor",
    )
    assert share.permission == "editor"


async def test_duplicate_upgrades_permission() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb()
    state.orgs["org-1"] = _org()
    state.members[("org-1", _TENANT)] = _member(org_id="org-1", tenant_id=_TENANT, role="admin")

    first = await service.share_knowledge_base(
        knowledge_base_id="kb-1",
        organization_id="org-1",
        user_id=_CREATOR,
        tenant_id=_TENANT,
        tenant_role="admin",
        permission="viewer",
    )
    second = await service.share_knowledge_base(
        knowledge_base_id="kb-1",
        organization_id="org-1",
        user_id=_CREATOR,
        tenant_id=_TENANT,
        tenant_role="admin",
        permission="editor",
    )
    assert first.id == second.id
    assert second.permission == "editor"


async def test_update_share_permission() -> None:
    service, state = _make_service()
    share = _share(id="s-1", knowledge_base_id="kb-1", permission="viewer")
    state.shares[share.id] = share
    state.kbs["kb-1"] = _kb()

    await service.update_share_permission(
        knowledge_base_id="kb-1",
        share_id=share.id,
        permission="editor",
        user_id=_CREATOR,
        tenant_id=_TENANT,
        tenant_role="contributor",
    )
    assert state.shares[share.id].permission == "editor"


async def test_update_other_tenant_is_not_found() -> None:
    service, state = _make_service()
    share = _share(id="s-1", knowledge_base_id="kb-1")
    state.shares[share.id] = share
    state.kbs["kb-1"] = _kb()

    with pytest.raises(NotFoundError):
        await service.update_share_permission(
            knowledge_base_id="kb-1",
            share_id=share.id,
            permission="editor",
            user_id="usr-dest-admin",
            tenant_id=_OTHER_TENANT,
            tenant_role="admin",
        )


async def test_dest_org_admin_cannot_mutate() -> None:
    service, state = _make_service()
    share = _share(id="s-1", knowledge_base_id="kb-1")
    state.shares[share.id] = share
    state.kbs["kb-1"] = _kb()
    state.members[("org-1", _OTHER_TENANT)] = _member(
        org_id="org-1", tenant_id=_OTHER_TENANT, role="admin"
    )

    with pytest.raises(NotFoundError):
        await service.remove_share(
            knowledge_base_id="kb-1",
            share_id=share.id,
            user_id="usr-dest-admin",
            tenant_id=_OTHER_TENANT,
            tenant_role="admin",
        )


async def test_original_sharer_can_revoke() -> None:
    service, state = _make_service()
    share = _share(id="s-1", knowledge_base_id="kb-1", shared_by_user_id="usr-sharer")
    state.shares[share.id] = share
    state.kbs["kb-1"] = _kb()

    await service.remove_share(
        knowledge_base_id="kb-1",
        share_id=share.id,
        user_id="usr-sharer",
        tenant_id=_TENANT,
        tenant_role="contributor",
    )
    assert state.shares[share.id].deleted_at is not None


async def test_list_shares_other_tenant_kb_is_not_found() -> None:
    service, state = _make_service()
    state.kbs["kb-1"] = _kb(tenant_id=_OTHER_TENANT)
    with pytest.raises(NotFoundError):
        await service.list_shares_by_knowledge_base(
            knowledge_base_id="kb-1",
            tenant_id=_TENANT,
            tenant_role="admin",
        )
