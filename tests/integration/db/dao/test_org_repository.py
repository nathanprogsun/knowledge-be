"""Integration tests for the organization repositories against the real applied schema.

Tests insert unique rows per run; isolation relies on unique org ids and
tenant ids. Tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.organization_repository import (
    OrganizationJoinRequestRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.models.organization import (
    JOIN_REQUEST_STATUS_APPROVED,
    JOIN_REQUEST_STATUS_PENDING,
    JOIN_REQUEST_TYPE_JOIN,
    JOIN_REQUEST_TYPE_UPGRADE,
    ORG_ROLE_ADMIN,
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    Organization,
    OrganizationJoinRequest,
    OrganizationTenantMember,
)
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _uid() -> str:
    return f"org-{uuid.uuid4().hex[:12]}"


def _org(*, tenant_id: int | None = None, name: str | None = None) -> Organization:
    return Organization(
        id=_uid(),
        name=name or f"org-{uuid.uuid4().hex[:8]}",
        description="per-test organization",
        owner_id=f"usr-{uuid.uuid4().hex[:12]}",
        owner_tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        invite_code=None,
        invite_code_expires_at=None,
        invite_code_validity_days=7,
        avatar="",
        require_approval=False,
        searchable=False,
        member_limit=50,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _member(
    *,
    organization_id: str,
    tenant_id: int | None = None,
    role: str = ORG_ROLE_VIEWER,
) -> OrganizationTenantMember:
    return OrganizationTenantMember(
        id=_uid(),
        organization_id=organization_id,
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        role=role,
        representative_user_id=f"usr-{uuid.uuid4().hex[:12]}",
        joined_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _join_request(
    *,
    organization_id: str,
    tenant_id: int | None = None,
    request_type: str = JOIN_REQUEST_TYPE_JOIN,
    status: str = JOIN_REQUEST_STATUS_PENDING,
) -> OrganizationJoinRequest:
    return OrganizationJoinRequest(
        id=_uid(),
        organization_id=organization_id,
        user_id=f"usr-{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        status=status,
        requested_role=ORG_ROLE_VIEWER,
        request_type=request_type,
        prev_role=None,
        message=None,
        reviewed_by=None,
        reviewed_at=None,
        review_message=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── OrganizationRepository ────────────────────────────────────────────


async def test_org_create_and_get_by_id(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)
    org = _org()

    stored = await repo.create(org)
    await session.commit()

    assert stored.id == org.id
    fetched = await repo.get_by_id_or_none(org.id)
    assert fetched is not None
    assert fetched.name == org.name
    assert fetched.owner_tenant_id == org.owner_tenant_id


async def test_org_get_by_invite_code(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)
    org = _org()
    await repo.create(org)
    invite_code = f"inv-{uuid.uuid4().hex[:12]}"
    await repo.update_invite_code(
        id=org.id,
        invite_code=invite_code,
        expires_at=_FUTURE,
        now=_NOW,
    )
    await session.commit()

    fetched = await repo.get_by_invite_code_or_none(invite_code)
    assert fetched is not None
    assert fetched.id == org.id
    assert fetched.invite_code == invite_code


async def test_org_get_by_invite_code_missing(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)

    fetched = await repo.get_by_invite_code_or_none("no-such-code")

    assert fetched is None


async def test_org_list_by_tenant(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    tenant_id = make_test_tenant_id()
    org = _org(tenant_id=tenant_id)
    await org_repo.create(org)
    await member_repo.add_member(_member(organization_id=org.id, tenant_id=tenant_id))
    await session.commit()

    orgs = await org_repo.list_by_tenant(tenant_id)

    assert [o.id for o in orgs] == [org.id]


async def test_org_list_searchable(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)
    name = f"Acme {uuid.uuid4().hex[:8]}"
    org = _org(name=name)
    org = org.model_copy(update={"searchable": True})
    await repo.create(org)
    await session.commit()

    hits = await repo.list_searchable(query="acme", limit=10)

    assert org.id in {o.id for o in hits}


async def test_org_soft_delete_hides_row(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)
    org = _org()
    await repo.create(org)
    await session.commit()

    deleted = await repo.soft_delete(id=org.id, now=_NOW)
    await session.commit()

    assert deleted is True
    assert await repo.get_by_id_or_none(org.id) is None


async def test_org_update_invite_code_clears_expiry(session: AsyncSession) -> None:
    repo = OrganizationRepository(session)
    org = _org()
    await repo.create(org)
    await session.commit()

    updated = await repo.update_invite_code(id=org.id, invite_code=None, expires_at=None, now=_NOW)
    await session.commit()

    assert updated is True
    fetched = await repo.get_by_id_or_none(org.id)
    assert fetched is not None
    assert fetched.invite_code is None
    assert fetched.invite_code_expires_at is None


# ── OrganizationMemberRepository ──────────────────────────────────────


async def test_member_add_and_get(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    member = _member(organization_id=org.id, tenant_id=tenant_id, role=ORG_ROLE_EDITOR)

    stored = await member_repo.add_member(member)
    await session.commit()

    assert stored is not None
    fetched = await member_repo.get_member(organization_id=org.id, tenant_id=tenant_id)
    assert fetched is not None
    assert fetched.role == ORG_ROLE_EDITOR


async def test_member_duplicate_is_suppressed(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    member = _member(organization_id=org.id, tenant_id=tenant_id)

    first = await member_repo.add_member(member)
    await session.commit()
    duplicate = await member_repo.add_member(member)
    await session.commit()

    assert first is not None
    assert duplicate is None


async def test_member_list_and_count(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    org = _org()
    await org_repo.create(org)
    await member_repo.add_member(_member(organization_id=org.id))
    await member_repo.add_member(_member(organization_id=org.id))
    await session.commit()

    members = await member_repo.list_members(org.id)
    count = await member_repo.count_members(org.id)

    assert len(members) == 2
    assert count == 2


async def test_member_update_role(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    await member_repo.add_member(_member(organization_id=org.id, tenant_id=tenant_id))
    await session.commit()

    updated = await member_repo.update_member_role(
        organization_id=org.id,
        tenant_id=tenant_id,
        role=ORG_ROLE_ADMIN,
    )
    await session.commit()

    assert updated is True
    fetched = await member_repo.get_member(organization_id=org.id, tenant_id=tenant_id)
    assert fetched is not None
    assert fetched.role == ORG_ROLE_ADMIN


async def test_member_remove(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    member_repo = OrganizationMemberRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    await member_repo.add_member(_member(organization_id=org.id, tenant_id=tenant_id))
    await session.commit()

    removed = await member_repo.remove_member(organization_id=org.id, tenant_id=tenant_id)
    await session.commit()

    assert removed is True
    assert await member_repo.get_member(organization_id=org.id, tenant_id=tenant_id) is None


# ── OrganizationJoinRequestRepository ────────────────────────────────


async def test_join_request_create_and_get(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    request_repo = OrganizationJoinRequestRepository(session)
    org = _org()
    await org_repo.create(org)
    request = _join_request(organization_id=org.id)

    stored = await request_repo.create_join_request(request)
    await session.commit()

    assert stored.id == request.id
    fetched = await request_repo.get_join_request_by_id(request.id)
    assert fetched is not None
    assert fetched.status == JOIN_REQUEST_STATUS_PENDING


async def test_join_request_pending_lookup(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    request_repo = OrganizationJoinRequestRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    await request_repo.create_join_request(
        _join_request(organization_id=org.id, tenant_id=tenant_id)
    )
    await session.commit()

    pending = await request_repo.get_pending_join_request(
        organization_id=org.id,
        tenant_id=tenant_id,
    )
    by_type = await request_repo.get_pending_request_by_type(
        organization_id=org.id,
        tenant_id=tenant_id,
        request_type=JOIN_REQUEST_TYPE_JOIN,
    )

    assert pending is not None
    assert by_type is not None
    assert pending.id == by_type.id


async def test_join_request_pending_lookup_ignores_approved(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    request_repo = OrganizationJoinRequestRepository(session)
    org = _org()
    await org_repo.create(org)
    tenant_id = make_test_tenant_id()
    request = _join_request(organization_id=org.id, tenant_id=tenant_id)
    await request_repo.create_join_request(request)
    await request_repo.update_join_request_status(
        id=request.id,
        status=JOIN_REQUEST_STATUS_APPROVED,
        reviewed_by=f"usr-{uuid.uuid4().hex[:12]}",
        review_message="welcome",
        reviewed_at=_NOW,
    )
    await session.commit()

    pending = await request_repo.get_pending_join_request(
        organization_id=org.id,
        tenant_id=tenant_id,
    )

    assert pending is None


async def test_join_request_list_and_count_by_status(session: AsyncSession) -> None:
    org_repo = OrganizationRepository(session)
    request_repo = OrganizationJoinRequestRepository(session)
    org = _org()
    await org_repo.create(org)
    await request_repo.create_join_request(_join_request(organization_id=org.id))
    await request_repo.create_join_request(
        _join_request(
            organization_id=org.id,
            request_type=JOIN_REQUEST_TYPE_UPGRADE,
        )
    )
    await session.commit()

    all_requests = await request_repo.list_join_requests(org.id)
    pending = await request_repo.list_join_requests(org.id, status=JOIN_REQUEST_STATUS_PENDING)
    count = await request_repo.count_join_requests(org.id)

    assert len(all_requests) == 2
    assert len(pending) == 2
    assert count == 2
