"""Unit tests for the new ``validate_active_tenant_association`` gate.

The gate reads the principal from ``request.state`` and asks the
tenant-member service for an active membership row. Failure modes
(missing user, missing tenant, missing membership, soft-deleted
membership) are covered; the success path returns a small DTO dict
the handler can consume.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request

from src.common.exception import UnauthorizedError


def _make_request(*, user_id: str | None, tenant_id: int) -> Request:
    """Build a minimal Starlette ``Request`` with the principal state set."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/test",
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)
    if user_id is not None:
        request.state.user_info = {"id": user_id, "is_active": "1"}
    request.state.tenant_id = str(tenant_id)
    return request


@pytest.fixture
def patched_member_service(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch the tenant-member service factory used by the gate."""
    service = AsyncMock()

    def _factory(session: object) -> AsyncMock:
        return service

    # The gate imports the factory lazily; patch by fully-qualified path
    # so the module import does not run at fixture time (avoids the
    # pre-existing circular-import cycle in src.web.deps).
    monkeypatch.setattr(
        "src.core.tenants.factory.build_tenant_member_service",
        _factory,
    )
    return service


@pytest.mark.asyncio
async def test_gate_returns_dto_for_active_membership(
    patched_member_service: AsyncMock,
) -> None:
    from src.web.deps.rbac import validate_active_tenant_association

    patched_member_service.get_membership = AsyncMock(
        return_value=SimpleNamespace(id=42, role="admin", deleted_at=None)
    )
    request = _make_request(user_id="u-1", tenant_id=7)

    result = await validate_active_tenant_association(
        request=request,
        session=AsyncMock(),
        user_id="u-1",
        tenant_id=7,
    )
    assert result["user_id"] == "u-1"
    assert result["tenant_id"] == 7
    assert result["role"] == "admin"
    assert result["membership_id"] == 42


@pytest.mark.asyncio
async def test_gate_rejects_missing_membership(
    patched_member_service: AsyncMock,
) -> None:
    from src.web.deps.rbac import validate_active_tenant_association

    patched_member_service.get_membership = AsyncMock(return_value=None)
    request = _make_request(user_id="u-1", tenant_id=7)

    with pytest.raises(UnauthorizedError):
        await validate_active_tenant_association(
            request=request,
            session=AsyncMock(),
            user_id="u-1",
            tenant_id=7,
        )


@pytest.mark.asyncio
async def test_gate_rejects_soft_deleted_membership(
    patched_member_service: AsyncMock,
) -> None:
    from src.web.deps.rbac import validate_active_tenant_association

    patched_member_service.get_membership = AsyncMock(
        return_value=SimpleNamespace(id=42, role="admin", deleted_at=12345)
    )
    request = _make_request(user_id="u-1", tenant_id=7)

    with pytest.raises(UnauthorizedError):
        await validate_active_tenant_association(
            request=request,
            session=AsyncMock(),
            user_id="u-1",
            tenant_id=7,
        )


@pytest.mark.asyncio
async def test_gate_rejects_unresolved_user_id() -> None:
    from src.web.deps.rbac import validate_active_tenant_association

    request = _make_request(user_id=None, tenant_id=7)
    with pytest.raises(UnauthorizedError):
        await validate_active_tenant_association(
            request=request,
            session=AsyncMock(),
            user_id=None,
            tenant_id=7,
        )


@pytest.mark.asyncio
async def test_gate_rejects_missing_tenant_context() -> None:
    from src.web.deps.rbac import validate_active_tenant_association

    request = _make_request(user_id="u-1", tenant_id=0)
    with pytest.raises(UnauthorizedError):
        await validate_active_tenant_association(
            request=request,
            session=AsyncMock(),
            user_id="u-1",
            tenant_id=0,
        )
