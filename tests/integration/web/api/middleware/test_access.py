"""Access-control middleware tests.

Covers :mod:`src.web.middleware.auth` end-to-end against the live
``require_auth`` dependency via :class:`fastapi.testclient.TestClient`.
The integration-test rig sets the ``x-knowledge-*`` trio as a header
channel; this file pins the failure modes that rig must NOT silently
allow:

- missing header trio -> 401.
- partial header trio (tenant without user, or vice versa) -> 401.
- user-id refers to a row that does not exist -> 401.

A "user is not a member of the header's tenant" case is documented
below as a current-code blocker: the header channel resolves the
principal purely from the ``users`` table and the route handlers do
not yet wire :func:`validate_active_tenant_association`, so a caller
with a known user_id but a foreign tenant_id passes the auth gate.
The cross-tenant guard ships as a separate gate; this file only
covers the channels the integration-test rig exercises today.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.core.tenants.member_service import ROLE_OWNER


def test_missing_knowledge_headers_returns_401(app) -> None:
    """No x-knowledge-* headers and no Authorization -> 401.

    With no header trio and no Bearer token the dependency falls
    through to the ``UnauthorizedError`` raised at the bottom of
    :func:`require_auth`.
    """
    with TestClient(app=app) as client:
        response = client.get("/tenants")
    assert response.status_code == 401, response.text


def test_partial_tenant_header_only_returns_401(app) -> None:
    """x-knowledge-tenant-id without x-knowledge-user-id -> 401.

    ``_resolve_header_auth`` returns False when either header is
    missing; the request then has no Authorization header and no
    API key, so it falls through to the missing-authentication 401.
    """
    with TestClient(app=app) as client:
        response = client.get(
            "/tenants",
            headers={"x-knowledge-tenant-id": "1"},
        )
    assert response.status_code == 401, response.text


def test_unknown_user_header_returns_401(app) -> None:
    """x-knowledge-user-id referring to a non-existent row -> 401.

    The header channel raises :class:`UnauthorizedError` with code
    ``auth.user_not_found`` when ``UserRepository.find_by_id`` misses;
    the exception handler maps that to HTTP 401.
    """
    with TestClient(app=app) as client:
        response = client.get(
            "/tenants",
            headers={
                "x-knowledge-user-id": "usr-does-not-exist",
                "x-knowledge-tenant-id": "1",
            },
        )
    assert response.status_code == 401, response.text


def test_cross_tenant_user_header_is_a_known_blocker(
    authed_client,
    admin_user: tuple[str, int],
    other_org_admin_user: tuple[str, int],
) -> None:
    """Documented blocker: foreign-tenant header trio is NOT rejected.

    The header channel currently resolves the principal from the
    ``users`` table only; it does not consult ``tenant_members``.
    Sending ``other_org_admin_user``'s ``user_id`` paired with
    ``admin_user``'s ``tenant_id`` therefore passes the auth gate
    (HTTP 200); the response lists the header principal's real
    memberships rather than the requested tenant.

    The intended fix wires :func:`validate_active_tenant_association`
    (or equivalent DB-backed membership check) into the route
    handlers; until that lands the cross-tenant rejection cannot be
    exercised at the HTTP layer. This test pins the current behavior
    so the blocker is visible in CI; flip the assertions once the
    gate is wired.
    """
    other_user_id, other_tenant_id = other_org_admin_user
    _admin_user_id, admin_tenant_id = admin_user
    assert other_tenant_id != admin_tenant_id

    headers = dict(authed_client.headers)
    headers["x-knowledge-user-id"] = other_user_id
    headers["x-knowledge-tenant-id"] = str(admin_tenant_id)

    response = authed_client.get("/tenants", headers=headers)
    # Current behavior: passes the auth gate; the response lists the
    # header principal's ACTUAL memberships (the other org's tenant),
    # not the header's tenant.
    assert response.status_code == 200, response.text
    payload = response.json()
    tenant_ids = [t["id"] for t in payload["data"]["items"]]
    assert other_tenant_id in tenant_ids
    assert admin_tenant_id not in tenant_ids


__all__ = [
    "ROLE_OWNER",
    "test_cross_tenant_user_header_is_a_known_blocker",
    "test_missing_knowledge_headers_returns_401",
    "test_partial_tenant_header_only_returns_401",
    "test_unknown_user_header_returns_401",
]
