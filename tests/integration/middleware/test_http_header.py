"""JWT / knowledge-prefixed header interaction tests.

Exercises :mod:`src.web.middleware.auth` end-to-end against the live
``require_auth`` dependency. Each test routes through a real
``TestClient`` against the integration app so the resolution order
observed here is the order the production app sees.

The auth dependency tries channels in this order:

1. The ``x-knowledge-*`` header trio (set by the integration-test rig).
2. ``Authorization: Bearer <jwt>``.
3. ``X-API-Key``.

When multiple channels are present the FIRST that resolves
successfully wins; only when none does, the request is rejected.

Covers:
- header-only request -> 200 (header channel resolves).
- JWT + header (both present) -> 200 (header channel wins because it
  runs first; the resolved principal matches the header trio, not the
  JWT's claims).
- garbage ``Authorization`` Bearer token -> 401 (token decode fails).
- JWT signed with the test secret carrying an unknown ``iss`` claim ->
  401. The project does not enforce ``iss`` at the decode layer, so
  this 401 comes from the token-not-found branch in
  ``AuthService._validate_token`` (``tokens_repo.find_by_token_value``
  raises ``NotFoundError`` -> ``UnauthorizedError``), not from a
  dedicated ``iss`` check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jose import jwt

from src.core.tenants.member_service import ROLE_OWNER
from src.settings import get_settings
from src.util.security import reset_secret_cache


def _project_secret() -> str:
    """Return the configured JWT secret, refreshing the module-level cache.

    The cache is reset so the latest ``Settings`` instance is consulted
    (the helper that mints test tokens must use the same secret the
    auth dependency verifies against).
    """
    reset_secret_cache()
    return get_settings().jwt_secret_key


def _access_token(
    *,
    user_id: str,
    email: str,
    tenant_id: int | None,
    secret: str,
    iss: str | None = None,
) -> str:
    """Mint an HS256 access JWT directly with the project secret.

    Mirrors ``create_access_token`` but lets the caller attach an
    arbitrary ``iss`` claim. The signed token is suitable for the
    ``Authorization: Bearer <token>`` header.
    """
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "user_id": user_id,
        "email": email,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
        "jti": "test-jti",
        "type": "access",
    }
    if iss is not None:
        claims["iss"] = iss
    return jwt.encode(claims, secret, algorithm="HS256")


def test_header_only_resolves_principal(
    authed_client,
    admin_user: tuple[str, int],
) -> None:
    """Header trio alone is sufficient to authenticate.

    ``require_auth`` short-circuits on the header channel before the
    JWT path runs. The protected ``GET /tenants`` returns the
    caller's workspace.
    """
    _user_id, tenant_id = admin_user
    response = authed_client.get("/tenants")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    tenant_ids = [t["id"] for t in payload["data"]["items"]]
    assert tenant_id in tenant_ids


def test_jwt_with_header_resolves_via_header_channel(
    authed_client,
    admin_user: tuple[str, int],
    other_org_admin_user: tuple[str, int],
) -> None:
    """When both JWT and headers are present, the header trio wins.

    The request carries an Authorization Bearer for the OTHER org's
    user (different ``user_id`` and ``tenant_id``) AND the local
    ``x-knowledge-*`` trio. Because ``require_auth`` checks headers
    before JWT, the principal resolved is the header principal; the
    response lists the local workspace (``admin_user``'s tenant), not
    the JWT's tenant.
    """
    header_user_id, header_tenant_id = admin_user
    jwt_user_id, jwt_tenant_id = other_org_admin_user
    assert jwt_user_id != header_user_id
    assert jwt_tenant_id != header_tenant_id

    secret = _project_secret()
    token = _access_token(
        user_id=jwt_user_id,
        email="other@example.test",
        tenant_id=jwt_tenant_id,
        secret=secret,
    )

    # Snapshot the headers set by ``authed_client`` so we can layer
    # the Authorization header on top without dropping the trio.
    headers = dict(authed_client.headers)
    headers["Authorization"] = f"Bearer {token}"

    response = authed_client.get("/tenants", headers=headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    tenant_ids = [t["id"] for t in payload["data"]["items"]]
    # The header channel wins: the response lists the header
    # principal's tenant, not the JWT principal's.
    assert header_tenant_id in tenant_ids
    assert jwt_tenant_id not in tenant_ids


def test_garbage_authorization_returns_401(app) -> None:
    """A garbage Bearer token (no x-knowledge-* headers) -> 401.

    With no header trio the dependency falls through to the JWT
    channel; the token's signature does not match the project
    secret so ``decode_token`` raises ``TokenError`` which is mapped
    to ``UnauthorizedError``.
    """
    from fastapi.testclient import TestClient

    with TestClient(app=app) as client:
        response = client.get(
            "/tenants",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
    assert response.status_code == 401, response.text


def test_jwt_with_bad_iss_returns_401(
    app,
    admin_user: tuple[str, int],
) -> None:
    """JWT signed with the test secret but carrying an unknown ``iss``.

    The project decoder (``jose.jwt.decode`` with no ``issuer=``
    option) accepts arbitrary ``iss`` claims, so this 401 is NOT
    produced by an explicit iss check. It comes from
    ``AuthService._validate_token`` failing to locate the token in
    the ``auth_tokens`` table (``NotFoundError`` -> 401). The test
    documents the current behavior: a hand-rolled token with bad
    iss cannot authenticate because it is not persisted.
    """
    from fastapi.testclient import TestClient

    user_id, tenant_id = admin_user
    secret = _project_secret()
    token = _access_token(
        user_id=user_id,
        email="user@example.test",
        tenant_id=tenant_id,
        secret=secret,
        iss="https://attacker.example/evil",
    )

    with TestClient(app=app) as client:
        response = client.get(
            "/tenants",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401, response.text


__all__ = [
    "ROLE_OWNER",
    "test_garbage_authorization_returns_401",
    "test_header_only_resolves_principal",
    "test_jwt_with_bad_iss_returns_401",
    "test_jwt_with_header_resolves_via_header_channel",
]
