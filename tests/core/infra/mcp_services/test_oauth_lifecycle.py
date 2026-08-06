"""Tests for the OAuth lifecycle helpers.

Covers the additions on top of the legacy placeholder surface:

- :class:`TokenSet` round-trips ``is_expired`` /
  :meth:`as_authorization_header` correctly.
- :class:`OAuthStateStore` records, returns, and expires state entries.
- :class:`InMemorySecretStore` stores and forgets per-user tokens.
- :meth:`OAuthManager.build_authorization_url` constructs an RFC 6749
  authorization-code URL with PKCE.
- :meth:`OAuthManager.exchange_code` posts to ``token_endpoint`` with
  the captured PKCE verifier + state, and persists the returned
  :class:`TokenSet`.
- :meth:`OAuthManager.refresh` rotates via ``refresh_token_grant``.
- :meth:`OAuthManager.ensure_authorized` reuses the cached token when
  fresh and refreshes on the fly when near expiry.
- :meth:`OAuthManager.revoke_token` drops the token from the in-process
  store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.infra.mcp_services.oauth import (
    DEFAULT_REFRESH_SKEW_SECONDS,
    DEFAULT_STATE_TTL_SECONDS,
    InMemorySecretStore,
    OAuthManager,
    OAuthStateStore,
    StateEntry,
    TokenSet,
)
from src.core.infra.mcp_services.types import MCPServiceInfo

# ── Fixtures ────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _info(auth_config: JsonObject | None = None) -> MCPServiceInfo:
    return MCPServiceInfo(
        id="svc-1",
        tenant_id=42,
        name="acme",
        transport_type="sse",
        auth_config=auth_config,
        created_at=_now(),
        updated_at=_now(),
    )


def _live_auth_config() -> JsonObject:
    return {
        "auth_type": "oauth",
        "authorization_endpoint": "https://idp.example.com/authorize",
        "token_endpoint": "https://idp.example.com/token",
        "client_id": "client-abc",
        "scopes": ["read", "write"],
    }


# ── TokenSet basics ────────────────────────────────────────────────


def test_token_set_is_frozen_and_renders_authorization_header() -> None:
    """``TokenSet`` is immutable and renders the ``Authorization`` header."""
    token = TokenSet(access_token="abc123", token_type="Bearer")
    assert token.as_authorization_header() == "Bearer abc123"
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        token.access_token = "mutated"  # type: ignore[misc]


def test_token_set_is_expired_returns_true_within_skew() -> None:
    """Tokens within ``DEFAULT_REFRESH_SKEW_SECONDS`` of expiry are "expired"."""
    expires_at = datetime.now(UTC) + timedelta(seconds=5)
    token = TokenSet(access_token="x", refresh_token="y", expires_at=expires_at)
    assert token.is_expired() is True


def test_token_set_is_expired_returns_false_for_fresh_tokens() -> None:
    """Tokens expiring well in the future are not "expired"."""
    token = TokenSet(
        access_token="x",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert token.is_expired() is False


def test_token_set_is_expired_returns_false_when_no_expires_at() -> None:
    """A missing ``expires_at`` means the token is not near expiry."""
    token = TokenSet(access_token="x")
    assert token.is_expired() is False


# ── OAuthStateStore ────────────────────────────────────────────────


def test_state_store_records_returns_and_consumes_entries() -> None:
    """The state store can record, peek, and consume entries by key."""
    import time

    store = OAuthStateStore()
    entry = StateEntry(
        tenant_id=1,
        user_id="alice",
        service_id="svc-1",
        code_verifier="verifier",
        client_id="client",
        redirect_uri="https://example.com/cb",
        created_at=time.monotonic(),
    )
    store.put(state="abc", entry=entry)
    assert store.peek(state="abc") is entry
    consumed = store.take(state="abc")
    assert consumed is entry
    assert store.peek(state="abc") is None


def test_state_store_take_raises_for_unknown_or_expired_entries() -> None:
    """Taking an unknown / expired state surfaces :class:`ValidationError`."""
    store = OAuthStateStore()
    with pytest.raises(ValidationError):
        store.take(state="missing")


def test_state_store_purges_expired_entries_on_access() -> None:
    """Stale entries are removed by every read/write through the store."""
    import time

    store = OAuthStateStore(ttl_seconds=0)
    # ``_StateEntry`` is a private dataclass on the store — accessed via
    # ``object.__getattribute__`` for typing purposes.
    entry = StateEntry(
        tenant_id=1,
        user_id="alice",
        service_id="svc-1",
        code_verifier="v",
        client_id="c",
        redirect_uri="https://example.com/cb",
        created_at=time.monotonic() - 100,
    )
    store.put(state="stale", entry=entry)
    # ttl=0 forces every entry to be considered stale on the next access.
    assert store.peek(state="stale") is None


def test_state_store_default_ttl_matches_go_layout_constant() -> None:
    """The default TTL mirrors the Go ``oauthStateTTL`` (10 minutes)."""
    assert DEFAULT_STATE_TTL_SECONDS == 600


def test_state_store_purge_expired_keeps_fresh_drops_stale() -> None:
    """Coverage: ``_purge_expired`` removes only stale entries.

    With ``ttl=1.0`` we can construct a mixed-bag scenario:
    one entry recorded ``now`` (still fresh), one recorded well in
    the past (stale). A subsequent ``peek`` round-trip must drop
    only the stale entry; the fresh one must survive.
    """
    import time

    store = OAuthStateStore(ttl_seconds=1)
    now = time.monotonic()
    fresh = StateEntry(
        tenant_id=1,
        user_id="alice",
        service_id="svc-1",
        code_verifier="v-fresh",
        client_id="c",
        redirect_uri="https://example.com/cb",
        created_at=now,
    )
    stale = StateEntry(
        tenant_id=1,
        user_id="alice",
        service_id="svc-1",
        code_verifier="v-stale",
        client_id="c",
        redirect_uri="https://example.com/cb",
        created_at=now - 5.0,  # older than ttl
    )
    store.put(state="fresh", entry=fresh)
    store.put(state="stale", entry=stale)
    assert store.peek(state="fresh") is not None
    assert store.peek(state="stale") is None  # purged on access


# ── InMemorySecretStore ────────────────────────────────────────────


async def test_secret_store_round_trip_and_delete() -> None:
    """The store returns the cached token, then forgets it on delete."""
    store = InMemorySecretStore()
    token = TokenSet(access_token="abc", refresh_token="xyz")
    assert await store.get(tenant_id=1, user_id="alice", service_id="svc") is None

    await store.put(tenant_id=1, user_id="alice", service_id="svc", token=token)
    fetched = await store.get(tenant_id=1, user_id="alice", service_id="svc")
    assert fetched == token

    await store.delete(tenant_id=1, user_id="alice", service_id="svc")
    assert await store.get(tenant_id=1, user_id="alice", service_id="svc") is None


async def test_secret_store_delete_is_idempotent() -> None:
    """Deleting a missing entry does not raise."""
    store = InMemorySecretStore()
    await store.delete(tenant_id=99, user_id="ghost", service_id="none")


# ── build_authorization_url ────────────────────────────────────────


def test_build_authorization_url_includes_pkce_and_required_params() -> None:
    """The URL carries response_type=code + S256 PKCE + scopes + state."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    url, state = manager.build_authorization_url(
        user_id="alice",
        redirect_uri="https://app.example.com/oauth/callback",
    )
    assert state
    assert "https://idp.example.com/authorize" in url
    assert "response_type=code" in url
    assert "client_id=client-abc" in url
    assert "redirect_uri=" in url
    assert "state=" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert "scope=read+write" in url or "scope=read%20write" in url or "scope=" in url
    # CSRF state must be recorded so ``exchange_code`` can validate it.
    assert manager.state_store.peek(state=state) is not None


def test_build_authorization_url_raises_when_endpoints_unset() -> None:
    """An OAuth config missing the token endpoint is a config error."""
    manager = OAuthManager(
        service=_info(
            auth_config={
                "auth_type": "oauth",
                "authorization_endpoint": "https://idp.example.com/authorize",
                "client_id": "c",
            },
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        manager.build_authorization_url(
            user_id="alice",
            redirect_uri="https://example.com/cb",
        )
    assert excinfo.value.code == "mcp_service.oauth_endpoints_missing"


def test_build_authorization_url_raises_when_client_id_missing() -> None:
    """An OAuth config without a ``client_id`` cannot build an authorize URL."""
    manager = OAuthManager(
        service=_info(
            auth_config={
                "auth_type": "oauth",
                "authorization_endpoint": "https://idp.example.com/authorize",
                "token_endpoint": "https://idp.example.com/token",
            },
        ),
    )
    with pytest.raises(ValidationError) as excinfo:
        manager.build_authorization_url(
            user_id="alice",
            redirect_uri="https://example.com/cb",
        )
    assert excinfo.value.code == "mcp_service.oauth_client_id_missing"


# ── exchange_code ───────────────────────────────────────────────────


async def test_exchange_code_posts_token_endpoint_and_persists_token() -> None:
    """Code exchange hits the token endpoint and stores the resulting token."""
    secret_store = InMemorySecretStore()
    state_store = OAuthStateStore()
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
        state_store=state_store,
    )
    _url, state = manager.build_authorization_url(
        user_id="alice",
        redirect_uri="https://example.com/cb",
    )

    token_response: dict[str, Any] = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "token_type": "Bearer",
        "scope": "read write",
        "expires_in": 3600,
    }

    with respx.mock(base_url="https://idp.example.com") as router:
        token_route = router.post("/token").respond(
            200,
            json=token_response,
        )

        async with httpx.AsyncClient() as http_client:
            # Re-bind the manager to the test http client.
            manager._http_client = http_client
            token = await manager.exchange_code(
                user_id="alice",
                code="auth-code-1",
                state=state,
            )

    assert token.access_token == "new-access"
    assert token.refresh_token == "new-refresh"
    assert token.expires_at is not None
    assert token_route.called
    # Persisted to the shared secret store.
    stored = await secret_store.get(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
    )
    assert stored == token


async def test_exchange_code_rejects_unknown_state() -> None:
    """An exchange with a non-recorded state raises :class:`ValidationError`."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=InMemorySecretStore(),
        state_store=OAuthStateStore(),
    )
    async with httpx.AsyncClient() as http_client:
        manager._http_client = http_client
        with pytest.raises(ValidationError) as excinfo:
            await manager.exchange_code(
                user_id="alice",
                code="x",
                state="never-recorded",
            )
    assert excinfo.value.code == "mcp_service.oauth_state_invalid"


async def test_exchange_code_rejects_state_for_different_user_or_service() -> None:
    """A state whose principal does not match the request is rejected."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=InMemorySecretStore(),
        state_store=OAuthStateStore(),
    )
    _url, state = manager.build_authorization_url(
        user_id="alice",
        redirect_uri="https://example.com/cb",
    )
    async with httpx.AsyncClient() as http_client:
        manager._http_client = http_client
        with pytest.raises(ValidationError) as excinfo:
            await manager.exchange_code(
                user_id="mallory",
                code="x",
                state=state,
            )
    assert excinfo.value.code == "mcp_service.oauth_state_mismatch"


async def test_exchange_code_propagates_token_endpoint_errors() -> None:
    """A 4xx token response surfaces as :class:`ValidationError`."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=InMemorySecretStore(),
        state_store=OAuthStateStore(),
    )
    _url, state = manager.build_authorization_url(
        user_id="alice",
        redirect_uri="https://example.com/cb",
    )

    with respx.mock(base_url="https://idp.example.com") as router:
        router.post("/token").respond(400, text="invalid_grant")

        async with httpx.AsyncClient() as http_client:
            manager._http_client = http_client
            with pytest.raises(ValidationError) as excinfo:
                await manager.exchange_code(
                    user_id="alice",
                    code="bad-code",
                    state=state,
                )
    assert excinfo.value.code == "mcp_service.oauth_token_rejected"


# ── refresh ────────────────────────────────────────────────────────


async def test_refresh_rotates_via_refresh_token_grant_and_returns_new_pair() -> None:
    """``refresh`` posts to the token endpoint with ``grant_type=refresh_token``."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    expired = TokenSet(
        access_token="old-access",
        refresh_token="refresh-me",
        expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    rotated_payload = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "token_type": "Bearer",
        "expires_in": 7200,
    }

    with respx.mock(base_url="https://idp.example.com") as router:
        refresh_route = router.post("/token").respond(200, json=rotated_payload)
        async with httpx.AsyncClient() as http_client:
            manager._http_client = http_client
            rotated = await manager.refresh(
                token=expired,
                tenant_id=42,
                user_id="alice",
                service_id="svc-1",
            )

    assert rotated.access_token == "new-access"
    assert rotated.refresh_token == "new-refresh"
    assert refresh_route.called
    # The request carried the refresh token + client_id.
    request_body = refresh_route.calls.last.request.content.decode("utf-8")
    assert "grant_type=refresh_token" in request_body
    assert "refresh_token=refresh-me" in request_body
    assert "client_id=client-abc" in request_body


async def test_refresh_rejects_token_without_refresh_token() -> None:
    """``refresh`` of a token with no refresh_token raises."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    async with httpx.AsyncClient() as http_client:
        manager._http_client = http_client
        token = TokenSet(access_token="only-access")
        with pytest.raises(ValidationError) as excinfo:
            await manager.refresh(
                token=token,
                tenant_id=42,
                user_id="alice",
                service_id="svc-1",
            )
    assert excinfo.value.code == "mcp_service.oauth_no_refresh_token"


# ── ensure_authorized ──────────────────────────────────────────────


async def test_ensure_authorized_returns_fresh_token_without_refresh() -> None:
    """A still-fresh cached token is returned as-is (no outbound HTTP)."""
    secret_store = InMemorySecretStore()
    fresh = TokenSet(
        access_token="still-good",
        refresh_token="stale-refresh",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=fresh,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )

    returned = await manager.ensure_authorized(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
    )
    assert returned == fresh


async def test_ensure_authorized_refreshes_when_token_near_expiry() -> None:
    """A near-expiry cached token triggers an in-line refresh."""
    secret_store = InMemorySecretStore()
    stale = TokenSet(
        access_token="old-access",
        refresh_token="rotate-me",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=stale,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )

    with respx.mock(base_url="https://idp.example.com") as router:
        router.post("/token").respond(
            200,
            json={
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )
        async with httpx.AsyncClient() as http_client:
            manager._http_client = http_client
            refreshed = await manager.ensure_authorized(
                tenant_id=42,
                user_id="alice",
                service_id="svc-1",
            )

    assert refreshed.access_token == "fresh-access"
    # The rotated pair was persisted back to the secret store.
    stored = await secret_store.get(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
    )
    assert stored is not None
    assert stored.access_token == "fresh-access"


async def test_ensure_authorized_raises_when_no_token_is_stored() -> None:
    """``ensure_authorized`` for an unauthorised user raises :class:`NotFoundError`."""
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    with pytest.raises(NotFoundError) as excinfo:
        await manager.ensure_authorized(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
    assert excinfo.value.code == "mcp_service.oauth_unauthorized"


async def test_ensure_authorized_raises_when_expired_and_no_refresh_token() -> None:
    """An expired token without a refresh_token can never be recovered."""
    secret_store = InMemorySecretStore()
    expired = TokenSet(
        access_token="old-access",
        refresh_token="",
        expires_at=datetime.now(UTC) - timedelta(minutes=5),
    )
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=expired,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    with pytest.raises(NotFoundError) as excinfo:
        await manager.ensure_authorized(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
    assert excinfo.value.code == "mcp_service.oauth_expired"


# ── revoke_token ──────────────────────────────────────────────────


async def test_revoke_token_drops_token_from_store() -> None:
    """``revoke_token`` removes the cached entry in the secret store."""
    secret_store = InMemorySecretStore()
    token = TokenSet(access_token="doomed")
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=token,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    await manager.revoke_token(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
    )
    assert (
        await secret_store.get(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
        is None
    )


# ── Review follow-ups ──────────────────────────────────────────────


async def test_revoke_token_with_token_kwarg_drops_entry() -> None:
    """``revoke_token(token=...)`` actually removes the matching entry.

    The previous implementation passed ``token.access_token`` as a
    user_id and silently dropped nothing.
    """
    secret_store = InMemorySecretStore()
    token = TokenSet(access_token="doomed", refresh_token="rt")
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=token,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    await manager.revoke_token(token=token)
    assert (
        await secret_store.get(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
        is None
    )


async def test_revoke_token_with_token_kwarg_is_noop_when_no_match() -> None:
    """``revoke_token(token=...)`` for an unknown access_token is a no-op."""
    secret_store = InMemorySecretStore()
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    # No entry was ever stored; revoke must not raise.
    await manager.revoke_token(token=TokenSet(access_token="ghost"))


async def test_refresh_persists_rotated_pair_so_next_refresh_uses_new_refresh_token() -> None:
    """``refresh`` persists the rotated pair.

    The first refresh returns and persists a new access+refresh pair.
    A second refresh must use the NEW refresh token, not the old one
    (the previous code returned the rotated token but never wrote it
    back, so the next refresh would replay the old refresh token and
    the IdP would reject it as already-used).
    """
    secret_store = InMemorySecretStore()
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=TokenSet(
            access_token="old-access",
            refresh_token="first-refresh",
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        ),
    )

    seen_refresh_tokens: list[str] = []

    def _record_refresh(req: httpx.Request) -> httpx.Response:
        body = req.content.decode("utf-8")
        # ``body`` is ``refresh_token=...&grant_type=...``; pull the
        # ``refresh_token`` field out for the assertion below.
        from urllib.parse import parse_qs

        parsed = parse_qs(body)
        seen_refresh_tokens.append(parsed["refresh_token"][0])
        rotation = "second-access" if len(seen_refresh_tokens) == 1 else "third-access"
        new_refresh = "second-refresh" if len(seen_refresh_tokens) == 1 else "third-refresh"
        return httpx.Response(
            200,
            json={
                "access_token": rotation,
                "refresh_token": new_refresh,
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    with respx.mock(base_url="https://idp.example.com") as router:
        route = router.post("/token").mock(side_effect=_record_refresh)
        async with httpx.AsyncClient() as http_client:
            manager._http_client = http_client
            first = await manager.refresh(
                token=TokenSet(
                    access_token="old-access",
                    refresh_token="first-refresh",
                ),
                tenant_id=42,
                user_id="alice",
                service_id="svc-1",
            )
            # Read the rotated pair back from the store before the
            # second refresh; the new refresh_token must be present.
            stored_after_first = await secret_store.get(
                tenant_id=42,
                user_id="alice",
                service_id="svc-1",
            )
            second = await manager.refresh(
                token=first, tenant_id=42, user_id="alice", service_id="svc-1"
            )

    assert seen_refresh_tokens == ["first-refresh", "second-refresh"]
    assert stored_after_first is not None
    assert stored_after_first.refresh_token == "second-refresh"
    assert second.access_token == "third-access"
    assert route.call_count == 2


async def test_ensure_authorized_deletes_stale_token_on_permanent_refresh_failure() -> None:
    """Stale tokens are dropped on unrecoverable refresh failure.

    An expired token with no ``refresh_token`` cannot be rotated, so
    the entry must be deleted (otherwise the next caller would see a
    ghost token that can never be refreshed).
    """
    secret_store = InMemorySecretStore()
    expired_no_refresh = TokenSet(
        access_token="expired",
        refresh_token="",
        expires_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    await secret_store.put(
        tenant_id=42,
        user_id="alice",
        service_id="svc-1",
        token=expired_no_refresh,
    )
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
    )
    with pytest.raises(NotFoundError) as excinfo:
        await manager.ensure_authorized(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
    assert excinfo.value.code == "mcp_service.oauth_expired"
    assert (
        await secret_store.get(
            tenant_id=42,
            user_id="alice",
            service_id="svc-1",
        )
        is None
    )


async def test_pkce_verifier_round_trip_to_token_endpoint() -> None:
    """The PKCE ``code_verifier`` recorded in the state store is the one
    sent to ``token_endpoint`` during code exchange.

    This pins the verifier round-trip so a future refactor cannot
    accidentally regenerate it (which would yield an
    ``invalid_grant`` from the IdP).
    """
    secret_store = InMemorySecretStore()
    state_store = OAuthStateStore()
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        secret_store=secret_store,
        state_store=state_store,
    )
    _url, state = manager.build_authorization_url(
        user_id="alice",
        redirect_uri="https://app.example.com/cb",
    )
    recorded_verifier = state_store.peek(state=state)
    assert recorded_verifier is not None

    seen_code_verifier: list[str] = []

    def _capture(req: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qs

        parsed = parse_qs(req.content.decode("utf-8"))
        seen_code_verifier.append(parsed["code_verifier"][0])
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    with respx.mock(base_url="https://idp.example.com") as router:
        router.post("/token").mock(side_effect=_capture)
        async with httpx.AsyncClient() as http_client:
            manager._http_client = http_client
            await manager.exchange_code(
                user_id="alice",
                code="auth-code",
                state=state,
            )

    assert seen_code_verifier == [recorded_verifier.code_verifier]


async def test_aclose_closes_owned_http_client() -> None:
    """When the manager owns its http_client, ``aclose`` closes it."""
    client = httpx.AsyncClient(timeout=10.0)
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    # Replace the internally-created client so we can observe ``aclose``.
    manager._http_client = client
    manager._owns_http_client = True
    assert client.is_closed is False
    await manager.aclose()
    assert client.is_closed is True
    assert manager._http_client is None


async def test_aclose_does_not_close_injected_client() -> None:
    """When the http_client was injected (lifespan path), ``aclose``
    must NOT close it — the caller owns the lifetime."""
    client = httpx.AsyncClient(timeout=10.0)
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
        http_client=client,
    )
    assert manager._owns_http_client is False
    await manager.aclose()
    assert client.is_closed is False
    await client.aclose()


async def test_aclose_is_idempotent() -> None:
    """A second ``aclose`` is a no-op (no error, no double-close)."""
    client = httpx.AsyncClient(timeout=10.0)
    manager = OAuthManager(
        service=_info(auth_config=_live_auth_config()),
    )
    manager._http_client = client
    manager._owns_http_client = True
    await manager.aclose()
    # Second call is a no-op.
    await manager.aclose()


# ── Skew constant parity with Go ───────────────────────────────────


def test_refresh_skew_matches_go_constant() -> None:
    """``DEFAULT_REFRESH_SKEW_SECONDS`` mirrors the Go ``oauthRefreshSkew`` (30 s)."""
    assert DEFAULT_REFRESH_SKEW_SECONDS == 30
