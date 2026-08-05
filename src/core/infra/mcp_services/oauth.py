"""MCP service OAuth flow — authorize URL, code exchange, refresh, revoke.

The Go ``internal/mcp/oauth_*.go`` layer (≈900 lines of code across
``oauth_lifecycle.go``, ``oauth_manager.go``, ``oauth_state.go`` and
``oauth_tokenstore.go``) implements the MCP-side OAuth 2.0
authorization-code flow with PKCE: discovery + dynamic client
registration + per-(tenant, principal, service) tokens + a CSRF
``state`` store that's kept in-process in the Lite mode or in Redis
once that lands.

This module mirrors the same surface on Python. Two layers ship here:

- :class:`TokenSet` — the wire-shape DTO used by both layers.
- :class:`OAuthStateStore` + :class:`InMemorySecretStore` — process-
  internal implementations of the CSRF state store and the per-user
  token store. The Go versions default to Redis / a DB repository
  when those are available; the Python port keeps the Lite defaults so
  the PR stays self-contained (DB-persisted tokens land in a later
  PR).
- :class:`OAuthManager` — the per-request facade that the web router
  calls today (``start_authorization`` / ``authorization_status`` /
  ``revoke``) and the fuller lifecycle methods needed by PR-17.5b
  (``ensure_authorized`` / ``authorize_url`` / ``exchange_code`` /
  ``refresh`` / ``revoke_token``).

The OAuth lifecycle is opt-in: when no ``transport`` /
``http_client`` / ``secret_store`` are supplied the manager keeps the
PR-17 placeholder behaviour so the legacy endpoints keep returning the
same shapes.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.infra.mcp_services.types import MCPServiceInfo

# Default TTL for in-flight OAuth state. Mirrors ``oauthStateTTL``
# (10 minutes) in ``internal/mcp/oauth_state.go`` — gives the user
# enough time to walk through the browser round-trip while keeping
# the CSRF window narrow.
DEFAULT_STATE_TTL_SECONDS: int = 10 * 60

# Default skeweable lifespan for a fresh access token. The Go side
# leans on ``oauthRefreshSkew = 30s`` to refresh a few seconds early;
# we use the same number so a refresh-on-the-fly path does the right
# thing without the caller setting it explicitly.
DEFAULT_REFRESH_SKEW_SECONDS: int = 30


# ── Token DTO ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenSet:
    """Wire shape mirroring ``transport.Token`` from the Go layer.

    Access + refresh tokens are kept in one place because the
    authorization-code flow (``exchange_code``) and the
    refresh_token_grant flow (``refresh``) both produce a complete
    bundle: callers must persist the new refresh token too or the
    next rotation will lose it.
    """

    access_token: str
    refresh_token: str = ""
    token_type: str = "Bearer"
    scope: str = ""
    expires_at: datetime | None = None
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_expired(self, *, skew_seconds: int = DEFAULT_REFRESH_SKEW_SECONDS) -> bool:
        """Return whether this token will expire within ``skew_seconds``."""
        if self.expires_at is None:
            return False
        cutoff = datetime.now(UTC) + timedelta(seconds=skew_seconds)
        return self.expires_at <= cutoff

    def as_authorization_header(self) -> str:
        """Render the ``Authorization`` header value for an HTTP request."""
        return f"{self.token_type} {self.access_token}".strip()


# ── CSRF state store ─────────────────────────────────────────────────


@dataclass
class StateEntry:
    """One in-flight OAuth attempt keyed by the OAuth ``state`` parameter.

    Public so callers (and tests) can construct entries to call
    :meth:`OAuthStateStore.put` directly without going through
    :meth:`OAuthManager.build_authorization_url`.
    """

    tenant_id: int
    user_id: str
    service_id: str
    code_verifier: str
    client_id: str
    redirect_uri: str
    created_at: float


# Backward alias — the entry used to be private and the previous name
# still appears in callers that imported it before PR-17.5b.
_StateEntry = StateEntry


class OAuthStateStore:
    """Process-internal CSRF state store for in-flight OAuth attempts.

    Mirrors the in-memory fallback branch of
    ``internal/mcp/oauth_state.go``'s ``oauthStateStore``: keys hold
    the PKCE code_verifier + redirect URI + principal identifiers,
    with a TTL so stale entries from abandoned browser round-trips
    cannot be replayed.
    """

    def __init__(self, *, ttl_seconds: int = DEFAULT_STATE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, _StateEntry] = {}

    def put(
        self,
        *,
        state: str,
        entry: StateEntry,
    ) -> None:
        """Record an in-flight OAuth attempt; overwrites any prior value."""
        self._purge_expired()
        self._store[state] = entry

    def take(self, *, state: str) -> StateEntry:
        """Consume a state entry; raises :class:`ValueError` if missing / expired."""
        self._purge_expired()
        entry = self._store.pop(state, None)
        if entry is None:
            raise ValueError(
                f"OAuth state {state!r} is unknown or has expired; "
                "the authorization round-trip likely timed out",
            )
        return entry

    def peek(self, *, state: str) -> StateEntry | None:
        """Return a state entry without consuming it."""
        self._purge_expired()
        return self._store.get(state)

    def _purge_expired(self) -> None:
        """Drop entries whose TTL elapsed; called on every read/write."""
        cutoff = time.monotonic() - self._ttl
        stale = [key for key, entry in self._store.items() if entry.created_at < cutoff]
        for key in stale:
            del self._store[key]


# ── Token store ──────────────────────────────────────────────────────


class TokenStore(Protocol):
    """Pluggable per-(tenant, user, service) token persistence."""

    async def get(self, *, tenant_id: int, user_id: str, service_id: str) -> TokenSet | None: ...

    async def put(
        self,
        *,
        tenant_id: int,
        user_id: str,
        service_id: str,
        token: TokenSet,
    ) -> None: ...

    async def delete(self, *, tenant_id: int, user_id: str, service_id: str) -> None: ...

    async def find_by_access_token(
        self,
        *,
        access_token: str,
    ) -> tuple[int, str, str] | None:
        """Return the (tenant_id, user_id, service_id) of the entry whose access_token matches.

        Used by :meth:`OAuthManager.revoke_token` when the caller hands
        in a :class:`TokenSet` rather than the principal triple.
        Returns ``None`` when no entry matches. Default impl uses an
        O(n) scan; concrete stores can replace with an indexed lookup
        when persistence moves to Redis / DB.
        """


class InMemorySecretStore:
    """Default :class:`TokenStore` — process-internal dict.

    Mirrors the ``tokenStatus`` lookup in ``internal/mcp/oauth_lifecycle.go``
    while staying self-contained. DB persistence lands in a later
    PR; the shape (``TokenSet``) is the persistent contract so
    swapping the store later doesn't change the call sites.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[int, str, str], TokenSet] = {}

    async def get(
        self,
        *,
        tenant_id: int,
        user_id: str,
        service_id: str,
    ) -> TokenSet | None:
        return self._store.get((tenant_id, user_id, service_id))

    async def put(
        self,
        *,
        tenant_id: int,
        user_id: str,
        service_id: str,
        token: TokenSet,
    ) -> None:
        self._store[(tenant_id, user_id, service_id)] = token

    async def delete(self, *, tenant_id: int, user_id: str, service_id: str) -> None:
        self._store.pop((tenant_id, user_id, service_id), None)

    async def find_by_access_token(
        self,
        *,
        access_token: str,
    ) -> tuple[int, str, str] | None:
        for key, token in self._store.items():
            if token.access_token == access_token:
                return key
        return None


# ── Auth-config helpers ──────────────────────────────────────────────


def _auth_type(auth_config: JsonObject | None) -> str:
    """Return the normalised ``auth_type`` field of a persisted auth_config."""
    if not isinstance(auth_config, dict):
        return ""
    raw = auth_config.get("auth_type")
    return raw.strip().lower() if isinstance(raw, str) else ""


def _is_oauth(auth_config: JsonObject | None) -> bool:
    """True when the persisted auth_config uses the OAuth strategy.

    Mirrors ``MCPAuthConfig.IsOAuth()`` in the Go codebase: an empty
    config is NOT OAuth.
    """
    return _auth_type(auth_config) == "oauth"


def _authorization_endpoint(auth_config: JsonObject | None) -> str:
    """Return the override ``authorization_endpoint`` (empty when unset)."""
    if not isinstance(auth_config, dict):
        return ""
    raw = auth_config.get("authorization_endpoint")
    return raw.strip() if isinstance(raw, str) else ""


def _token_endpoint(auth_config: JsonObject | None) -> str:
    """Return the override ``token_endpoint`` (empty when unset)."""
    if not isinstance(auth_config, dict):
        return ""
    raw = auth_config.get("token_endpoint")
    return raw.strip() if isinstance(raw, str) else ""


def _client_id_override(auth_config: JsonObject | None) -> str:
    """Return the override ``client_id`` (empty when unset)."""
    if not isinstance(auth_config, dict):
        return ""
    raw = auth_config.get("client_id")
    return raw.strip() if isinstance(raw, str) else ""


def _scopes(auth_config: JsonObject | None) -> list[str]:
    """Return the configured OAuth scopes (empty list when unset)."""
    if not isinstance(auth_config, dict):
        return []
    raw = auth_config.get("scopes")
    if isinstance(raw, list):
        return [str(scope) for scope in raw if isinstance(scope, str) and scope]
    if isinstance(raw, str):
        return [chunk for chunk in raw.split() if chunk]
    return []


# ── Public DTOs (PR-17 surface kept) ─────────────────────────────────


@dataclass(frozen=True)
class OAuthAuthorizeOutcome:
    """Result of an authorization-URL kickoff."""

    authorization_url: str
    authorization_attempt: str


class OAuthStatusResult(BaseModel):
    """Wire shape mirroring the upstream ``/oauth/status`` envelope."""

    model_config = ConfigDict(frozen=True)

    authorized: bool
    state: str = Field(default="authorized")
    refresh_available: bool = Field(default=False)
    expires_at: datetime | None = Field(default=None)


# ── OAuthManager ─────────────────────────────────────────────────────


class OAuthManager:
    """Per-request MCP OAuth flow surface.

    The constructor is dual-mode:

    - **Legacy mode** (only ``service`` supplied): the manager behaves
      exactly like the PR-17 placeholder so the existing router keeps
      working.
    - **Full lifecycle mode** (also ``transport`` / ``http_client`` /
      ``secret_store`` supplied): the new ``ensure_authorized`` /
      ``authorize_url`` / ``exchange_code`` / ``refresh`` /
      ``revoke_token`` methods activate and talk to a real
      authorization server.
    """

    def __init__(
        self,
        *,
        service: MCPServiceInfo,
        transport: JsonValue | None = None,
        http_client: httpx.AsyncClient | None = None,
        secret_store: TokenStore | None = None,
        state_store: OAuthStateStore | None = None,
    ) -> None:
        self._service = service
        # ``transport`` is reserved for the mark3labs-style transport
        # upgrade — the Python side talks directly to the configured
        # ``authorization_endpoint`` / ``token_endpoint`` via the
        # provided ``http_client``.
        self._transport = transport
        # PR-17.5c: do NOT auto-create an ``httpx.AsyncClient`` here.
        # The legacy PR-17 path constructed one per request and
        # never closed it, leaking one TCP/TLS connection per call.
        # Callers that need the live lifecycle (the lifespan-wired
        # factory and tests) MUST inject ``http_client`` explicitly;
        # legacy-mode callers get ``http_client=None`` and the
        # lifecycle methods raise a clear ``oauth_not_configured``
        # if they try to use it.
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._secret_store: TokenStore = secret_store or InMemorySecretStore()
        self._state_store = state_store or OAuthStateStore()

    # ── Shared helpers (legacy mode + full mode) ────────────────────

    @property
    def service(self) -> MCPServiceInfo:
        """Return the wired service. Exposed for callers that need it."""
        return self._service

    @property
    def state_store(self) -> OAuthStateStore:
        """Return the OAuth CSRF state store. Exposed for tests."""
        return self._state_store

    # ── Legacy PR-17 surface (router still uses these) ──────────────

    def start_authorization(
        self,
        *,
        redirect_uri: str,
        frontend_redirect: str | None,
        user_id: str,
    ) -> OAuthAuthorizeOutcome:
        """Kick off an OAuth attempt; return URL + attempt id.

        When full-mode dependencies are wired in, this delegates to
        :meth:`build_authorization_url` so the legacy router endpoint
        benefits from PKCE and the real ``authorization_endpoint``.
        Otherwise the PR-17 placeholder shape is preserved so the
        routes keep working.
        """
        if not _is_oauth(self._service.auth_config):
            raise ValidationError(
                code="mcp_service.oauth_not_configured",
                message="MCP service is not configured to use OAuth",
            )
        if not redirect_uri:
            raise ValidationError(
                code="mcp_service.redirect_uri_required",
                message="redirect_uri is required",
            )
        if not user_id:
            raise NotFoundError(
                code="mcp_service.user_missing",
                message="authenticated user is required",
            )

        attempt = uuid.uuid4().hex
        # Full lifecycle mode: build a real PKCE-flavoured authorize URL.
        if self._lifecycle_ready():
            url, _state = self.build_authorization_url(
                user_id=user_id,
                redirect_uri=redirect_uri,
            )
            return OAuthAuthorizeOutcome(
                authorization_url=url,
                authorization_attempt=attempt,
            )
        # Placeholder URL: kept identical to PR-17 so existing UI tests
        # continue to pass when the service is configured for OAuth but
        # the lifespan has not wired the live dependencies.
        del frontend_redirect  # accepted for API stability
        url = f"{redirect_uri}?attempt={attempt}&user={user_id}"
        return OAuthAuthorizeOutcome(
            authorization_url=url,
            authorization_attempt=attempt,
        )

    def authorization_status(self, *, user_id: str) -> OAuthStatusResult:
        """Return the per-user authorization status."""
        if not user_id:
            raise NotFoundError(
                code="mcp_service.user_missing",
                message="authenticated user is required",
            )
        # Async lookup is intentionally skipped on the sync legacy path;
        # full-mode callers should consult :meth:`ensure_authorized`.
        return OAuthStatusResult(
            authorized=False,
            state="pending",
            refresh_available=False,
            expires_at=None,
        )

    def revoke(self, *, user_id: str) -> None:
        """Drop the per-user token; legacy mode is a no-op."""
        if not user_id:
            raise NotFoundError(
                code="mcp_service.user_missing",
                message="authenticated user is required",
            )
        # Full-mode callers should use ``revoke_token`` explicitly; the
        # legacy path is intentionally a no-op so an accidental wire
        # attempt doesn't drop tokens during a fresh start-up.

    # ── PR-17.5b lifecycle surface ──────────────────────────────────

    def _lifecycle_ready(self) -> bool:
        """True when ``authorize_url`` / ``exchange_code`` / ``refresh`` are usable."""
        if self._http_client is None:
            return False
        endpoint = _authorization_endpoint(self._service.auth_config)
        token_endpoint = _token_endpoint(self._service.auth_config)
        return bool(endpoint and token_endpoint)

    def build_authorization_url(
        self,
        *,
        user_id: str,
        redirect_uri: str,
        scope: list[str] | None = None,
        client_id: str | None = None,
    ) -> tuple[str, str]:
        """Construct an RFC 6749 authorization-code URL with PKCE.

        Returns ``(authorization_url, state)``. The state is also
        recorded in :attr:`state_store` so :meth:`exchange_code` can
        reject callbacks that don't match the original request.
        """
        if not _is_oauth(self._service.auth_config):
            raise ValidationError(
                code="mcp_service.oauth_not_configured",
                message="MCP service is not configured for live OAuth",
            )
        auth_endpoint = _authorization_endpoint(self._service.auth_config)
        token_endpoint = _token_endpoint(self._service.auth_config)
        if not (auth_endpoint and token_endpoint):
            raise ValidationError(
                code="mcp_service.oauth_endpoints_missing",
                message=(
                    "auth_config must supply both authorization_endpoint "
                    "and token_endpoint for live OAuth"
                ),
            )
        client_id = client_id or _client_id_override(self._service.auth_config)
        if not client_id:
            raise ValidationError(
                code="mcp_service.oauth_client_id_missing",
                message="auth_config.client_id is required for live OAuth",
            )
        scopes = scope or _scopes(self._service.auth_config)
        state = secrets.token_urlsafe(32)
        code_verifier = _generate_code_verifier()
        challenge = _code_challenge(code_verifier)
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if scopes:
            params["scope"] = " ".join(scopes)
        authorization_url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
        self._state_store.put(
            state=state,
            entry=_StateEntry(
                tenant_id=self._service.tenant_id,
                user_id=user_id,
                service_id=self._service.id,
                code_verifier=code_verifier,
                client_id=client_id,
                redirect_uri=redirect_uri,
                created_at=time.monotonic(),
            ),
        )
        return authorization_url, state

    async def exchange_code(
        self,
        *,
        user_id: str,
        code: str,
        state: str,
        redirect_uri: str | None = None,
    ) -> TokenSet:
        """Exchange an authorization ``code`` for a :class:`TokenSet`.

        Mirrors the ``CompleteAuthorization`` half of Go's
        ``OAuthManager``: take the CSRF state from
        :attr:`state_store` to fetch the matching PKCE verifier +
        client_id, POST to the configured ``token_endpoint``, and
        persist the returned tokens via :attr:`_secret_store`.
        """
        if self._http_client is None:
            raise ValidationError(
                code="mcp_service.oauth_not_configured",
                message="OAuth live lifecycle needs an http_client",
            )
        try:
            entry = self._state_store.take(state=state)
        except ValueError as exc:
            raise ValidationError(
                code="mcp_service.oauth_state_invalid",
                message=str(exc),
            ) from exc
        if entry.user_id != user_id or entry.service_id != self._service.id:
            raise ValidationError(
                code="mcp_service.oauth_state_mismatch",
                message=(
                    "the OAuth state belongs to a different user or service; "
                    "refusing to exchange the code"
                ),
            )
        token_endpoint = _token_endpoint(self._service.auth_config)
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": entry.client_id,
            "redirect_uri": redirect_uri or entry.redirect_uri,
            "code_verifier": entry.code_verifier,
        }
        token = await _post_token_request(
            http_client=self._http_client,
            token_endpoint=token_endpoint,
            payload=payload,
        )
        await self._secret_store.put(
            tenant_id=entry.tenant_id,
            user_id=entry.user_id,
            service_id=entry.service_id,
            token=token,
        )
        return token

    async def refresh(
        self,
        *,
        token: TokenSet,
        tenant_id: int,
        user_id: str,
        service_id: str,
    ) -> TokenSet:
        """Rotate ``token`` via refresh_token_grant and persist the result.

        Mirrors Go's ``oauthRuntime.refreshWithLease`` (without the
        distributed-lease layer; the Python side keeps refreshes
        single-flight within one process — concurrent refresh is
        coalesced in PR-17.7+ once we persist tokens in Redis).

        The rotated pair is persisted to :attr:`_secret_store` before
        being returned so the next refresh uses the new refresh token
        and the next read sees the rotated access token.
        """
        if self._http_client is None:
            raise ValidationError(
                code="mcp_service.oauth_not_configured",
                message="OAuth live lifecycle needs an http_client",
            )
        if not token.refresh_token:
            raise ValidationError(
                code="mcp_service.oauth_no_refresh_token",
                message="token has no refresh_token and cannot be rotated",
            )
        token_endpoint = _token_endpoint(self._service.auth_config)
        client_id = _client_id_override(self._service.auth_config)
        if not client_id:
            raise ValidationError(
                code="mcp_service.oauth_client_id_missing",
                message="auth_config.client_id is required for live OAuth",
            )
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": client_id,
        }
        scopes = _scopes(self._service.auth_config)
        if scopes:
            payload["scope"] = " ".join(scopes)
        rotated = await _post_token_request(
            http_client=self._http_client,
            token_endpoint=token_endpoint,
            payload=payload,
        )
        # Persist before returning so callers see the rotated pair on
        # the next ``get`` and the next ``refresh`` uses the new
        # ``refresh_token`` (PR-17.5c H2).
        await self._secret_store.put(
            tenant_id=tenant_id,
            user_id=user_id,
            service_id=service_id,
            token=rotated,
        )
        return rotated

    async def ensure_authorized(
        self,
        *,
        tenant_id: int,
        user_id: str,
        service_id: str,
        scope: list[str] | None = None,
    ) -> TokenSet:
        """Return a usable access token; refresh on the fly when near expiry.

        This is the entry point the connection manager can call before
        opening a new session to make sure the OAuth bearer token is
        fresh. The refresh-on-the-fly path is a no-op when the stored
        token is still usable; when it has expired but has a refresh
        token, :meth:`refresh` rotates and persists the pair before
        returning.
        """
        del scope  # scope is passed to build_authorization_url on first use
        existing = await self._secret_store.get(
            tenant_id=tenant_id,
            user_id=user_id,
            service_id=service_id,
        )
        if existing is None:
            raise NotFoundError(
                code="mcp_service.oauth_unauthorized",
                message=(
                    "no MCP OAuth token is stored for this user/service; "
                    "start the authorization-code flow first"
                ),
            )
        if not existing.is_expired():
            return existing
        if not existing.refresh_token:
            # PR-17.5c H3: drop the unrecoverable token so the next
            # caller sees a clean "unauthorized" state instead of a
            # ghost token that can never be refreshed.
            await self._secret_store.delete(
                tenant_id=tenant_id,
                user_id=user_id,
                service_id=service_id,
            )
            raise NotFoundError(
                code="mcp_service.oauth_expired",
                message="access token has expired and no refresh_token is available",
            )
        return await self.refresh(
            token=existing,
            tenant_id=tenant_id,
            user_id=user_id,
            service_id=service_id,
        )

    async def revoke_token(
        self,
        *,
        token: TokenSet | None = None,
        tenant_id: int | None = None,
        user_id: str | None = None,
        service_id: str | None = None,
    ) -> None:
        """Drop the token entry from the in-process secret store.

        Accepts either the exact ``(tenant_id, user_id, service_id)``
        triple (mirrors Go's ``OAuthManager.Revoke``), or a ``token``
        whose ``access_token`` the manager finds via the store's
        ``find_by_access_token`` lookup.

        Passing ``None`` for both forms is a no-op so legacy callers do
        not silently delete state. PR-17.5c C1: the ``token=`` form
        now actually deletes the matching entry; the previous
        implementation passed ``token.access_token`` as a user_id and
        silently dropped nothing.
        """
        if tenant_id is not None and user_id is not None and service_id is not None:
            await self._secret_store.delete(
                tenant_id=tenant_id,
                user_id=user_id,
                service_id=service_id,
            )
            return
        if token is not None:
            key = await self._secret_store.find_by_access_token(
                access_token=token.access_token,
            )
            if key is None:
                return
            resolved_tenant_id, resolved_user_id, resolved_service_id = key
            await self._secret_store.delete(
                tenant_id=resolved_tenant_id,
                user_id=resolved_user_id,
                service_id=resolved_service_id,
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client when this manager owns it.

        Idempotent: a second call after the client was already closed
        is a no-op. When ``http_client`` was injected by the caller
        (the lifespan-wired factory path), ``aclose`` deliberately does
        NOT close it — the caller owns the lifetime.
        """
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._owns_http_client = False


# ── Module helpers ───────────────────────────────────────────────────


def _lifecycle_enabled(auth_config: JsonObject | None) -> bool:
    """True when the auth_config has enough fields to run the live lifecycle."""
    return _is_oauth(auth_config) and bool(
        _authorization_endpoint(auth_config) and _token_endpoint(auth_config)
    )


def _generate_code_verifier() -> str:
    """Generate a high-entropy PKCE code verifier (RFC 7636 §4.1)."""
    # 32 bytes -> 43 url-safe base64 chars, well above the 43-char
    # minimum and below the 128-char maximum.
    return secrets.token_urlsafe(32)


def _code_challenge(verifier: str) -> str:
    """Derive a S256 PKCE code challenge from a verifier (RFC 7636 §4.2)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _b64url(digest)


def _b64url(data: bytes) -> str:
    """RFC 4648 §5 base64url (no padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


async def _post_token_request(
    *,
    http_client: httpx.AsyncClient,
    token_endpoint: str,
    payload: dict[str, str],
) -> TokenSet:
    """POST ``payload`` to ``token_endpoint`` and decode the ``TokenSet``."""
    try:
        response = await http_client.post(
            token_endpoint,
            data=payload,
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise ValidationError(
            code="mcp_service.oauth_transport_error",
            message=f"OAuth token endpoint failed: {type(exc).__name__}: {exc}",
        ) from exc
    if response.status_code >= 400:
        raise ValidationError(
            code="mcp_service.oauth_token_rejected",
            message=(f"token endpoint returned HTTP {response.status_code}: {response.text[:200]}"),
        )
    try:
        decoded: JsonObject = response.json()
    except Exception as exc:  # pragma: no cover - depends on server
        raise ValidationError(
            code="mcp_service.oauth_token_malformed",
            message=f"token endpoint response is not JSON: {exc}",
        ) from exc
    if not isinstance(decoded, dict):
        raise ValidationError(
            code="mcp_service.oauth_token_malformed",
            message="token endpoint response must be a JSON object",
        )
    return _decode_token_payload(decoded)


def _decode_token_payload(decoded: JsonObject) -> TokenSet:
    """Translate an OAuth 2.0 token response into a :class:`TokenSet`."""
    access_token = decoded.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValidationError(
            code="mcp_service.oauth_token_malformed",
            message="token response is missing a usable access_token",
        )
    refresh_token = decoded.get("refresh_token")
    token_type = decoded.get("token_type")
    scope = decoded.get("scope")
    expires_in = decoded.get("expires_in")
    expires_at: datetime | None = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expires_at = datetime.now(UTC) + timedelta(seconds=float(expires_in))
    return TokenSet(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else "",
        token_type=token_type if isinstance(token_type, str) and token_type else "Bearer",
        scope=scope if isinstance(scope, str) else "",
        expires_at=expires_at,
    )


__all__ = [
    "DEFAULT_REFRESH_SKEW_SECONDS",
    "DEFAULT_STATE_TTL_SECONDS",
    "InMemorySecretStore",
    "OAuthAuthorizeOutcome",
    "OAuthManager",
    "OAuthStateStore",
    "OAuthStatusResult",
    "TokenSet",
    "TokenStore",
]
