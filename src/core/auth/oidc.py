"""OIDC SSO service - authorization URL + code-exchange login.

Existing-user bind only; unknown emails raise
``oidc.provisioning_unavailable``. Never imports ``web``/``fastapi``;
raises ``ApplicationError`` subclasses only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

from src.common.exception import (
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.common.oidc_client import OidcClient, OIDCUserInfoClaims
from src.core.auth.service import LoginResult, mint_token_pair
from src.core.auth.types import UserInfo
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.settings import get_settings
from src.util.security import _secret

# ── State signing ────────────────────────────────────────────────────

_STATE_MAX_AGE_SECONDS: int = 600  # 10 minutes
_STATE_FUTURE_TOLERANCE_SECONDS: int = 60


@dataclass(frozen=True, slots=True)
class OIDCStatePayload:
    """Signed state carried in the authorize URL and verified on callback."""

    nonce: str
    redirect_uri: str
    iat: int


def _b64url(data: bytes) -> str:
    """base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    """base64url decode, re-adding the padding ``urlsafe_b64decode`` needs."""
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(raw + padding)


def _state_secret() -> str:
    """HMAC key for OIDC state.

    Shares the JWT signing secret (cached, with an ephemeral-random
    fallback when ``JWT_SECRET_KEY`` is unset or ``"change-me"``) - never
    a known constant.
    """
    return _secret()


def _sign_state(payload: OIDCStatePayload) -> str:
    """Return ``base64url(payload).base64url(hmac)`` (tamper-evident state)."""
    raw = json.dumps(
        {
            "nonce": payload.nonce,
            "redirect_uri": payload.redirect_uri,
            "iat": payload.iat,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(_state_secret().encode("utf-8"), raw, hashlib.sha256).digest()
    return _b64url(raw) + "." + _b64url(signature)


def _verify_state(raw: str) -> OIDCStatePayload:
    """Validate HMAC + freshness; raise ``UnauthorizedError`` on any failure."""
    parts = raw.split(".")
    if len(parts) != 2:
        raise UnauthorizedError("invalid OIDC state format", code="oidc.invalid_state")
    try:
        payload_bytes = _b64url_decode(parts[0])
        signature = _b64url_decode(parts[1])
    except ValueError as exc:
        raise UnauthorizedError("invalid OIDC state encoding", code="oidc.invalid_state") from exc
    expected = hmac.new(_state_secret().encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise UnauthorizedError("OIDC state signature mismatch", code="oidc.invalid_state")
    try:
        decoded = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise UnauthorizedError("invalid OIDC state payload", code="oidc.invalid_state") from exc
    if not isinstance(decoded, dict):
        raise UnauthorizedError("invalid OIDC state payload", code="oidc.invalid_state")
    redirect_uri = decoded.get("redirect_uri")
    if not isinstance(redirect_uri, str) or not redirect_uri.strip():
        raise UnauthorizedError("state.redirect_uri is required", code="oidc.invalid_state")
    iat = decoded.get("iat")
    if not isinstance(iat, int):
        raise UnauthorizedError("state.iat is required", code="oidc.invalid_state")
    now = int(time.time())
    if now - iat > _STATE_MAX_AGE_SECONDS or iat - now > _STATE_FUTURE_TOLERANCE_SECONDS:
        raise UnauthorizedError(
            "OIDC state expired or invalid timestamp", code="oidc.invalid_state"
        )
    nonce = decoded.get("nonce")
    if not isinstance(nonce, str) or not nonce.strip():
        raise UnauthorizedError("state.nonce is required", code="oidc.invalid_state")
    return OIDCStatePayload(nonce=nonce, redirect_uri=redirect_uri, iat=iat)


# ── Result DTOs (internal; web layer maps to contracts/auth.py) ────────


@dataclass(frozen=True, slots=True)
class OIDCAuthorizationURL:
    """Result of :meth:`OidcService.get_authorization_url`.

    ``nonce`` is not part of the wire response; the web layer binds it to
    an HttpOnly cookie and verifies it on callback.
    """

    provider_display_name: str
    authorization_url: str
    state: str
    nonce: str


@dataclass(frozen=True, slots=True)
class OIDCCallbackResult:
    """Result of :meth:`OidcService.login_with_oidc`.

    ``success=False`` with a ``message`` is a legitimate non-error outcome
    (the bound user is disabled) returned rather than raised, so the web
    layer serialises the HTTP 200 ``{success:false, message}`` body.
    ``tenant`` / ``memberships`` are intentionally absent; the web
    layer fills them with ``None`` / ``[]``.
    """

    success: bool
    message: str = ""
    user: UserInfo | None = None
    access_token: str = ""
    refresh_token: str = ""
    is_new_user: bool = False


# ── Resolved OIDC config (settings + discovery) ──────────────────────


@dataclass(frozen=True, slots=True)
class _OIDCConfig:
    provider_display_name: str
    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    user_info_endpoint: str
    scopes: tuple[str, ...]
    user_info_mapping_username: str
    user_info_mapping_email: str


# ── Service ──────────────────────────────────────────────────────────


class OidcService:
    """Request-scoped OIDC SSO orchestrator.

    Constructed per request: repos hold the per-request ``AsyncSession``;
    the ``OidcClient`` is stateless and may be shared. The service never
    touches the session directly.
    """

    def __init__(
        self,
        *,
        users_repo: UserRepository,
        tokens_repo: AuthTokenRepository,
        oidc_client: OidcClient,
    ) -> None:
        self._users_repo = users_repo
        self._tokens_repo = tokens_repo
        self._oidc_client = oidc_client

    async def get_authorization_url(
        self,
        *,
        redirect_uri: str,
    ) -> OIDCAuthorizationURL:
        """Build the provider authorize URL + signed state for ``redirect_uri``."""
        redirect_uri = redirect_uri.strip()
        if not redirect_uri:
            raise ValidationError("redirect_uri is required", code="oidc.invalid_request")
        cfg = await self._load_oidc_config()
        nonce = secrets.token_urlsafe(18)
        state = _sign_state(
            OIDCStatePayload(
                nonce=nonce,
                redirect_uri=redirect_uri,
                iat=int(time.time()),
            )
        )
        query = urlencode(
            {
                "response_type": "code",
                "client_id": cfg.client_id,
                "redirect_uri": redirect_uri,
                "scope": " ".join(cfg.scopes),
                "state": state,
            }
        )
        separator = "&" if "?" in cfg.authorization_endpoint else "?"
        return OIDCAuthorizationURL(
            provider_display_name=cfg.provider_display_name,
            authorization_url=cfg.authorization_endpoint + separator + query,
            state=state,
            nonce=nonce,
        )

    async def login_with_oidc(
        self,
        *,
        code: str,
        redirect_uri: str,
    ) -> OIDCCallbackResult:
        """Exchange ``code`` for provider tokens, bind to an existing local user."""
        code = code.strip()
        redirect_uri = redirect_uri.strip()
        if not code:
            raise ValidationError("code is required", code="oidc.invalid_request")
        if not redirect_uri:
            raise ValidationError("redirect_uri is required", code="oidc.invalid_request")
        cfg = await self._load_oidc_config()
        token_resp = await self._oidc_client.exchange_code(
            token_endpoint=cfg.token_endpoint,
            client_id=cfg.client_id,
            client_secret=cfg.client_secret,
            code=code,
            redirect_uri=redirect_uri,
        )
        info: OIDCUserInfoClaims = await self._oidc_client.resolve_userinfo(
            user_info_endpoint=cfg.user_info_endpoint,
            access_token=token_resp.access_token,
            id_token=token_resp.id_token,
            username_claim=cfg.user_info_mapping_username,
            email_claim=cfg.user_info_mapping_email,
        )
        if not info.email:
            raise ValidationError(
                "OIDC provider did not return an email claim",
                code="oidc.missing_email",
            )
        try:
            user_row = await self._users_repo.find_by_email(info.email)
        except NotFoundError:
            raise ExternalServiceError(
                "OIDC new-user provisioning is not available in this build",
                code="oidc.provisioning_unavailable",
            ) from None
        if not user_row.is_active:
            # Returns success=False as a non-error outcome (HTTP 200 body),
            # not a raise, so the web layer produces the same response.
            return OIDCCallbackResult(success=False, message="Account is disabled")
        result: LoginResult = await mint_token_pair(
            tokens_repo=self._tokens_repo,
            info=UserInfo.map_from_db(user_row),
        )
        return OIDCCallbackResult(
            success=True,
            message="登录成功",
            user=result.user,
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            is_new_user=False,
        )

    # ── Internal helpers ────────────────────────────────────────────

    async def _load_oidc_config(self) -> _OIDCConfig:
        """Load + validate OIDC settings, discovering endpoints if needed."""
        settings = get_settings()
        if not settings.oidc_enable:
            raise PermissionDeniedError("OIDC login is disabled", code="oidc.disabled")
        authorization = settings.oidc_authorization_endpoint.strip()
        token = settings.oidc_token_endpoint.strip()
        userinfo = settings.oidc_user_info_endpoint.strip()
        if (not authorization or not token) and settings.oidc_discovery_url.strip():
            doc = await self._oidc_client.discover_endpoints(settings.oidc_discovery_url.strip())
            authorization = authorization or doc.authorization_endpoint
            token = token or doc.token_endpoint
            userinfo = userinfo or doc.user_info_endpoint
        if not authorization or not token:
            raise ExternalServiceError(
                "OIDC authorization_endpoint and token_endpoint are required "
                "(configure them or set oidc_discovery_url)",
                code="oidc.misconfigured",
            )
        return _OIDCConfig(
            provider_display_name=settings.oidc_provider_display_name,
            client_id=settings.oidc_client_id,
            client_secret=settings.oidc_client_secret,
            authorization_endpoint=authorization,
            token_endpoint=token,
            user_info_endpoint=userinfo,
            scopes=tuple(settings.oidc_scopes),
            user_info_mapping_username=settings.oidc_user_info_mapping_username,
            user_info_mapping_email=settings.oidc_user_info_mapping_email,
        )


__all__ = [
    "OIDCAuthorizationURL",
    "OIDCCallbackResult",
    "OIDCStatePayload",
    "OidcService",
]
