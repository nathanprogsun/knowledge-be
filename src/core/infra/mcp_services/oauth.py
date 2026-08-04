"""MCP service OAuth flow surface — authorisation URL, status, revoke.

The Go ``mcp_oauth.go`` handler delegates to an ``mcp.OAuthManager``
that performs RFC 9728 discovery + dynamic client registration + PKCE
and persists per-user tokens. The Python scaffold exposes the same
three endpoints (URL, status, revoke) but without a live provider —
the manager here returns deterministic placeholders so the route
handlers, error envelopes, and per-user scoping are exercised
end-to-end while the live OAuth integration lands in a later
checkpoint.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import NotFoundError, ValidationError
from src.core.infra.mcp_services.types import MCPServiceInfo


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


class OAuthManager:
    """Per-request OAuth flow surface.

    The Go side implements discovery + dynamic registration + PKCE +
    per-user token store; that machinery depends on a live
    authorization server. The Python surface keeps the per-user
    scoping and the deterministic ``authorize_attempt`` id used by
    the callback round-trip; a real OAuth transport is wired in a
    later checkpoint.
    """

    def __init__(self, *, service: MCPServiceInfo) -> None:
        self._service = service

    # ── Authorize URL ──────────────────────────────────────────────

    def start_authorization(
        self,
        *,
        redirect_uri: str,
        frontend_redirect: str | None,
        user_id: str,
    ) -> OAuthAuthorizeOutcome:
        """Kick off an OAuth attempt; return URL + attempt id."""
        if not self._service.auth_config:
            raise ValidationError(
                code="mcp_service.oauth_not_configured",
                message="MCP service is not configured to use OAuth",
            )
        if not redirect_uri:
            raise ValidationError(
                code="mcp_service.redirect_uri_required",
                message="redirect_uri is required",
            )
        attempt = uuid.uuid4().hex
        # Placeholder URL: the real implementation rounds the user out
        # to the discovered authorization server endpoint.
        url = f"{redirect_uri}?attempt={attempt}&user={user_id}"
        return OAuthAuthorizeOutcome(
            authorization_url=url,
            authorization_attempt=attempt,
        )

    # ── Status ─────────────────────────────────────────────────────

    def authorization_status(self, *, user_id: str) -> OAuthStatusResult:
        """Return the per-user authorization status.

        The default implementation reports the user as not yet
        authorized; a real implementation consults the token store.
        """
        if not user_id:
            raise NotFoundError(
                code="mcp_service.user_missing",
                message="authenticated user is required",
            )
        return OAuthStatusResult(
            authorized=False,
            state="pending",
            refresh_available=False,
            expires_at=None,
        )

    # ── Revoke ─────────────────────────────────────────────────────

    def revoke(self, *, user_id: str) -> None:
        """Drop the per-user token; this placeholder is a no-op."""
        if not user_id:
            raise NotFoundError(
                code="mcp_service.user_missing",
                message="authenticated user is required",
            )
        return


__all__ = [
    "OAuthAuthorizeOutcome",
    "OAuthManager",
    "OAuthStatusResult",
]
