"""Pure helpers for password hashing and JWT signing.

Two responsibilities:

- ``hash_password`` / ``verify_password`` use the ``bcrypt`` library
  directly. ``passlib`` was the prior choice but its version-detection
  breaks against bcrypt 4.x+, and the project ships bcrypt 5.x.
- ``create_access_token`` / ``create_refresh_token`` / ``decode_token``
  use ``python-jose`` with HS256, matching the upstream
  ``github.com/golang-jwt/jwt/v5`` claim layout.

The JWT secret is loaded lazily from ``Settings.jwt_secret_key``; on
first call a fresh ``Settings`` instance is built via ``get_settings``
and the secret is cached on the module level. The cache is invalidated
by ``reset_secret_cache()`` so tests that mutate the env can re-read.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import bcrypt
from jose import JWTError, jwt

from src.settings import get_settings

# bcrypt cost — upstream uses ``bcrypt.DefaultCost`` (10).
_BCRYPT_ROUNDS: Final = 10

# Secret cache (reset by ``reset_secret_cache`` for tests).
_cached_secret: str | None = None


def reset_secret_cache() -> None:
    """Drop the memoized JWT secret (used by tests that mutate env)."""
    global _cached_secret
    _cached_secret = None


def _secret() -> str:
    global _cached_secret
    if _cached_secret is None:
        secret = get_settings().jwt_secret_key
        if not secret or secret == "change-me":
            # Match the upstream behavior: if no secret is configured,
            # generate an ephemeral one. Restart of the process
            # invalidates all outstanding tokens.
            secret = secrets.token_urlsafe(32)
        _cached_secret = secret
    return _cached_secret


# ── Password helpers ────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt digest of ``plain`` at the configured cost."""
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    digest = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return digest.decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """True iff ``plain`` matches the bcrypt ``hashed`` digest."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ── JWT helpers ─────────────────────────────────────────────────────

ACCESS_TOKEN_TTL: Final = timedelta(hours=24)
REFRESH_TOKEN_TTL: Final = timedelta(days=7)


def create_access_token(
    *,
    user_id: str,
    email: str,
    tenant_id: int | None,
    ttl: timedelta = ACCESS_TOKEN_TTL,
) -> tuple[str, datetime]:
    """Mint an HS256 access JWT. Returns ``(token, expires_at)``.

    Claim layout mirrors the upstream ``generateTokensForTenant``:
    ``user_id``, ``email``, ``tenant_id``, ``exp``, ``iat``, ``type="access"``.
    """
    now = datetime.now(UTC)
    expires_at = now + ttl
    claims: dict[str, Any] = {
        "user_id": user_id,
        "email": email,
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": secrets.token_urlsafe(8),
        "type": "access",
    }
    token = jwt.encode(claims, _secret(), algorithm="HS256")
    return token, expires_at


def create_refresh_token(
    *,
    user_id: str,
    ttl: timedelta = REFRESH_TOKEN_TTL,
) -> tuple[str, datetime]:
    """Mint an HS256 refresh JWT. Returns ``(token, expires_at)``."""
    now = datetime.now(UTC)
    expires_at = now + ttl
    claims: dict[str, Any] = {
        "user_id": user_id,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": secrets.token_urlsafe(8),
        "type": "refresh",
    }
    token = jwt.encode(claims, _secret(), algorithm="HS256")
    return token, expires_at


class TokenError(Exception):
    """Raised when a JWT cannot be decoded or has the wrong shape."""


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify ``token``. Raises ``TokenError`` on any failure.

    Returns the claim dict so callers can inspect ``type``, ``user_id``,
    ``exp`` etc.
    """
    try:
        return jwt.decode(token, _secret(), algorithms=["HS256"])
    except JWTError as exc:
        raise TokenError(str(exc)) from exc


__all__ = [
    "ACCESS_TOKEN_TTL",
    "REFRESH_TOKEN_TTL",
    "TokenError",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "hash_password",
    "reset_secret_cache",
    "verify_password",
]
