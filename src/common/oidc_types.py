"""OIDC domain types.

Decoupled from ``src.common.oidc_client`` so the type vocabulary can grow
without inflating the runtime client module. All types here are pure
data-shape carriers — no I/O, no side effects.

Three layers of abstraction:

1. ``OIDCStandardClaims`` / ``AddressClaim`` — TypedDicts mirroring the
   OIDC Core 1.0 §2 (ID Token) + §5.1 (UserInfo) field declarations.
   Used to document the **known** claim shapes; ``total=False`` because
   different providers expose different subsets.

2. ``OIDCClaimsDict`` — union of ``OIDCStandardClaims`` and a generic
   ``dict[str, JsonValue]`` so callers can either consume the standard
   shape or pass through arbitrary custom claims that go beyond the
   spec. Reuses the canonical ``JsonValue`` from ``src.common.json``
   (Pydantic-backed) rather than redefining a parallel alias.

3. ``OIDCUserInfoMapping`` — frozen dataclass carrying the deploy-time
   claim-name → local-field mapping. Validation rejects empty
   claim names so misconfiguration fails fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypedDict

from src.common.json import JsonValue

# ── OIDC Core 1.0 §2 + §5.1 claim shapes ──────────────────────────────


class AddressClaim(TypedDict, total=False):
    """OIDC Core 5.1.1 ``address`` claim. All keys optional."""

    country: str
    region: str
    locality: str
    street_address: str
    postal_code: str
    formatted: str


class OIDCStandardClaims(TypedDict, total=False):
    """OIDC Core 1.0 §2 (ID Token) + §5.1 (UserInfo) standard claims.

    All keys optional — providers expose different subsets, and ID
    Tokens only require ``iss``, ``sub``, ``aud``, ``exp``, ``iat`` at
    the spec level but the ``iss``/``sub``/``aud``/``exp``/``iat``
    fields are still optional here because invalidating the type when
    a custom provider omits one would defeat the purpose of having a
    single type for the whole claim dict.
    """

    # ── ID Token (§2) ────────────────────────────────────────────────
    iss: str
    sub: str
    aud: str | list[str]  # OIDC §2: string OR array of string
    exp: int  # Unix seconds
    iat: int  # Unix seconds
    nbf: int  # Unix seconds
    auth_time: int  # Unix seconds
    nonce: str
    acr: str
    amr: list[str]
    azp: str
    # ── UserInfo (§5.1) ──────────────────────────────────────────────
    name: str
    given_name: str
    family_name: str
    middle_name: str
    nickname: str
    preferred_username: str
    profile: str
    picture: str
    website: str
    email: str
    email_verified: bool
    gender: str
    birthdate: str  # ISO YYYY-MM-DD
    zoneinfo: str
    locale: str
    phone_number: str
    phone_number_verified: bool
    address: AddressClaim
    updated_at: int  # Unix seconds


# ── Container alias ─────────────────────────────────────────────────


OIDCClaimsDict: TypeAlias = OIDCStandardClaims | dict[str, JsonValue]
"""Union for "OIDC claim dict" — either a known OIDCStandardClaims shape
or an arbitrary dict of custom claims. Replaces ``dict[str, JsonValue]``
at API boundaries where the OIDC claim vocabulary applies."""


# ── Deploy-time claim mapping ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OIDCUserInfoMapping:
    """Configuration: maps OIDC claim names to local UserInfo fields.

    Both fields are required to be non-empty (whitespace-only also
    rejected). Validation lives in ``__post_init__`` so misconfiguration
    fails at construction time, not at first user login.
    """

    username_claim: str
    email_claim: str

    def __post_init__(self) -> None:
        if not self.username_claim or not self.username_claim.strip():
            raise ValueError("OIDCUserInfoMapping.username_claim must be non-empty")
        if not self.email_claim or not self.email_claim.strip():
            raise ValueError("OIDCUserInfoMapping.email_claim must be non-empty")


__all__ = [
    "AddressClaim",
    "OIDCClaimsDict",
    "OIDCStandardClaims",
    "OIDCUserInfoMapping",
]
