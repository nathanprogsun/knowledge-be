"""Unit tests for ``src.ai.oidc_types``.

Pure type-system tests: TypedDict instances are normal dicts at runtime,
so we verify the *type* can be subscripted and field names are valid
rather than enforcing runtime shape.
"""

from __future__ import annotations

import pytest

# ── OIDCUserInfoMapping ──────────────────────────────────────────────


def test_oidc_user_info_mapping_constructs_with_valid_fields() -> None:
    from src.ai.oidc_types import OIDCUserInfoMapping

    m = OIDCUserInfoMapping(username_claim="name", email_claim="email")
    assert m.username_claim == "name"
    assert m.email_claim == "email"


def test_oidc_user_info_mapping_rejects_empty_username_claim() -> None:
    from src.ai.oidc_types import OIDCUserInfoMapping

    with pytest.raises(ValueError, match="username_claim"):
        OIDCUserInfoMapping(username_claim="", email_claim="email")


def test_oidc_user_info_mapping_rejects_empty_email_claim() -> None:
    from src.ai.oidc_types import OIDCUserInfoMapping

    with pytest.raises(ValueError, match="email_claim"):
        OIDCUserInfoMapping(username_claim="name", email_claim="")


def test_oidc_user_info_mapping_rejects_whitespace_only() -> None:
    from src.ai.oidc_types import OIDCUserInfoMapping

    with pytest.raises(ValueError):
        OIDCUserInfoMapping(username_claim="   ", email_claim="email")


def test_oidc_user_info_mapping_is_frozen() -> None:
    from src.ai.oidc_types import OIDCUserInfoMapping

    m = OIDCUserInfoMapping(username_claim="name", email_claim="email")
    with pytest.raises((AttributeError, Exception)):
        m.username_claim = "other"  # type: ignore[misc]


# ── OIDCStandardClaims ───────────────────────────────────────────────


def test_oidc_standard_claims_accepts_typical_payload() -> None:
    from src.ai.oidc_types import OIDCStandardClaims

    claims: OIDCStandardClaims = {
        "sub": "user-123",
        "email": "user@example.com",
        "email_verified": True,
        "aud": ["client-1", "client-2"],
    }
    assert claims["sub"] == "user-123"
    assert claims["email_verified"] is True
    assert claims["aud"] == ["client-1", "client-2"]


def test_oidc_standard_claims_all_keys_optional() -> None:
    """total=False means an empty dict is a valid OIDCStandardClaims."""
    from src.ai.oidc_types import OIDCStandardClaims

    claims: OIDCStandardClaims = {}
    assert isinstance(claims, dict)


def test_oidc_standard_claims_exposes_required_oidc_fields() -> None:
    """Spot-check the OIDC Core 1.0 §2 + §5.1 fields are present."""
    from src.ai.oidc_types import OIDCStandardClaims

    # Use __annotations__ to verify the TypedDict declares the expected keys.
    annotations = OIDCStandardClaims.__annotations__
    expected = {
        "iss",
        "sub",
        "aud",
        "exp",
        "iat",
        "name",
        "email",
        "email_verified",
        "address",
        "updated_at",
    }
    assert expected.issubset(annotations.keys()), (
        f"missing OIDC fields: {expected - annotations.keys()}"
    )


# ── AddressClaim ─────────────────────────────────────────────────────


def test_address_claim_accepts_typical_payload() -> None:
    from src.ai.oidc_types import AddressClaim

    addr: AddressClaim = {
        "country": "US",
        "region": "CA",
        "locality": "San Francisco",
    }
    assert addr["country"] == "US"


def test_address_claim_all_keys_optional() -> None:
    from src.ai.oidc_types import AddressClaim

    addr: AddressClaim = {}
    assert isinstance(addr, dict)


# ── OIDCClaimsDict type alias ────────────────────────────────────────


def test_oidc_claims_dict_accepts_standard_claims() -> None:
    from src.ai.oidc_types import OIDCClaimsDict, OIDCStandardClaims

    standard: OIDCStandardClaims = {"sub": "user-1"}
    # A OIDCStandardClaims instance is also a valid OIDCClaimsDict
    # (TypedDict instances are dicts).
    d: OIDCClaimsDict = standard
    assert d["sub"] == "user-1"


def test_oidc_claims_dict_accepts_custom_claims() -> None:
    from src.ai.oidc_types import OIDCClaimsDict

    custom: OIDCClaimsDict = {
        "tenant_id": "acme",
        "groups": ["admin", "dev"],
        "custom_number": 42,
    }
    assert custom["tenant_id"] == "acme"
    assert custom["groups"] == ["admin", "dev"]
