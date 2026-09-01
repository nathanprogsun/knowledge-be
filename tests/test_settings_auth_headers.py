"""Tests for the auth header name constants on `Settings`."""

from __future__ import annotations

from src.settings import Settings


def test_auth_header_user_id_default() -> None:
    assert Settings().auth_header_user_id == "X-User-Id"


def test_auth_header_tenant_id_default() -> None:
    assert Settings().auth_header_tenant_id == "X-Tenant-ID"


def test_auth_header_roles_default() -> None:
    assert Settings().auth_header_roles == "X-Roles"


def test_header_auth_disabled_by_default() -> None:
    """The self-asserted header channel is opt-in (secure default)."""
    assert Settings().header_auth_enabled is False


def test_auto_setup_disabled_by_default() -> None:
    """The anonymous bootstrap endpoint is opt-in (secure default)."""
    assert Settings().auto_setup_enabled is False
