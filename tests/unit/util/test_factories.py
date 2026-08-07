"""Smoke tests for the polyfactory factories.

Each test only asserts that ``factory.build()`` succeeds and produces
a value on a non-empty field — the goal is to keep the scaffold
exercised by CI so a future change to a row model that breaks the
factory surfaces immediately, not as a cryptic import error in an
unrelated test.
"""

from __future__ import annotations


def test_user_factory_builds(user_factory):
    user = user_factory.build()
    assert user.email


def test_auth_token_factory_builds(auth_token_factory):
    token = auth_token_factory.build()
    assert token.token


def test_tenant_factory_builds(tenant_factory):
    tenant = tenant_factory.build()
    assert tenant.name


def test_tenant_member_factory_builds(tenant_member_factory):
    member = tenant_member_factory.build()
    assert member.user_id


def test_tenant_api_key_factory_builds(tenant_api_key_factory):
    key = tenant_api_key_factory.build()
    assert key.name
    assert key.key_hash


def test_tenant_kv_factory_builds(tenant_kv_factory):
    kv = tenant_kv_factory.build()
    assert kv.key


def test_tenant_invitation_factory_builds(tenant_invitation_factory):
    invitation = tenant_invitation_factory.build()
    assert invitation.role


def test_audit_log_factory_builds(audit_log_factory):
    log = audit_log_factory.build()
    assert log.action


def test_system_setting_factory_builds(system_setting_factory):
    setting = system_setting_factory.build()
    assert setting.key


def test_model_factory_builds(model_factory):
    model = model_factory.build()
    assert model.name
    assert model.type


def test_storage_backend_factory_builds(storage_backend_factory):
    backend = storage_backend_factory.build()
    assert backend.name
    assert backend.provider


def test_vector_store_factory_builds(vector_store_factory):
    store = vector_store_factory.build()
    assert store.name
    assert store.engine_type


def test_web_search_provider_factory_builds(web_search_provider_factory):
    provider = web_search_provider_factory.build()
    assert provider.name
    assert provider.provider
