"""Conftest for the ``tests.util`` package.

The package hosts shared test scaffolding (factories, base classes).
This conftest has one job: expose each factory class defined in
:mod:`tests.util.factories` as a pytest fixture so test modules in
``tests/util/`` and elsewhere can request them by name
(``user_factory``, ``tenant_factory``, …).
"""

from __future__ import annotations

import pytest

from tests.util import factories as _factories

# Maps the snake_case fixture name the tests use -> the factory class
# that satisfies it. Keep this table small and explicit so a typo in
# a new entry fails at import time, not as a confusing test failure.
_FACTORY_FIXTURES: tuple[tuple[str, type], ...] = (
    ("user_factory", _factories.UserFactory),
    ("auth_token_factory", _factories.AuthTokenFactory),
    ("tenant_factory", _factories.TenantFactory),
    ("tenant_member_factory", _factories.TenantMemberFactory),
    ("tenant_api_key_factory", _factories.TenantAPIKeyFactory),
    ("tenant_kv_factory", _factories.TenantKVFactory),
    ("tenant_invitation_factory", _factories.TenantInvitationFactory),
    ("audit_log_factory", _factories.AuditLogFactory),
    ("system_setting_factory", _factories.SystemSettingFactory),
    ("model_factory", _factories.ModelFactory_),
    ("storage_backend_factory", _factories.StorageBackendFactory),
    ("vector_store_factory", _factories.VectorStoreFactory),
    ("web_search_provider_factory", _factories.WebSearchProviderFactory),
)


for _fixture_name, _factory_cls in _FACTORY_FIXTURES:

    def _make(_cls: type) -> type:
        @pytest.fixture(name=_fixture_name)
        def _fixture() -> type:
            return _cls

        return _fixture

    globals()[_fixture_name] = _make(_factory_cls)  # type: ignore[arg-type]
