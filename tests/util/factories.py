"""Polyfactory factories exposed as pytest fixtures.

Each factory class is decorated with ``@register_fixture(scope="session")``
so polyfactory registers it as a session-scoped pytest fixture under its
snake-cased class name (e.g. ``UserFactory`` -> ``user_factory``). Test
modules can then request the fixture by name: ``def test_x(user_factory)``.

Two practical notes:

1. The row models derive from ``src.common.table_model.TableModel``, which
   sets ``model_config = ConfigDict(frozen=True)``. polyfactory does not
   need ``frozen=False`` to populate them - it routes through
   ``model_construct`` internally and writes the attributes before the
   freeze is applied. We keep the defaults.
2. Some fields carry ``JsonObject`` / ``list[JsonObject]`` payloads whose
   shape is domain-driven (e.g. ``Tenant.retriever_engines``).
   polyfactory's faker provider produces dicts of primitives, which is
   enough for the row-shape assertions we run in unit tests.
"""

from __future__ import annotations

from polyfactory.factories.pydantic_factory import ModelFactory
from polyfactory.pytest_plugin import register_fixture

from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.db.models.infra.model import Model
from src.db.models.infra.vector_store import VectorStore
from src.db.models.infra.web_search_provider import WebSearchProvider
from src.db.models.storage_backend import StorageBackend
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from src.db.models.tenants.tenant_invitations import TenantInvitation
from src.db.models.tenants.tenant_kv import TenantKV
from src.db.models.tenants.tenant_members import TenantMember
from src.db.models.tenants.tenants import Tenant


@register_fixture(scope="session")
class UserFactory(ModelFactory[User]):
    __model__ = User


@register_fixture(scope="session")
class AuthTokenFactory(ModelFactory[AuthToken]):
    __model__ = AuthToken


@register_fixture(scope="session")
class TenantFactory(ModelFactory[Tenant]):
    __model__ = Tenant


@register_fixture(scope="session")
class TenantMemberFactory(ModelFactory[TenantMember]):
    __model__ = TenantMember


@register_fixture(scope="session")
class TenantAPIKeyFactory(ModelFactory[TenantAPIKey]):
    __model__ = TenantAPIKey


@register_fixture(scope="session")
class TenantKVFactory(ModelFactory[TenantKV]):
    __model__ = TenantKV


@register_fixture(scope="session")
class TenantInvitationFactory(ModelFactory[TenantInvitation]):
    __model__ = TenantInvitation


@register_fixture(scope="session")
class AuditLogFactory(ModelFactory[AuditLog]):
    __model__ = AuditLog


@register_fixture(scope="session")
class SystemSettingFactory(ModelFactory[SystemSetting]):
    __model__ = SystemSetting


@register_fixture(scope="session", name="model_factory")
class ModelRowFactory(ModelFactory[Model]):
    __model__ = Model


@register_fixture(scope="session")
class StorageBackendFactory(ModelFactory[StorageBackend]):
    __model__ = StorageBackend


@register_fixture(scope="session")
class VectorStoreFactory(ModelFactory[VectorStore]):
    __model__ = VectorStore


@register_fixture(scope="session")
class WebSearchProviderFactory(ModelFactory[WebSearchProvider]):
    __model__ = WebSearchProvider


__all__ = [
    "AuditLogFactory",
    "AuthTokenFactory",
    "ModelRowFactory",
    "StorageBackendFactory",
    "SystemSettingFactory",
    "TenantAPIKeyFactory",
    "TenantFactory",
    "TenantInvitationFactory",
    "TenantKVFactory",
    "TenantMemberFactory",
    "UserFactory",
    "VectorStoreFactory",
    "WebSearchProviderFactory",
]
