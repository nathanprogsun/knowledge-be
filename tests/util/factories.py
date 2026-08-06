"""Polyfactory factories for the test suite.

The factories here are deliberately small: one factory per row model
under ``src/db/models/``. They are exposed as pytest fixtures by
:mod:`tests.util.conftest` (which wraps each class in ``pytest.fixture``
under the documented fixture name) — so test files can request them
by name, e.g. ``def test_x(user_factory)``.

Two practical notes:

1. The row models derive from ``src.common.table_model.TableModel``,
   which sets ``model_config = ConfigDict(frozen=True)``. polyfactory
   does not need ``frozen=False`` to populate them — it routes through
   ``model_construct`` internally and writes the attributes before the
   freeze is applied. We keep the defaults.
2. Some fields carry ``JsonObject`` / ``list[JsonObject]`` payloads
   whose shape is domain-driven (e.g. ``Tenant.retriever_engines``).
   polyfactory's faker provider produces dicts of primitives, which
   is enough for the row-shape assertions we run in unit tests.
"""

from __future__ import annotations

from polyfactory.factories.pydantic_factory import ModelFactory

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


class UserFactory(ModelFactory[User]):
    __model__ = User


class AuthTokenFactory(ModelFactory[AuthToken]):
    __model__ = AuthToken


class TenantFactory(ModelFactory[Tenant]):
    __model__ = Tenant


class TenantMemberFactory(ModelFactory[TenantMember]):
    __model__ = TenantMember


class TenantAPIKeyFactory(ModelFactory[TenantAPIKey]):
    __model__ = TenantAPIKey


class TenantKVFactory(ModelFactory[TenantKV]):
    __model__ = TenantKV


class TenantInvitationFactory(ModelFactory[TenantInvitation]):
    __model__ = TenantInvitation


class AuditLogFactory(ModelFactory[AuditLog]):
    __model__ = AuditLog


class SystemSettingFactory(ModelFactory[SystemSetting]):
    __model__ = SystemSetting


class ModelFactory_(ModelFactory[Model]):
    __model__ = Model


class StorageBackendFactory(ModelFactory[StorageBackend]):
    __model__ = StorageBackend


class VectorStoreFactory(ModelFactory[VectorStore]):
    __model__ = VectorStore


class WebSearchProviderFactory(ModelFactory[WebSearchProvider]):
    __model__ = WebSearchProvider


__all__ = [
    "AuthTokenFactory",
    "AuditLogFactory",
    "ModelFactory_",
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
