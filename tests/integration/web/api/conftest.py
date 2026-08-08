"""Per-router ``default_create_*_request`` fixtures shared across API tests.

Per the alignment doc §5.4 every router that exposes a ``POST /...``
create endpoint has a ``default_create_<router>_request`` fixture
returning a minimum-valid Pydantic ``model_dump()`` for the request
body. Knowledge-be does not yet use per-router ``conftest.py`` files
(the doc reserves them for heavyweight routers), so the fixtures
live here in the shared ``tests/integration/web/api/conftest.py``
and are resolved by pytest's parent-directory-walk for every test
under ``tests/integration/web/api/<router>/``.

The fixtures are ``function``-scoped so each test can rely on a
fresh dict. They are deliberately minimal so the failing-path tests
can override individual fields without re-declaring the full body.
"""

from __future__ import annotations

import pytest

from src.core.contracts.infra import (
    CreateDataSourceRequest,
    CreateMCPServiceRequest,
    CreateModelRequest,
    CreateStorageBackendRequest,
    CreateVectorStoreRequest,
)


@pytest.fixture
def default_create_datasource_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /datasource``.

    Connector ``type`` is set to a string placeholder; tests that
    exercise the connector registry override this with an enum the
    ``StubConnector`` double recognises.
    """
    return CreateDataSourceRequest(
        knowledge_base_id="kb-1",
        name="placeholder",
        type="placeholder",
    ).model_dump(mode="json", exclude_none=True)


@pytest.fixture
def default_create_initialization_request() -> dict[str, object]:
    """Minimal-valid body for the ``POST /initialization/*`` probe endpoints.

    Initialization has several POST endpoints (each with its own
    request DTO). The shared fixture covers the most-exercised shape
    (``ModelTestRequest``); per-endpoint tests can re-declare their
    own body when the probe needs provider-specific fields.
    """
    return {
        "modelName": "stub-model",
        "baseUrl": "https://example.com/v1",
    }


@pytest.fixture
def default_create_mcp_services_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /mcp-services``."""
    return CreateMCPServiceRequest(
        name="placeholder",
        transport_type="stdio",
    ).model_dump(mode="json", exclude_none=True)


@pytest.fixture
def default_create_models_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /models``.

    The TypeError-prone field is ``parameters``: the MCP service
    catalog expects a ``ModelParameters`` wire object, so an empty
    dict is rejected — supply a minimal one with just ``provider``
    set so downstream tests have a known surface to mutate.
    """
    return CreateModelRequest(
        name="placeholder",
        type="Embedding",
        source="builtin",
        parameters={"provider": "openai"},
    ).model_dump(mode="json", exclude_none=True)


@pytest.fixture
def default_create_storage_backends_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /storage-backends``.

    ``provider`` is set to ``"minio"`` because the MCP service doubles
    used elsewhere assume that driver; tests that need a different
    provider override this field.
    """
    return CreateStorageBackendRequest(
        name="placeholder",
        provider="minio",
    ).model_dump(mode="json", exclude_none=True)


@pytest.fixture
def default_create_system_request() -> dict[str, object]:
    """Minimal-valid body for ``PUT /system/admin/settings/{key}``.

    The system router has no POST create endpoint — its writable
    surface is ``PUT /system/admin/settings/{key}``. The fixture
    emits a value-shape that the registry accepts for a string
    setting so tests can mutate it freely.
    """
    return {"value": "placeholder"}


@pytest.fixture
def default_create_tenant_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /tenants``.

    ``name`` is the only mandatory field; ``description`` and
    ``business`` are optional. The fixture omits ``retriever_engines``
    so the service applies the default.
    """
    return {"name": "placeholder", "description": "default create tenant fixture"}


@pytest.fixture
def default_create_register_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /auth/register``.

    Matches the three required fields on
    ``src.core.contracts.auth.RegisterRequest``: ``username``,
    ``email`` and ``password``.
    """
    return {
        "username": "fixture",
        "email": "fixture@example.test",
        "password": "fixture-password",
    }


@pytest.fixture
def default_create_tag_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /knowledge-bases/{id}/tags``.

    ``name`` is the only mandatory field on
    ``src.core.contracts.knowledge.CreateTagRequest``; ``color`` and
    ``sort_order`` are optional and omitted so the service applies its
    defaults.
    """
    return {"name": "placeholder"}


@pytest.fixture
def default_create_vector_stores_request() -> dict[str, object]:
    """Minimal-valid body for ``POST /vector-stores``."""
    return CreateVectorStoreRequest(
        name="placeholder",
        engine_type="postgres",
        connection_config={"use_default_connection": True},
    ).model_dump(mode="json", exclude_none=True)


__all__ = [
    "default_create_datasource_request",
    "default_create_initialization_request",
    "default_create_mcp_services_request",
    "default_create_models_request",
    "default_create_register_request",
    "default_create_storage_backends_request",
    "default_create_system_request",
    "default_create_tag_request",
    "default_create_tenant_request",
    "default_create_vector_stores_request",
]
