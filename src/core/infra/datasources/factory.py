"""Data-source request-scoped service factory.

The single sanctioned construction point for ``DataSourceService``:
repositories are built here on the shared per-request ``AsyncSession``,
so a mutation and its audit row commit together. ``web`` never imports
``db`` — ``src/web/deps/infra_datasources.py`` forwards to this module.

The connector registry is process-wide (connectors are stateless; the
per-tenant credentials arrive as a call argument), so it is built once at
import time rather than per request.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.datasource_connectors import build_connector_registry_entries
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.core.system.audit_service import AuditLogService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.datasource_repository import DataSourceRepository, SyncLogRepository


def _build_connector_registry() -> ConnectorRegistry:
    """Assemble the registry from the concrete ``ai`` connectors.

    Registry construction lives here, not in ``ai``: ``ConnectorRegistry``
    is a ``core`` type and the layer rules forbid ``ai -> core``. ``ai``
    supplies the instances; ``core`` decides what is live.
    """
    registry = ConnectorRegistry()
    for connector in build_connector_registry_entries():
        registry.register(connector)
    return registry


# Process-wide connector registry. Connectors hold no per-request state;
# rebuilding it per request would re-run 13 registrations on every call.
_CONNECTOR_REGISTRY: ConnectorRegistry = _build_connector_registry()


def build_datasource_service(session: AsyncSession) -> DataSourceService:
    """Per-request ``DataSourceService`` with fresh repositories.

    All three repositories share ``session`` so a create/update and the
    audit row it emits land in one transaction.
    """
    return DataSourceService(
        ds_repo=DataSourceRepository(session),
        sync_log_repo=SyncLogRepository(session),
        connector_registry=_CONNECTOR_REGISTRY,
        audit_service=AuditLogService(audit_repo=AuditLogRepository(session)),
    )


def get_connector_registry() -> ConnectorRegistry:
    """Return the process-wide connector registry.

    Exposed so the ``GET /datasources/types`` endpoint can list connector
    metadata without building a session-bound service.
    """
    return _CONNECTOR_REGISTRY


__all__ = ["build_datasource_service", "get_connector_registry"]
