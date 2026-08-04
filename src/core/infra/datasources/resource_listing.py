"""Resource browsing — ``ListAvailableResources`` / ``ResolveResourceAncestors``.

Mixed into ``DataSourceService``. Both methods are thin ownership +
config-parsing wrappers around the connector; the hierarchy walk itself
belongs to the connector because only it knows the external system's
shape.

``list_available_resources`` supports lazy loading: ``parent_id=""``
returns the top level, a non-empty value returns that node's direct
children. ``resolve_resource_ancestors`` is the inverse — given a deep
selection, it returns the ids a picker must expand to reveal it, in
O(depth) rather than by re-walking the tree.
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.types import Resource, parse_config
from src.db.models.datasource import DataSource


class ResourceListingMixin:
    """Resource-browsing methods for ``DataSourceService``."""

    _connector_registry: ConnectorRegistry

    async def _require_owned(self, *, id: str, tenant_id: int) -> DataSource:  # pragma: no cover
        raise NotImplementedError

    async def list_available_resources(
        self,
        *,
        id: str,
        tenant_id: int,
        parent_id: str = "",
    ) -> list[Resource]:
        """List the external system's syncable resources, one level deep.

        Pass ``parent_id`` to load a node's direct children on demand;
        omit it for the top level. Connectors with a flat listing return
        an empty list for any non-empty ``parent_id``.
        """
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        connector = self._connector_registry.get(row.type)
        config = parse_config(row.config)
        if config is None:
            raise ValidationError(
                code="datasource.invalid_config",
                message="invalid configuration",
            )
        return await connector.list_resources(config, parent_id)

    async def resolve_resource_ancestors(
        self,
        *,
        id: str,
        tenant_id: int,
        resource_ids: list[str],
    ) -> list[str]:
        """Return the ancestor ids needed to reveal ``resource_ids``.

        An empty request short-circuits before any ownership or connector
        work — Go returns an empty slice immediately, and the picker calls
        this on every edit-form open, including ones with no selection.
        """
        if not resource_ids:
            return []
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        connector = self._connector_registry.get(row.type)
        config = parse_config(row.config)
        if config is None:
            raise ValidationError(
                code="datasource.invalid_config",
                message="invalid configuration",
            )
        return await connector.resolve_resource_ancestors(config, resource_ids)


__all__ = ["ResourceListingMixin"]
