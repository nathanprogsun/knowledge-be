"""Connection + credential validation — ``ValidateConnection`` / ``ValidateCredentials``.

Mixed into ``DataSourceService``. Two entry points, both delegating the
actual reachability test to the connector:

``validate_connection(id)``
    Tests a *persisted* source and writes the outcome back: a failure
    flips ``status`` to ``error`` and stores the message; a success clears
    a previously-recorded error. Mirrors Go's ``ValidateConnection``,
    including the status side effects.

``validate_credentials(type, credentials)``
    Tests a raw credential map with **nothing persisted** — the
    "Test Connection" button on the creation form. Mirrors Go's
    ``ValidateCredentials``.

``_validate_config_if_credentialed`` is the shared internal gate used by
create/update: Go skips live validation when no credentials are stored
yet, because a validator with no token to present would always fail.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import ApplicationError, ValidationError
from src.common.json import JsonObject
from src.core.infra.datasources.connector_base import ConnectorRegistry
from src.core.infra.datasources.types import (
    DATA_SOURCE_STATUS_ACTIVE,
    DATA_SOURCE_STATUS_ERROR,
    DataSourceConfig,
    DataSourceInfo,
    parse_config,
)
from src.db.dao.datasource_repository import DataSourceRepository
from src.db.models.datasource import DataSource


class ConnectivityMixin:
    """``ValidateConnection`` / ``ValidateCredentials`` for ``DataSourceService``.

    Declares the collaborators it reads off ``self`` so the mixin
    typechecks standalone; ``DataSourceService.__init__`` assigns them.
    """

    _ds_repo: DataSourceRepository
    _connector_registry: ConnectorRegistry

    async def _require_owned(self, *, id: str, tenant_id: int) -> DataSource:  # pragma: no cover
        raise NotImplementedError

    # ── Persisted source ──────────────────────────────────────────────

    async def validate_connection(self, *, id: str, tenant_id: int) -> DataSourceInfo:
        """Test a stored source's connection and record the outcome.

        Returns the refreshed projection so the caller can surface the new
        ``status`` / ``error_message`` without a second read. Re-raises the
        connector's error after persisting it, so the HTTP layer still
        answers 4xx on an unreachable source.
        """
        row = await self._require_owned(id=id, tenant_id=tenant_id)
        connector = self._connector_registry.get(row.type)
        config = parse_config(row.config)
        if config is None:
            raise ValidationError(
                code="datasource.invalid_config",
                message="invalid configuration",
            )
        try:
            await connector.validate(config)
        except ApplicationError as exc:
            await self._record_validation_failure(row, exc.message)
            raise
        refreshed = await self._clear_validation_error(row)
        return DataSourceInfo.map_from_db(refreshed)

    # ── Raw credentials, nothing persisted ────────────────────────────

    async def validate_credentials(
        self,
        *,
        type: str,
        credentials: JsonObject,
    ) -> None:
        """Test connectivity with a raw credential map, persisting nothing.

        Raises when the connector type is unknown or the credentials do
        not authenticate; returns ``None`` on success.
        """
        connector = self._connector_registry.get(type)
        await connector.validate(DataSourceConfig(type=type, credentials=credentials))

    # ── Shared create/update gate ─────────────────────────────────────

    async def _validate_config_if_credentialed(
        self,
        config: JsonObject | None,
        connector_type: str,
    ) -> None:
        """Validate ``config`` against its connector when it has secrets.

        A config with no configured credentials is accepted unvalidated:
        the source is created in a "needs credentials" state and the user
        completes it through the credential subresource. Validating it
        here would reject every partially-filled form.
        """
        parsed = parse_config(config)
        if parsed is None or not parsed.has_configured_credentials(connector_type):
            return
        connector = self._connector_registry.get(connector_type)
        await connector.validate(parsed)

    # ── Status side effects ───────────────────────────────────────────

    async def _record_validation_failure(self, row: DataSource, message: str) -> DataSource:
        """Flip the row to ``error`` and store the failure message."""
        return await self._ds_repo.update(
            row.model_copy(
                update={
                    "status": DATA_SOURCE_STATUS_ERROR,
                    "error_message": message,
                    "updated_at": datetime.now(UTC),
                }
            )
        )

    async def _clear_validation_error(self, row: DataSource) -> DataSource:
        """Clear a previously-recorded error after a successful test.

        A source that was never in ``error`` is left untouched — Go writes
        only on the error→active transition, so a healthy source's
        ``updated_at`` does not move on every connection test.
        """
        if row.status != DATA_SOURCE_STATUS_ERROR:
            return row
        return await self._ds_repo.update(
            row.model_copy(
                update={
                    "status": DATA_SOURCE_STATUS_ACTIVE,
                    "error_message": "",
                    "updated_at": datetime.now(UTC),
                }
            )
        )


__all__ = ["ConnectivityMixin"]
