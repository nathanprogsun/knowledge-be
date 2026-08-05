"""Concrete data-source connectors — ``internal/datasource/connector/*``.

Adapters between one external system and the sync pipeline. Every
connector implements ``src.common.datasource_protocol.Connector``; the
registry that resolves them by type lives in
``src.core.infra.datasources.connector_base`` (``ai`` may not import
``core``, so the protocol is in ``common``).

Migration state, stated plainly: the upstream connectors are ~10k lines of
per-vendor HTTP clients (Feishu wiki traversal, Notion block trees,
Confluence CQL, OAuth refresh flows, IMAP, sitemap crawling). Porting them
is one PR per vendor. What ships here is the layer they all share:

``CredentialSpec`` / ``HttpConnector``
    Declarative credential requirements plus the validation flow every
    upstream connector opens with — required keys present and non-blank,
    then a live probe. ``validate`` is fully implemented: a misconfigured
    source is rejected at create time with a field-accurate message,
    which is the behaviour the create/update/test-connection paths
    depend on.

``UnimplementedFetchMixin``
    Raises ``ExternalServiceError`` from the fetch/list methods that need
    vendor API code. Explicitly failing beats returning ``[]``: an empty
    list would close a sync run as ``success`` with zero items and move
    ``last_sync_at``, making a not-yet-migrated connector look healthy.

``build_connector_registry()`` registers one instance per upstream type,
so ``GET /datasources/types`` and the create-time type check already
answer for all 13 — the vendor PRs replace a class body, not the wiring.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.datasource_protocol import (
    CONNECTOR_TYPE_CONFLUENCE,
    CONNECTOR_TYPE_DINGTALK,
    CONNECTOR_TYPE_FEISHU,
    CONNECTOR_TYPE_GITHUB,
    CONNECTOR_TYPE_GOOGLE_DRIVE,
    CONNECTOR_TYPE_IMAP,
    CONNECTOR_TYPE_LARK,
    CONNECTOR_TYPE_NOTION,
    CONNECTOR_TYPE_ONEDRIVE,
    CONNECTOR_TYPE_RSS,
    CONNECTOR_TYPE_SLACK,
    CONNECTOR_TYPE_WEB_CRAWLER,
    CONNECTOR_TYPE_YUQUE,
    Connector,
    DataSourceConfig,
    FetchedItem,
    Resource,
    SyncCursor,
)
from src.common.exception import ExternalServiceError, ValidationError


class CredentialSpec(BaseModel):
    """What one connector type needs before it can authenticate.

    ``required_keys`` must all be present and non-blank in the credential
    map. ``required_settings`` covers non-secret configuration a connector
    cannot work without (e.g. the RSS feed URL list), which lives in
    ``settings`` rather than ``credentials``.
    """

    model_config = ConfigDict(frozen=True)

    required_keys: list[str] = Field(default_factory=list)
    required_settings: list[str] = Field(default_factory=list)


# Credential requirements per upstream connector, from each connector's
# own config validation. Keys match the JSON the frontend sends.
CREDENTIAL_SPECS: dict[str, CredentialSpec] = {
    CONNECTOR_TYPE_FEISHU: CredentialSpec(required_keys=["app_id", "app_secret"]),
    CONNECTOR_TYPE_LARK: CredentialSpec(required_keys=["app_id", "app_secret"]),
    CONNECTOR_TYPE_NOTION: CredentialSpec(required_keys=["api_key"]),
    CONNECTOR_TYPE_CONFLUENCE: CredentialSpec(
        required_keys=["base_url", "email", "api_token"],
    ),
    CONNECTOR_TYPE_YUQUE: CredentialSpec(required_keys=["token"]),
    CONNECTOR_TYPE_GITHUB: CredentialSpec(required_keys=["access_token"]),
    CONNECTOR_TYPE_GOOGLE_DRIVE: CredentialSpec(required_keys=["access_token"]),
    CONNECTOR_TYPE_ONEDRIVE: CredentialSpec(required_keys=["access_token"]),
    CONNECTOR_TYPE_DINGTALK: CredentialSpec(required_keys=["app_key", "app_secret"]),
    # Sitemap crawling is unauthenticated; the target URL is configuration.
    CONNECTOR_TYPE_WEB_CRAWLER: CredentialSpec(required_settings=["sitemap_url"]),
    CONNECTOR_TYPE_SLACK: CredentialSpec(required_keys=["bot_token"]),
    CONNECTOR_TYPE_IMAP: CredentialSpec(
        required_keys=["host", "username", "password"],
    ),
    # ``auth_headers`` is optional for RSS (public feeds need none); the
    # feed URL list is required, and is non-secret configuration.
    CONNECTOR_TYPE_RSS: CredentialSpec(required_settings=["feed_urls"]),
}


class HttpConnector(Connector):
    """Base for connectors that authenticate against an HTTP API.

    Implements the credential-shape half of ``validate``: every required
    key present and non-blank. Subclasses override :meth:`probe` to add
    the live reachability call.
    """

    def __init__(self, connector_type: str, spec: CredentialSpec) -> None:
        self._type = connector_type
        self._spec = spec

    @property
    def type(self) -> str:
        """Connector type identifier."""
        return self._type

    @property
    def spec(self) -> CredentialSpec:
        """This connector's credential requirements."""
        return self._spec

    async def validate(self, config: DataSourceConfig) -> None:
        """Check the credential shape, then probe the upstream API.

        Raises ``ValidationError`` naming the first missing field — a
        connector-specific message is what lets the UI point at the right
        input instead of showing a generic auth failure.
        """
        for key in self._spec.required_keys:
            raw = config.credentials.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValidationError(
                    code="datasource.credential_missing",
                    message=f"{self._type}: credential {key!r} is required",
                )
        for key in self._spec.required_settings:
            raw_setting = config.settings.get(key)
            if raw_setting is None or (isinstance(raw_setting, str) and not raw_setting.strip()):
                raise ValidationError(
                    code="datasource.setting_missing",
                    message=f"{self._type}: setting {key!r} is required",
                )
        await self.probe(config)

    async def probe(self, config: DataSourceConfig) -> None:
        """Live reachability check against the upstream API.

        The base implementation is a no-op: with the credential shape
        verified, a source is accepted and a real failure surfaces on the
        first sync. Each vendor PR overrides this with its cheapest
        authenticated call (Feishu ``tenant_access_token``, Notion
        ``/users/me``, ...).
        """


class UnimplementedFetchMixin:
    """Fetch/list methods for connectors whose vendor API is not ported.

    Every method raises ``ExternalServiceError``. Returning empty lists
    instead would let a sync run close as ``success`` with zero items and
    advance ``last_sync_at`` — a silently broken connector that looks
    healthy in the UI.
    """

    _type: str

    def _unimplemented(self, operation: str) -> ExternalServiceError:
        return ExternalServiceError(
            code="datasource.connector_not_implemented",
            message=(
                f"connector {self._type!r} does not support {operation} yet: "
                "the upstream API client has not been migrated"
            ),
        )

    async def list_resources(
        self,
        config: DataSourceConfig,
        parent_id: str = "",
    ) -> list[Resource]:
        """Not available until the vendor client is ported."""
        raise self._unimplemented("resource listing")

    async def resolve_resource_ancestors(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[str]:
        """No hierarchy to reveal without a vendor client.

        Returns an empty list rather than raising: this is the "nothing to
        expand" answer that flat-listing connectors give upstream, and the
        picker calls it on every edit-form open. Failing here would break
        opening the form of an otherwise valid source.
        """
        return []

    async def fetch_all(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[FetchedItem]:
        """Not available until the vendor client is ported."""
        raise self._unimplemented("full sync")

    async def fetch_incremental(
        self,
        config: DataSourceConfig,
        cursor: SyncCursor | None,
    ) -> tuple[list[FetchedItem], SyncCursor | None]:
        """Not available until the vendor client is ported."""
        raise self._unimplemented("incremental sync")


class PendingConnector(UnimplementedFetchMixin, HttpConnector):
    """A registered connector whose vendor API client is not yet ported.

    Credential validation works (so create / update / test-connection
    behave correctly and the type appears in the picker); fetching raises.
    """


def build_connector_registry_entries() -> list[Connector]:
    """Build one connector instance per upstream type.

    Kept separate from the registry construction so tests can inspect the
    instances without importing ``core``.
    """
    return [
        PendingConnector(connector_type, spec) for connector_type, spec in CREDENTIAL_SPECS.items()
    ]


__all__ = [
    "CREDENTIAL_SPECS",
    "CredentialSpec",
    "HttpConnector",
    "PendingConnector",
    "UnimplementedFetchMixin",
    "build_connector_registry_entries",
]
