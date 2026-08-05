"""Connector registry + UI metadata — ``internal/datasource/connector.go``.

The ``Connector`` abstract base and the payload value objects live in
``src/common/datasource_protocol.py`` (concrete connectors live in
``src/ai/``, and ``ai`` may not import ``core``); they are re-exported
here so the domain has one import surface.

This module owns what is domain-side:

- ``ConnectorRegistry`` — type → connector lookup, injected into the
  service so it never talks to an external API directly
- ``ConnectorMetadata`` + ``CONNECTOR_METADATA_REGISTRY`` — the UI-facing
  descriptors for the 13 upstream connector types
- ``list_available_connectors()`` — the descriptors sorted by
  ``priority`` (lower first), backing ``GET /datasources/types``
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
    StreamHandler,
    StreamingConnector,
)
from src.common.exception import NotFoundError, ValidationError


class ConnectorMetadata(BaseModel):
    """UI descriptor for one connector type — ``ConnectorMetadata``.

    ``priority`` orders the picker (lower = shown first). ``auth_type`` is
    one of ``oauth2`` / ``api_key`` / ``token`` / ``password`` / ``none``
    / ``custom``. ``capabilities`` names optional behaviours
    (``incremental``, ``deletion_sync``, ``webhook``).
    """

    model_config = ConfigDict(frozen=True)

    type: str
    name: str
    description: str = ""
    icon: str = ""
    priority: int
    auth_type: str = ""
    capabilities: list[str] = Field(default_factory=list)


class ConnectorRegistry:
    """Type → connector lookup — ``datasource.ConnectorRegistry``.

    Built once per process and injected into the service. Registration is
    an explicit call rather than an import side effect, so the set of live
    connectors is grep-able from one place.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}

    def register(self, connector: Connector) -> None:
        """Add ``connector`` under its own ``type``.

        Raises ``ValidationError`` for a blank type
        (``ErrConnectorTypeEmpty``). Re-registering a type replaces the
        earlier entry, matching Go's map assignment.
        """
        if not connector.type:
            raise ValidationError(
                code="datasource.connector_type_empty",
                message="connector type is empty",
            )
        self._connectors[connector.type] = connector

    def get(self, connector_type: str) -> Connector:
        """Return the connector for ``connector_type``.

        Raises ``NotFoundError`` when unregistered
        (``ErrConnectorNotFound``).
        """
        connector = self._connectors.get(connector_type)
        if connector is None:
            raise NotFoundError(
                code="datasource.connector_not_found",
                message=f"connector type {connector_type!r} not found in registry",
            )
        return connector

    def list_types(self) -> list[str]:
        """Return every registered connector type."""
        return list(self._connectors)


# ── Built-in connector metadata (13 upstream types) ──────────────────

CONNECTOR_METADATA_REGISTRY: dict[str, ConnectorMetadata] = {
    CONNECTOR_TYPE_FEISHU: ConnectorMetadata(
        type=CONNECTOR_TYPE_FEISHU,
        name="Feishu (飞书)",
        description="Sync documents, wikis, and content from Feishu",
        priority=0,
        auth_type="oauth2",
        capabilities=["incremental", "deletion_sync"],
    ),
    CONNECTOR_TYPE_LARK: ConnectorMetadata(
        type=CONNECTOR_TYPE_LARK,
        name="Lark",
        description="Sync documents, wikis, and content from Lark (Feishu international)",
        priority=0,
        auth_type="oauth2",
        capabilities=["incremental", "deletion_sync"],
    ),
    CONNECTOR_TYPE_NOTION: ConnectorMetadata(
        type=CONNECTOR_TYPE_NOTION,
        name="Notion",
        description="Sync pages and databases from Notion",
        priority=1,
        auth_type="api_key",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_CONFLUENCE: ConnectorMetadata(
        type=CONNECTOR_TYPE_CONFLUENCE,
        name="Confluence",
        description="Sync spaces and pages from Atlassian Confluence",
        priority=2,
        auth_type="api_key",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_YUQUE: ConnectorMetadata(
        type=CONNECTOR_TYPE_YUQUE,
        name="Yuque (语雀)",
        description="Sync knowledge bases and documents from Yuque",
        priority=3,
        auth_type="api_key",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_GITHUB: ConnectorMetadata(
        type=CONNECTOR_TYPE_GITHUB,
        name="GitHub",
        description="Sync repositories, wikis, and issues from GitHub",
        priority=4,
        auth_type="oauth2",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_GOOGLE_DRIVE: ConnectorMetadata(
        type=CONNECTOR_TYPE_GOOGLE_DRIVE,
        name="Google Drive",
        description="Sync documents and files from Google Drive",
        priority=5,
        auth_type="oauth2",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_ONEDRIVE: ConnectorMetadata(
        type=CONNECTOR_TYPE_ONEDRIVE,
        name="OneDrive / SharePoint",
        description="Sync documents and files from Microsoft OneDrive",
        priority=6,
        auth_type="oauth2",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_DINGTALK: ConnectorMetadata(
        type=CONNECTOR_TYPE_DINGTALK,
        name="DingTalk (钉钉)",
        description="Sync documents and content from DingTalk",
        priority=7,
        auth_type="api_key",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_WEB_CRAWLER: ConnectorMetadata(
        type=CONNECTOR_TYPE_WEB_CRAWLER,
        name="Web Crawler (Sitemap)",
        description="Crawl websites via Sitemap.xml",
        priority=9,
        auth_type="none",
        capabilities=[],
    ),
    CONNECTOR_TYPE_SLACK: ConnectorMetadata(
        type=CONNECTOR_TYPE_SLACK,
        name="Slack",
        description="Sync channel messages and files from Slack",
        priority=10,
        auth_type="oauth2",
        capabilities=["incremental"],
    ),
    CONNECTOR_TYPE_IMAP: ConnectorMetadata(
        type=CONNECTOR_TYPE_IMAP,
        name="Email (IMAP)",
        description="Sync email content from IMAP servers",
        priority=11,
        auth_type="password",
        capabilities=[],
    ),
    CONNECTOR_TYPE_RSS: ConnectorMetadata(
        type=CONNECTOR_TYPE_RSS,
        name="RSS / Atom Feed",
        description="Sync articles from RSS/Atom feeds",
        priority=12,
        auth_type="custom",
        capabilities=["incremental"],
    ),
}


def list_available_connectors() -> list[ConnectorMetadata]:
    """Return every connector descriptor sorted by ``priority``.

    Ties keep registry insertion order (Python's sort is stable), which
    matches the intent of Go's insertion sort over the metadata map.
    """
    return sorted(CONNECTOR_METADATA_REGISTRY.values(), key=lambda m: m.priority)


__all__ = [
    "CONNECTOR_METADATA_REGISTRY",
    "Connector",
    "ConnectorMetadata",
    "ConnectorRegistry",
    "StreamHandler",
    "StreamingConnector",
    "list_available_connectors",
]
