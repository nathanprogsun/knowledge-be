"""Connector protocol for external data sources — layer-neutral half.

Lives in ``common`` rather than ``core`` because concrete connectors are
implemented in ``src/ai/`` and the layer rules forbid ``ai -> core``.
This module therefore holds exactly what both sides must agree on:

- connector type identifiers
- the connector-facing config (credentials **included**)
- the payload value objects a connector produces (``Resource``,
  ``FetchedItem``, ``SyncCursor``, ``SyncResult``, ``SyncItemError``)
- the ``Connector`` abstract base and the optional streaming protocol

Everything domain-shaped — status constants, DB projections, the registry
and its UI metadata — stays in ``src/core/infra/datasources/``.

Field and JSON names mirror ``internal/types/datasource.go`` and
``internal/datasource/connector.go`` exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject

# ── Connector types (internal/types/datasource.go) ───────────────────

CONNECTOR_TYPE_FEISHU = "feishu"
# Feishu's international edition; shares the Feishu connector, only the
# API host and tenant differ.
CONNECTOR_TYPE_LARK = "lark"
CONNECTOR_TYPE_NOTION = "notion"
CONNECTOR_TYPE_CONFLUENCE = "confluence"
CONNECTOR_TYPE_YUQUE = "yuque"
CONNECTOR_TYPE_GITHUB = "github"
CONNECTOR_TYPE_GOOGLE_DRIVE = "google_drive"
CONNECTOR_TYPE_ONEDRIVE = "onedrive"
CONNECTOR_TYPE_DINGTALK = "dingtalk"
CONNECTOR_TYPE_WEB_CRAWLER = "web_crawler"
CONNECTOR_TYPE_SLACK = "slack"
CONNECTOR_TYPE_IMAP = "imap"
CONNECTOR_TYPE_RSS = "rss"

# ``feed_urls`` is non-secret RSS configuration that older rows stored in
# the encrypted credential blob; stripped on write, surfaced through
# ``settings`` on read (Go: ``StripNonSecretCredentials`` /
# ``enrichRSSFeedURLsInSettings``).
RSS_NON_SECRET_CREDENTIAL_KEYS: frozenset[str] = frozenset({"feed_urls"})
RSS_SECRET_CREDENTIAL_KEY = "auth_headers"

# The single logical credential field. Go exposes connector credentials as
# one atomic map rather than per-field, because half-configured connector
# auth cannot authenticate (see ``dto.NewDataSourceResponse``).
CREDENTIALS_FIELD = "credentials"

# Child external-id separator. ``SubtreeChildID`` producers and the stale
# ``SubtreeChildPrefix`` sweep must encode the same separator, so both
# read it from here.
_SUBTREE_SEPARATOR = "#"


def subtree_child_id(parent_external_id: str, kind: str, token: str) -> str:
    """Build a sub-item's external id — ``types.SubtreeChildID``.

    Shape is ``"<parent>#<kind>#<token>"``. ``kind`` is a short
    discriminator (``"file"``, ``"image"``); ``token`` is the source
    system's id for the child and is assumed separator-free.
    """
    return f"{parent_external_id}{_SUBTREE_SEPARATOR}{kind}{_SUBTREE_SEPARATOR}{token}"


def subtree_child_prefix(parent_external_id: str) -> str:
    """External-id prefix matching every child of a parent node.

    ``types.SubtreeChildPrefix``. The stale-child sweep deletes prior
    children with this prefix that are absent from ``subtree_keep``.
    """
    return f"{parent_external_id}{_SUBTREE_SEPARATOR}"


# ── Connector-facing config (credentials INCLUDED) ───────────────────


class DataSourceConfig(BaseModel):
    """Decrypted connector configuration — ``types.DataSourceConfig``.

    Distinct from the contract ``DataSourceConfig`` in
    ``src/core/contracts/infra.py``, which has the credential map removed
    by construction (it is the *response* shape). This one carries
    credentials because a connector cannot authenticate without them; it
    never leaves the service.

    ``multimodal_enabled`` mirrors Go's ``json:"-"`` field: populated per
    sync run from the target knowledge base, never persisted. Connectors
    read it to decide whether extracting embedded images for OCR is worth
    the calls.
    """

    model_config = ConfigDict(frozen=True)

    type: str = ""
    credentials: JsonObject = Field(default_factory=dict)
    resource_ids: list[str] = Field(default_factory=list)
    settings: JsonObject = Field(default_factory=dict)
    multimodal_enabled: bool = False

    def has_credentials(self) -> bool:
        """Whether the credential map carries any value at all."""
        return len(self.credentials) > 0

    def has_configured_credentials(self, connector_type: str) -> bool:
        """Whether user-facing *secret* credentials are stored.

        RSS feed URLs are non-secret configuration, so for that connector
        only ``auth_headers`` counts
        (``DataSourceConfig.HasConfiguredCredentials``).
        """
        if not self.credentials:
            return False
        if connector_type != CONNECTOR_TYPE_RSS:
            return True
        raw = self.credentials.get(RSS_SECRET_CREDENTIAL_KEY)
        return isinstance(raw, str) and raw.strip() != ""

    def strip_non_secret_credentials(self, connector_type: str) -> Self:
        """Return a copy with non-secret values removed from credentials.

        Immutable counterpart of Go's in-place
        ``StripNonSecretCredentials``.
        """
        if connector_type != CONNECTOR_TYPE_RSS or not self.credentials:
            return self
        kept = {
            k: v for k, v in self.credentials.items() if k not in RSS_NON_SECRET_CREDENTIAL_KEYS
        }
        if kept == self.credentials:
            return self
        return self.model_copy(update={"credentials": kept})


# ── Connector payload value objects ──────────────────────────────────


class Resource(BaseModel):
    """A syncable resource (document / folder / space) — ``types.Resource``."""

    model_config = ConfigDict(frozen=True)

    external_id: str
    name: str
    type: str
    description: str = ""
    url: str = ""
    modified_at: datetime | None = None
    parent_id: str = ""
    has_children: bool = False
    metadata: JsonObject = Field(default_factory=dict)


class FetchedItem(BaseModel):
    """One content item fetched from a source — ``types.FetchedItem``.

    ``replaces_subtree`` + ``subtree_keep`` drive the stale-child sweep
    for connectors that fan one source node out into a parent document
    plus attachment/image sub-items. A connector setting
    ``replaces_subtree`` MUST populate ``subtree_keep`` with every child
    still present in the source: an empty keep-set means "keep nothing"
    and sweeps every existing child under the prefix.
    """

    model_config = ConfigDict(frozen=True)

    external_id: str
    title: str = ""
    content: bytes = b""
    content_type: str = ""
    file_name: str = ""
    url: str = ""
    updated_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    is_deleted: bool = False
    source_resource_id: str = ""
    replaces_subtree: bool = False
    subtree_keep: list[str] = Field(default_factory=list)


class SyncCursor(BaseModel):
    """Incremental-sync position — ``types.SyncCursor``."""

    model_config = ConfigDict(frozen=True)

    last_sync_time: datetime | None = None
    connector_cursor: JsonObject = Field(default_factory=dict)
    last_schema_hash: str = ""


class SyncItemError(BaseModel):
    """One user-facing per-item failure sample — ``types.SyncItemError``.

    ``code`` is a stable i18n key the frontend localises; ``message`` is
    the non-localised fallback for clients without the key. Raw upstream
    status/body never lands here — that stays in the server logs.
    """

    model_config = ConfigDict(frozen=True)

    title: str = ""
    code: str = ""
    params: dict[str, str] = Field(default_factory=dict)
    message: str = ""

    def display(self) -> str:
        """Non-localised single-line rendering for logs / error detail."""
        if self.title and self.message:
            return f"{self.title}: {self.message}"
        return self.message or self.title


class SyncResult(BaseModel):
    """Outcome summary of one sync run — ``types.SyncResult``."""

    model_config = ConfigDict(frozen=True)

    total: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[SyncItemError] = Field(default_factory=list)
    next_cursor: SyncCursor | None = None


# ── Connector protocol ───────────────────────────────────────────────


class Connector(ABC):
    """One external system's adapter to the sync pipeline.

    Every method takes the decrypted :class:`DataSourceConfig` rather than
    reading credentials itself, so one instance per type is safe to share
    across tenants — which is what lets the registry be process-wide.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Connector type identifier (e.g. ``"feishu"``)."""

    @abstractmethod
    async def validate(self, config: DataSourceConfig) -> None:
        """Verify credentials + connectivity.

        Raises an ``ApplicationError`` subclass when the configuration is
        unusable; returns ``None`` on success (Go returns ``error``).
        """

    @abstractmethod
    async def list_resources(
        self,
        config: DataSourceConfig,
        parent_id: str = "",
    ) -> list[Resource]:
        """List syncable resources, one hierarchy level at a time.

        ``parent_id == ""`` returns the top level; a non-empty value
        returns only that resource's direct children. Connectors whose
        listing is flat may ignore ``parent_id`` for the root call and
        return an empty list for any non-empty value.
        """

    @abstractmethod
    async def resolve_resource_ancestors(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[str]:
        """Return the ancestor ids a lazy picker must expand.

        Lets a lazily-loaded tree reveal a pre-existing deep selection in
        O(depth) instead of re-walking the whole tree. Connectors that
        return the full tree, or a flat list, have nothing to reveal and
        return an empty list. The result is deduplicated and unordered.
        """

    @abstractmethod
    async def fetch_all(
        self,
        config: DataSourceConfig,
        resource_ids: list[str],
    ) -> list[FetchedItem]:
        """Full sync — return every item under ``resource_ids``."""

    @abstractmethod
    async def fetch_incremental(
        self,
        config: DataSourceConfig,
        cursor: SyncCursor | None,
    ) -> tuple[list[FetchedItem], SyncCursor | None]:
        """Incremental sync — return changed items plus the next cursor."""


@runtime_checkable
class StreamHandler(Protocol):
    """Receives items and cursor checkpoints during a streaming fetch.

    The service implements this to ingest each item as it arrives
    (bounding memory to one item instead of the whole source) and to
    persist the connector cursor at page boundaries, so a sync that times
    out mid-traversal resumes from the last checkpoint.
    """

    async def emit(self, item: FetchedItem) -> None:
        """Ingest a single item. Raising aborts the stream."""
        ...

    async def checkpoint(self, cursor: SyncCursor) -> None:
        """Persist progress so far.

        The cursor MUST be a complete resumable snapshot, not a delta —
        the service treats a checkpoint as a safe restart point.
        """
        ...


@runtime_checkable
class StreamingConnector(Protocol):
    """Optional capability: interleave fetch → ingest → checkpoint.

    Connectors that do not implement it fall back to
    :meth:`Connector.fetch_all` / :meth:`Connector.fetch_incremental`
    unchanged.
    """

    async def fetch_stream(
        self,
        config: DataSourceConfig,
        cursor: SyncCursor | None,
        handler: StreamHandler,
    ) -> SyncCursor | None:
        """Walk the configured resources from ``cursor`` (``None`` = full
        sync), calling ``handler.emit`` per changed item and
        ``handler.checkpoint`` at page boundaries. Returns the final
        cursor for the next sync.
        """
        ...


__all__ = [
    "CONNECTOR_TYPE_CONFLUENCE",
    "CONNECTOR_TYPE_DINGTALK",
    "CONNECTOR_TYPE_FEISHU",
    "CONNECTOR_TYPE_GITHUB",
    "CONNECTOR_TYPE_GOOGLE_DRIVE",
    "CONNECTOR_TYPE_IMAP",
    "CONNECTOR_TYPE_LARK",
    "CONNECTOR_TYPE_NOTION",
    "CONNECTOR_TYPE_ONEDRIVE",
    "CONNECTOR_TYPE_RSS",
    "CONNECTOR_TYPE_SLACK",
    "CONNECTOR_TYPE_WEB_CRAWLER",
    "CONNECTOR_TYPE_YUQUE",
    "CREDENTIALS_FIELD",
    "RSS_NON_SECRET_CREDENTIAL_KEYS",
    "RSS_SECRET_CREDENTIAL_KEY",
    "Connector",
    "DataSourceConfig",
    "FetchedItem",
    "Resource",
    "StreamHandler",
    "StreamingConnector",
    "SyncCursor",
    "SyncItemError",
    "SyncResult",
    "subtree_child_id",
    "subtree_child_prefix",
]
