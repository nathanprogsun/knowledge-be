"""Unit tests for :mod:`src.common.datasource_protocol`.

Covers the connector type constants, the ``DataSourceConfig`` credential
helpers (including the RSS secret / non-secret split), the subtree-id
builders, ``SyncItemError.display``, and protocol conformance for the
``Connector`` ABC and the runtime-checkable streaming protocols.
"""

from __future__ import annotations

import pytest

from src.common.datasource_protocol import (
    CONNECTOR_TYPE_FEISHU,
    CONNECTOR_TYPE_RSS,
    CREDENTIALS_FIELD,
    Connector,
    DataSourceConfig,
    Resource,
    StreamHandler,
    StreamingConnector,
    SyncItemError,
    SyncResult,
    subtree_child_id,
    subtree_child_prefix,
)

# ── Constants + helpers ─────────────────────────────────────────────


def test_connector_type_constants() -> None:
    assert CONNECTOR_TYPE_FEISHU == "feishu"
    assert CONNECTOR_TYPE_RSS == "rss"


def test_credentials_field_constant() -> None:
    assert CREDENTIALS_FIELD == "credentials"


def test_subtree_child_id_format() -> None:
    assert subtree_child_id("parent-1", "file", "tok-9") == "parent-1#file#tok-9"


def test_subtree_child_prefix() -> None:
    assert subtree_child_prefix("parent-1") == "parent-1#"


# ── DataSourceConfig ───────────────────────────────────────────────


def test_data_source_config_defaults() -> None:
    cfg = DataSourceConfig()
    assert cfg.type == ""
    assert cfg.credentials == {}
    assert cfg.resource_ids == []
    assert cfg.settings == {}
    assert cfg.multimodal_enabled is False
    assert cfg.has_credentials() is False


def test_data_source_config_has_credentials() -> None:
    cfg = DataSourceConfig(credentials={"token": "t"})
    assert cfg.has_credentials() is True


def test_has_configured_credentials_false_when_empty() -> None:
    assert DataSourceConfig().has_configured_credentials(CONNECTOR_TYPE_FEISHU) is False


def test_has_configured_credentials_true_for_non_rss() -> None:
    cfg = DataSourceConfig(credentials={"token": "t"})
    assert cfg.has_configured_credentials(CONNECTOR_TYPE_FEISHU) is True


def test_has_configured_credentials_rss_requires_auth_headers() -> None:
    feed_only = DataSourceConfig(
        credentials={"feed_urls": ["http://example.com/feed"]},
    )
    assert feed_only.has_configured_credentials(CONNECTOR_TYPE_RSS) is False

    with_auth = DataSourceConfig(credentials={"auth_headers": "Bearer x"})
    assert with_auth.has_configured_credentials(CONNECTOR_TYPE_RSS) is True


def test_strip_non_secret_credentials_rss_removes_feed_urls() -> None:
    original = DataSourceConfig(
        credentials={
            "feed_urls": ["http://example.com/feed"],
            "auth_headers": "Bearer x",
        },
    )
    stripped = original.strip_non_secret_credentials(CONNECTOR_TYPE_RSS)

    assert "feed_urls" not in stripped.credentials
    assert stripped.credentials["auth_headers"] == "Bearer x"
    # Immutability: the original config is untouched.
    assert "feed_urls" in original.credentials


def test_strip_non_secret_credentials_non_rss_is_noop() -> None:
    original = DataSourceConfig(
        type=CONNECTOR_TYPE_FEISHU,
        credentials={"token": "t"},
    )
    assert original.strip_non_secret_credentials(CONNECTOR_TYPE_FEISHU) is original


def test_strip_non_secret_credentials_no_credentials_is_noop() -> None:
    cfg = DataSourceConfig()
    assert cfg.strip_non_secret_credentials(CONNECTOR_TYPE_RSS) is cfg


# ── Value objects ──────────────────────────────────────────────────


def test_resource_defaults() -> None:
    r = Resource(external_id="x", name="n", type="doc")
    assert r.description == ""
    assert r.url == ""
    assert r.parent_id == ""
    assert r.has_children is False
    assert r.metadata == {}


def test_sync_result_defaults() -> None:
    result = SyncResult()
    assert result.total == 0
    assert result.errors == []
    assert result.next_cursor is None


@pytest.mark.parametrize(
    ("title", "message", "expected"),
    [
        ("Title", "Message", "Title: Message"),
        ("", "Message", "Message"),
        ("Title", "", "Title"),
        ("", "", ""),
    ],
)
def test_sync_item_error_display(title: str, message: str, expected: str) -> None:
    assert SyncItemError(title=title, message=message).display() == expected


# ── Connector ABC ──────────────────────────────────────────────────


class _DummyConnector(Connector):
    @property
    def type(self) -> str:
        return "dummy"

    async def validate(self, config: DataSourceConfig) -> None:
        return None

    async def list_resources(self, config: DataSourceConfig, parent_id: str = "") -> list[Resource]:
        return []

    async def resolve_resource_ancestors(
        self, config: DataSourceConfig, resource_ids: list[str]
    ) -> list[str]:
        return []

    async def fetch_all(self, config: DataSourceConfig, resource_ids: list[str]) -> list:
        return []

    async def fetch_incremental(
        self, config: DataSourceConfig, cursor: object
    ) -> tuple[list, object]:
        return [], None


def test_connector_is_abstract() -> None:
    with pytest.raises(TypeError):
        Connector()  # type: ignore[abstract]


def test_dummy_connector_is_a_connector() -> None:
    assert isinstance(_DummyConnector(), Connector)


# ── Runtime-checkable streaming protocols ──────────────────────────


class _Streamer:
    async def fetch_stream(
        self, config: DataSourceConfig, cursor: object, handler: StreamHandler
    ) -> object:
        return None


class _Handler:
    async def emit(self, item: object) -> None:
        return None

    async def checkpoint(self, cursor: object) -> None:
        return None


def test_streaming_connector_protocol_recognises_implementer() -> None:
    assert isinstance(_Streamer(), StreamingConnector)


def test_streaming_connector_protocol_rejects_plain_object() -> None:
    assert not isinstance(object(), StreamingConnector)


def test_stream_handler_protocol_recognises_implementer() -> None:
    assert isinstance(_Handler(), StreamHandler)


def test_stream_handler_protocol_rejects_plain_object() -> None:
    assert not isinstance(object(), StreamHandler)
