"""Shared base type and small helpers for the web-search provider clients.

Every provider in this package implements :class:`WebSearchClient` and
returns hits as plain ``dict[str, str]`` (keys: ``title``, ``url``,
``snippet``, ``content``, ``source``, ``published_at``). The dispatch
service consumes that shape, so the ``ai`` layer never needs to import a
core DTO.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from src.common.exception import ValidationError
from src.common.json import JsonValue

# A ``WebSearchClient.search`` result is a list of plain hit dicts.
SearchHit = dict[str, str]

# Desktop-chrome user agent; some providers gate requests by it.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
)


class WebSearchClient:
    """Protocol-like base; concrete subclasses override :meth:`search`."""

    provider_type: str

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        """Run a search; return a list of hits (may be empty)."""
        raise NotImplementedError(f"{type(self).__name__} must implement search()")

    def close(self) -> None:
        """Best-effort release of any underlying resources."""


class HttpSearchClient(WebSearchClient):
    """Base for providers backed by a sync ``httpx.Client``.

    Subclasses pass their fully-configured client in (tests inject one
    with a ``httpx.MockTransport``) and issue requests through
    ``self._client``. :meth:`close` releases the pooled connections.
    """

    def __init__(self, *, client: httpx.Client) -> None:
        self._client = client

    def close(self) -> None:
        """Release the pooled ``httpx.Client`` connections."""
        self._client.close()


def require_non_empty_query(query: str) -> None:
    """Reject a blank query before any outbound call is made."""
    if not query or not query.strip():
        raise ValidationError(
            code="web_search_provider.query_required",
            message="query must be a non-empty string",
        )


def coerce_str(value: JsonValue | None) -> str:
    """Coerce a JSON parameter value to a trimmed ``str``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def coerce_extra(value: JsonValue | None) -> dict[str, str]:
    """Coerce ``extra_config`` to a ``dict[str, str]`` (empty when absent)."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if isinstance(v, (str, int, float, bool))}


def parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO-8601 / RFC-3339 timestamp, or ``None`` on failure."""
    if not value or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


__all__ = [
    "DEFAULT_USER_AGENT",
    "HttpSearchClient",
    "SearchHit",
    "WebSearchClient",
    "coerce_extra",
    "coerce_str",
    "parse_iso_datetime",
    "require_non_empty_query",
]
