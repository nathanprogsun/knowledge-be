"""Unit tests for ``WebSearchSearchService``.

Mirrors ``tests/core/system/test_service.py`` style: Protocol-based
mocks for the repository and a stub client registry. Covers the search
main path (by-id resolution, blacklist filtering, source labelling),
the legacy fallback, and the missing-query / missing-provider
validation errors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject
from src.core.infra.web_search.provider_service import (
    WebSearchClient,
    WebSearchClientRegistry,
)
from src.core.infra.web_search.search_service import WebSearchSearchService
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository
from src.db.models.infra.web_search_provider import WebSearchProvider

_NOT_FOUND_CODE = "web_search_provider.not_found"


# ── Protocol doubles (non-repository collaborators) ──────────────────


class FakeClient:
    """In-memory ``WebSearchClient`` returning canned results."""

    def __init__(self, provider_type: str, results: list[dict[str, str]]) -> None:
        self.provider_type = provider_type
        self._results = results
        self.calls: list[tuple[str, int, bool]] = []

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[dict[str, str]]:
        self.calls.append((query, max_results, include_date))
        return list(self._results)


class FakeRegistry:
    """In-memory ``WebSearchClientRegistry``."""

    def __init__(self) -> None:
        self.clients: dict[str, FakeClient] = {}

    def add(
        self,
        provider_type: str,
        results: list[dict[str, str]],
    ) -> None:
        self.clients[provider_type] = FakeClient(provider_type, results)

    def create_provider(
        self,
        provider_type: str,
        params: JsonObject,
    ) -> WebSearchClient:
        return self.clients[provider_type]


# ── Repository mock ─────────────────────────────────────────────────


def _get_by_id_for(
    rows: dict[tuple[int, str], WebSearchProvider],
):
    """Return a side_effect that resolves ``(tenant_id, provider_id)``."""

    async def _get_by_id(tenant_id: int, provider_id: str) -> WebSearchProvider | None:
        row = rows.get((tenant_id, provider_id))
        if row is None or row.deleted_at is not None:
            return None
        return row

    return _get_by_id


def _make_repo() -> tuple[AsyncMock, dict[tuple[int, str], WebSearchProvider]]:
    """WebSearch-provider repo mock with closure-captured state.

    ``get_by_id`` is driven by a side_effect so a test that pre-loads a
    row into ``rows`` gets that row back, and a missing key yields
    ``None`` (the missing-provider error path).
    """
    repo = AsyncMock(spec=WebSearchProviderRepository)
    rows: dict[tuple[int, str], WebSearchProvider] = {}
    repo.get_by_id.side_effect = _get_by_id_for(rows)
    return repo, rows


def _make_svc() -> tuple[
    WebSearchSearchService, AsyncMock, dict[tuple[int, str], WebSearchProvider], FakeRegistry
]:
    repo, rows = _make_repo()
    reg = FakeRegistry()
    return (
        WebSearchSearchService(
            provider_repo=repo,
            registry=cast("WebSearchClientRegistry", reg),
            timeout_seconds=2,
        ),
        repo,
        rows,
        reg,
    )


def _row(
    tenant_id: int,
    provider_id: str,
    provider: str,
    *,
    api_key: str = "k",
) -> WebSearchProvider:
    now = datetime.now(UTC)
    return WebSearchProvider(
        id=provider_id,
        tenant_id=tenant_id,
        name=provider_id,
        provider=provider,
        description=None,
        parameters={
            "api_key": api_key,
            "engine_id": "e",
            "base_url": "https://example.com",
            "proxy_url": "",
        },
        is_default=False,
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


# ── main path ──────────────────────────────────────────────────────


async def test_search_resolves_by_id_and_returns_results() -> None:
    svc, _, rows, reg = _make_svc()
    rows[(1, "wsp-1")] = _row(tenant_id=1, provider_id="wsp-1", provider="bing")
    reg.add(
        "bing",
        [
            {
                "title": "Hello",
                "url": "https://example.com/hello",
                "snippet": "world",
                "content": "...",
            }
        ],
    )
    results = await svc.search(tenant_id=1, provider_id="wsp-1", query="hi")
    assert len(results) == 1
    assert results[0].title == "Hello"
    assert results[0].url == "https://example.com/hello"
    assert results[0].source == "bing"
    assert reg.clients["bing"].calls == [("hi", 10, False)]


async def test_search_missing_query_raises() -> None:
    svc, _, _, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.search(tenant_id=1, provider_id="wsp-1", query="")
    assert exc.value.code == "web_search_provider.query_required"


async def test_search_missing_provider_id_raises() -> None:
    svc, _, _, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.search(tenant_id=1, provider_id="wsp-missing", query="hi")
    assert exc.value.code == _NOT_FOUND_CODE


async def test_search_no_provider_and_no_legacy_raises() -> None:
    svc, _, _, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.search(tenant_id=1, provider_id="", query="hi")
    assert exc.value.code == "web_search_provider.no_provider_configured"


# ── legacy fallback ────────────────────────────────────────────────


async def test_search_legacy_provider_path() -> None:
    svc, _, _, reg = _make_svc()
    reg.add(
        "duckduckgo",
        [
            {
                "title": "Legacy",
                "url": "https://example.com/legacy",
                "snippet": "...",
            }
        ],
    )
    results = await svc.search(
        tenant_id=1,
        provider_id="",
        query="hi",
        legacy_provider="duckduckgo",
        legacy_api_key="",
    )
    assert len(results) == 1
    assert results[0].source == "duckduckgo"


# ── blacklist filtering ────────────────────────────────────────────


async def test_search_blacklist_filters_matches() -> None:
    svc, _, rows, reg = _make_svc()
    rows[(1, "wsp-1")] = _row(tenant_id=1, provider_id="wsp-1", provider="bing")
    reg.add(
        "bing",
        [
            {"title": "ok", "url": "https://example.com/a", "snippet": ""},
            {"title": "block", "url": "https://blocked.com/b", "snippet": ""},
            {"title": "glob", "url": "https://x.example.com/c", "snippet": ""},
        ],
    )
    results = await svc.search(
        tenant_id=1,
        provider_id="wsp-1",
        query="hi",
        blacklist=["*://blocked.com/*", "*://*.example.com/*"],
    )
    urls = [r.url for r in results]
    assert "https://example.com/a" in urls
    assert "https://blocked.com/b" not in urls
    assert "https://x.example.com/c" not in urls


async def test_search_blacklist_regex_syntax() -> None:
    svc, _, rows, reg = _make_svc()
    rows[(1, "wsp-1")] = _row(tenant_id=1, provider_id="wsp-1", provider="bing")
    reg.add(
        "bing",
        [
            {"title": "ok", "url": "https://example.com/a", "snippet": ""},
            {"title": "bad", "url": "https://foo.org/b", "snippet": ""},
        ],
    )
    results = await svc.search(
        tenant_id=1,
        provider_id="wsp-1",
        query="hi",
        blacklist=["/(example|foo)\\.(com|org)/"],
    )
    assert results == []


# ── timeout / failure ──────────────────────────────────────────────


async def test_search_wraps_provider_failure() -> None:
    svc, _, rows, reg = _make_svc()
    rows[(1, "wsp-1")] = _row(tenant_id=1, provider_id="wsp-1", provider="bing")

    class _RaisingClient(WebSearchClient):
        provider_type = "bing"

        def search(
            self,
            query: str,
            max_results: int,
            include_date: bool,
        ) -> list[dict[str, str]]:
            raise ExternalServiceError(
                code="upstream.unreachable",
                message="upstream down",
            )

    reg.clients["bing"] = cast("FakeClient", _RaisingClient())

    with pytest.raises(ExternalServiceError) as exc:
        await svc.search(tenant_id=1, provider_id="wsp-1", query="hi")
    assert exc.value.code == "web_search_provider.search_failed"


async def test_search_includes_date_flag_is_propagated() -> None:
    svc, _, rows, reg = _make_svc()
    rows[(1, "wsp-1")] = _row(tenant_id=1, provider_id="wsp-1", provider="bing")
    reg.add("bing", [{"title": "x", "url": "https://example.com/a", "snippet": ""}])
    await svc.search(
        tenant_id=1,
        provider_id="wsp-1",
        query="hi",
        include_date=True,
    )
    assert reg.clients["bing"].calls == [("hi", 10, True)]


__all__ = []
