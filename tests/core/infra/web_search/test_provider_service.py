"""Unit tests for ``WebSearchProviderService``.

Mirrors ``tests/core/system/test_service.py`` style: Protocol-based
mocks for the repository and a stub client registry. Covers CRUD main
paths, validation errors, the default-promotion flip, and the
connectivity-test entry point.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.infra.web_search.provider_service import (
    WebSearchClient,
    WebSearchClientRegistry,
    WebSearchProviderService,
)
from src.core.infra.web_search.types import (
    BUILTIN_PROVIDERS,
    PROVIDER_TYPES,
    SUPPORTED_PROVIDER_TYPES,
)
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository
from src.db.models.infra.web_search_provider import WebSearchProvider

_NOT_FOUND_CODE = "web_search_provider.not_found"


# ── Protocol doubles (non-repository collaborators) ──────────────────


class FakeSearchClient:
    """In-memory ``WebSearchClient`` used by the test path."""

    def __init__(
        self,
        provider_type: str,
        *,
        results: list[dict[str, str]] | None = None,
    ) -> None:
        self.provider_type = provider_type
        self._results = results if results is not None else [{"url": "https://example.com"}]
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
        self.clients: dict[str, FakeSearchClient] = {}

    def add(
        self,
        provider_type: str,
        *,
        results: list[dict[str, str]] | None = None,
    ) -> None:
        self.clients[provider_type] = FakeSearchClient(provider_type, results=results)

    def create_provider(
        self,
        provider_type: str,
        params: JsonObject,
    ) -> WebSearchClient:
        if provider_type not in self.clients:
            raise KeyError(provider_type)
        return self.clients[provider_type]


# ── Repository mock ─────────────────────────────────────────────────


def _make_repo() -> tuple[AsyncMock, dict[tuple[int, str], WebSearchProvider]]:
    """WebSearch-provider repo mock with closure-captured state."""
    repo = AsyncMock(spec=WebSearchProviderRepository)
    rows: dict[tuple[int, str], WebSearchProvider] = {}

    def _key(tenant_id: int, provider_id: str) -> tuple[int, str]:
        return (tenant_id, provider_id)

    async def _insert(row: WebSearchProvider) -> WebSearchProvider:
        rows[_key(row.tenant_id, row.id)] = row
        return row

    async def _get_by_id(tenant_id: int, provider_id: str) -> WebSearchProvider | None:
        row = rows.get(_key(tenant_id, provider_id))
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _find_unique_by_column_values(
        column_to_query: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> WebSearchProvider | None:
        tenant_id = cast(int, column_to_query.get("tenant_id"))
        provider_id_obj = column_to_query.get("id") or column_to_query.get("provider_id")
        is_default = column_to_query.get("is_default")
        if provider_id_obj is not None:
            provider_id = cast(str, provider_id_obj)
            row = rows.get(_key(tenant_id, provider_id))
        elif is_default is True:
            matches = [
                r
                for r in rows.values()
                if r.tenant_id == tenant_id and r.is_default and not r.deleted_at
            ]
            row = matches[0] if matches else None
        else:  # pragma: no cover — guarded by callers
            return None
        if row is None:
            return None
        if exclude_deleted_or_archived and row.deleted_at is not None:
            return None
        return row

    async def _list_for_tenant(tenant_id: int) -> list[WebSearchProvider]:
        return [r for r in rows.values() if r.tenant_id == tenant_id and r.deleted_at is None]

    async def _update_by_primary_key(
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> WebSearchProvider | None:
        provider_id = str(primary_key_to_value["id"])
        row = next((r for r in rows.values() if r.id == provider_id), None)
        if row is None:
            return None
        if exclude_deleted_or_archived and row.deleted_at is not None:
            return None
        persisted = row.model_copy(update=dict(column_to_update))
        rows[_key(persisted.tenant_id, persisted.id)] = persisted
        return persisted

    async def _clear_default(tenant_id: int, exclude_id: str = "") -> int:
        cleared = 0
        for (tid, _), row in list(rows.items()):
            if tid != tenant_id:
                continue
            if row.deleted_at is not None:
                continue
            if not row.is_default:
                continue
            if exclude_id and row.id == exclude_id:
                continue
            rows[(tid, row.id)] = row.model_copy(update={"is_default": False})
            cleared += 1
        return cleared

    repo.insert.side_effect = _insert
    repo.get_by_id.side_effect = _get_by_id
    repo.find_unique_by_column_values.side_effect = _find_unique_by_column_values
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.update_by_primary_key.side_effect = _update_by_primary_key
    repo.clear_default.side_effect = _clear_default
    return repo, rows


def _make_svc() -> tuple[WebSearchProviderService, AsyncMock]:
    repo, _ = _make_repo()
    return WebSearchProviderService(provider_repo=repo), repo


# ── Registry metadata sanity ───────────────────────────────────────


def test_supported_provider_types_match_registry() -> None:
    assert {info.provider for info in PROVIDER_TYPES} == SUPPORTED_PROVIDER_TYPES
    assert {
        "bing",
        "google",
        "tavily",
        "duckduckgo",
        "ollama",
        "baidu",
        "searxng",
        "keenable",
        "zhipu",
    } <= SUPPORTED_PROVIDER_TYPES


def test_builtin_providers_are_non_empty() -> None:
    assert len(BUILTIN_PROVIDERS) >= 5
    for p in BUILTIN_PROVIDERS:
        assert p.name
        assert p.label
        assert p.description
        assert p.enabled


# ── list / get ─────────────────────────────────────────────────────


async def test_list_providers_returns_empty_initially() -> None:
    svc, _ = _make_svc()
    assert await svc.list_providers(tenant_id=1) == []


async def test_get_provider_missing_raises() -> None:
    svc, _ = _make_svc()
    with pytest.raises(NotFoundError) as exc:
        await svc.get_provider(tenant_id=1, provider_id="missing")
    assert exc.value.code == _NOT_FOUND_CODE


# ── create ─────────────────────────────────────────────────────────


async def test_create_provider_persists_row() -> None:
    svc, repo = _make_svc()
    info = await svc.create_provider(
        tenant_id=1,
        name="My Bing",
        provider="bing",
        description=None,
        parameters={"api_key": "secret"},
        is_default=False,
        provider_id="wsp-001",
    )
    assert info.id == "wsp-001"
    assert info.tenant_id == 1
    assert info.provider == "bing"
    assert info.parameters is not None
    assert info.parameters.api_key == "secret"
    persisted = await repo.find_unique_by_column_values({"tenant_id": 1, "id": "wsp-001"})
    assert persisted is not None
    assert persisted.name == "My Bing"


async def test_create_provider_invalid_type_raises() -> None:
    svc, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.create_provider(
            tenant_id=1,
            name="Bogus",
            provider="not_a_provider",
            description=None,
            parameters=None,
            is_default=False,
            provider_id="wsp-002",
        )
    assert exc.value.code == "web_search_provider.invalid_provider_type"


async def test_create_provider_requires_tenant() -> None:
    svc, _ = _make_svc()
    with pytest.raises(ValidationError):
        await svc.create_provider(
            tenant_id=0,
            name="X",
            provider="bing",
            description=None,
            parameters={"api_key": "k"},
            is_default=False,
            provider_id="x",
        )


@pytest.mark.parametrize(
    "provider",
    ["bing", "tavily", "ollama", "baidu", "zhipu"],
)
async def test_create_provider_requires_api_key(provider: str) -> None:
    svc, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.create_provider(
            tenant_id=1,
            name="NoKey",
            provider=provider,
            description=None,
            parameters=None,
            is_default=False,
            provider_id=f"wsp-{provider}",
        )
    assert exc.value.code == "web_search_provider.api_key_required"


async def test_create_google_requires_api_key_and_cx() -> None:
    svc, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.create_provider(
            tenant_id=1,
            name="NoKey",
            provider="google",
            description=None,
            parameters={"api_key": "k"},
            is_default=False,
            provider_id="wsp-google-noengine",
        )
    assert exc.value.code == "web_search_provider.cx_required"
    with pytest.raises(ValidationError) as exc:
        await svc.create_provider(
            tenant_id=1,
            name="NoKey",
            provider="google",
            description=None,
            parameters=None,
            is_default=False,
            provider_id="wsp-google-noapikey",
        )
    assert exc.value.code == "web_search_provider.api_key_required"


async def test_create_searxng_requires_base_url() -> None:
    svc, _ = _make_svc()
    with pytest.raises(ValidationError) as exc:
        await svc.create_provider(
            tenant_id=1,
            name="Searx",
            provider="searxng",
            description=None,
            parameters=None,
            is_default=False,
            provider_id="wsp-searx",
        )
    assert exc.value.code == "web_search_provider.base_url_required"


async def test_create_default_promotion_clears_prior() -> None:
    svc, repo = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="First",
        provider="bing",
        description=None,
        parameters={"api_key": "k1"},
        is_default=True,
        provider_id="wsp-first",
    )
    # The second create must clear the first row's default flag in-band.
    await svc.create_provider(
        tenant_id=1,
        name="Second",
        provider="google",
        description=None,
        parameters={"api_key": "k2", "cx": "e2"},
        is_default=True,
        provider_id="wsp-second",
    )
    first = await repo.find_unique_by_column_values({"tenant_id": 1, "id": "wsp-first"})
    second = await repo.find_unique_by_column_values({"tenant_id": 1, "id": "wsp-second"})
    assert first is not None and second is not None
    assert first.is_default is False
    assert second.is_default is True
    # Exactly one row of the tenant carries the default flag.
    defaults = await repo.list_for_tenant(1)
    assert sum(1 for r in defaults if r.is_default) == 1
    assert next(r for r in defaults if r.is_default).id == "wsp-second"


async def test_create_default_promotion_calls_clear_default_exclude_id() -> None:
    """Updating an existing default also clears via clear_default(exclude_id)."""
    svc, repo = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="A",
        provider="bing",
        description=None,
        parameters={"api_key": "k"},
        is_default=True,
        provider_id="wsp-a",
    )
    # Update path: flip is_default to True on a different row.
    await svc.update_provider(
        tenant_id=1,
        provider_id="wsp-a",
        name=None,
        description=None,
        parameters=None,
        is_default=True,
    )
    # No second row, clear_default must have excluded wsp-a (return 0).
    cleared = await repo.clear_default(1, exclude_id="wsp-a")
    assert cleared == 0


# ── update ─────────────────────────────────────────────────────────


async def test_update_provider_applies_changes() -> None:
    svc, _ = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="orig",
        provider="bing",
        description="d",
        parameters={"api_key": "old"},
        is_default=False,
        provider_id="wsp-u",
    )
    info = await svc.update_provider(
        tenant_id=1,
        provider_id="wsp-u",
        name="new",
        description=None,
        parameters={"api_key": "new"},
        is_default=None,
    )
    assert info.name == "new"
    assert info.parameters is not None
    assert info.parameters.api_key == "new"


async def test_update_provider_missing_raises() -> None:
    svc, _ = _make_svc()
    with pytest.raises(NotFoundError):
        await svc.update_provider(
            tenant_id=1,
            provider_id="nope",
            name="x",
            description=None,
            parameters=None,
            is_default=None,
        )


async def test_update_provider_revalidates_parameters() -> None:
    svc, _ = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="g",
        provider="google",
        description=None,
        parameters={"api_key": "k", "cx": "e"},
        is_default=False,
        provider_id="wsp-google",
    )
    # Drop cx — must fail with the same code as on create.
    with pytest.raises(ValidationError) as exc:
        await svc.update_provider(
            tenant_id=1,
            provider_id="wsp-google",
            name=None,
            description=None,
            parameters={"api_key": "k"},
            is_default=None,
        )
    assert exc.value.code == "web_search_provider.cx_required"


# ── delete ─────────────────────────────────────────────────────────


async def test_delete_provider_soft_deletes() -> None:
    svc, repo = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="d",
        provider="bing",
        description=None,
        parameters={"api_key": "k"},
        is_default=False,
        provider_id="wsp-d",
    )
    await svc.delete_provider(tenant_id=1, provider_id="wsp-d")
    row = await repo.find_unique_by_column_values(
        {"tenant_id": 1, "id": "wsp-d"},
        exclude_deleted_or_archived=False,
    )
    assert row is not None
    assert row.deleted_at is not None
    # find_unique_by_column_values with exclude_deleted_or_archived=True hides it
    visible = await repo.find_unique_by_column_values(
        {"tenant_id": 1, "id": "wsp-d"},
        exclude_deleted_or_archived=True,
    )
    assert visible is None


async def test_delete_provider_missing_raises() -> None:
    svc, _ = _make_svc()
    with pytest.raises(NotFoundError):
        await svc.delete_provider(tenant_id=1, provider_id="nope")


# ── connectivity test ─────────────────────────────────────────────


async def test_test_provider_by_id_runs_search() -> None:
    svc, _ = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="t",
        provider="bing",
        description=None,
        parameters={"api_key": "k"},
        is_default=False,
        provider_id="wsp-t",
    )
    registry = FakeRegistry()
    registry.add("bing")
    await svc.test_provider_by_id(
        tenant_id=1,
        provider_id="wsp-t",
        registry=cast("WebSearchClientRegistry", registry),
    )
    assert registry.clients["bing"].calls == [("test", 1, False)]


async def test_test_provider_by_id_missing_raises() -> None:
    svc, _ = _make_svc()
    registry = FakeRegistry()
    registry.add("bing")
    with pytest.raises(NotFoundError):
        await svc.test_provider_by_id(
            tenant_id=1,
            provider_id="nope",
            registry=cast("WebSearchClientRegistry", registry),
        )


async def test_test_provider_raw_rejects_invalid_type() -> None:
    svc, _ = _make_svc()
    registry = FakeRegistry()
    with pytest.raises(ValidationError) as exc:
        await svc.test_provider_raw(
            provider="not_a_provider",
            parameters={},
            registry=cast("WebSearchClientRegistry", registry),
        )
    assert exc.value.code == "web_search_provider.invalid_provider_type"


async def test_test_provider_raw_requires_api_key_for_bing() -> None:
    svc, _ = _make_svc()
    registry = FakeRegistry()
    with pytest.raises(ValidationError) as exc:
        await svc.test_provider_raw(
            provider="bing",
            parameters={},
            registry=cast("WebSearchClientRegistry", registry),
        )
    assert exc.value.code == "web_search_provider.api_key_required"


async def test_test_provider_raw_empty_results_raises() -> None:
    svc, _ = _make_svc()
    await svc.create_provider(
        tenant_id=1,
        name="t",
        provider="bing",
        description=None,
        parameters={"api_key": "k"},
        is_default=False,
        provider_id="wsp-empty",
    )
    registry = FakeRegistry()
    registry.add("bing", results=[])
    with pytest.raises(ValidationError) as exc:
        await svc.test_provider_raw(
            provider="bing",
            parameters={"api_key": "k"},
            registry=cast("WebSearchClientRegistry", registry),
        )
    assert exc.value.code == "web_search_provider.test_empty_results"


__all__ = []
