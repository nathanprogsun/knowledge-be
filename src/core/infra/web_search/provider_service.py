"""Web-search-provider service — tenant-scoped CRUD + connectivity test.

Maps the methods declared in the upstream
``internal/application/service/web_search_provider.go::webSearchProviderService``.

The service depends only on its repository. The web layer constructs a
fresh repo + service per request (via ``factory.build_*_service``).

Behaviour parity notes:

- ``CreateProvider`` validates the ``provider`` field against
  ``SUPPORTED_PROVIDER_TYPES`` and validates the ``parameters`` blob
  against the provider type's required fields. The set of required
  fields is mirrored from the Go validator.
- ``UpdateProvider`` enforces ``provider`` immutability post-creation
  (the web layer rejects a body that tries to change it; the service
  guards the same invariant when called directly).
- ``TestProvider`` runs the search through a fresh registry-built client
  with a tiny ``"test"`` query. Empty results surface as a typed error
  so the web layer can render the right diagnostic message.
- The default flag is promoted atomically via
  ``repo.clear_default(tenant_id, exclude_id=new_id)`` before the new
  row is inserted, so a workspace never holds two simultaneous defaults.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import ValidationError
from src.common.json import BindParams, JsonObject
from src.core.contracts.infra import WebSearchProviderParameters
from src.core.infra.web_search.types import (
    SUPPORTED_PROVIDER_TYPES,
    WebSearchProviderInfo,
)
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository
from src.db.models.infra.web_search_provider import WebSearchProvider

_NOT_FOUND_CODE = "web_search_provider.not_found"


class WebSearchClient(Protocol):
    """Minimal interface a registry-built client must satisfy."""

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[dict[str, str]]:
        """Run a test search; return at least one item or fail."""


class WebSearchClientRegistry(Protocol):
    """Minimal interface ``TestProvider`` needs from the provider registry.

    ``params`` is the JSON-shaped parameter blob (``api_key``,
    ``cx`` (Google CSE engine id), ``base_url``, ``proxy_url``,
    ``extra_config``) so the Protocol does not depend on the typed
    ``WebSearchProviderParameters`` contract — that keeps the boundary
    clean for implementations living in a lower layer (e.g. ``src/ai``).
    """

    def create_provider(
        self,
        provider_type: str,
        params: JsonObject,
    ) -> WebSearchClient:
        """Build a client for ``provider_type`` with the given parameters."""


class WebSearchProviderService:
    """CRUD + default-flip + connectivity test for web search providers."""

    def __init__(
        self,
        *,
        provider_repo: WebSearchProviderRepository,
    ) -> None:
        self._repo = provider_repo

    # ── Reads ───────────────────────────────────────────────────────

    async def list_providers(self, tenant_id: int) -> list[WebSearchProviderInfo]:
        """Return every live provider of the tenant, oldest first."""
        rows = await self._repo.list_for_tenant(tenant_id)
        return [WebSearchProviderInfo.map_from_db(r) for r in rows]

    async def get_provider(
        self,
        tenant_id: int,
        provider_id: str,
    ) -> WebSearchProviderInfo:
        """Return one provider by id, or raise ``ValidationError``."""
        row = await self._repo.get_by_id(tenant_id, provider_id)
        if row is None:
            raise ValidationError(
                code=_NOT_FOUND_CODE,
                message=f"web search provider {provider_id} not found",
            )
        return WebSearchProviderInfo.map_from_db(row)

    # ── Mutations ───────────────────────────────────────────────────

    async def create_provider(
        self,
        *,
        tenant_id: int,
        name: str,
        provider: str,
        description: str | None,
        parameters: JsonObject | None,
        is_default: bool,
        provider_id: str,
    ) -> WebSearchProviderInfo:
        """Insert a new provider; promote as default atomically when requested."""
        _require_tenant_id(tenant_id)
        _validate_provider_type(provider)
        params = _validate_provider_parameters(provider, parameters)
        now = datetime.now(UTC)
        row = WebSearchProvider(
            id=provider_id,
            tenant_id=tenant_id,
            name=name,
            provider=provider,
            description=description,
            parameters=_parameters_to_json(params),
            is_default=bool(is_default),
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        if is_default:
            await self._repo.clear_default(tenant_id, exclude_id="")
        persisted = await self._repo.insert(row)
        return WebSearchProviderInfo.map_from_db(persisted)

    async def update_provider(
        self,
        *,
        tenant_id: int,
        provider_id: str,
        name: str | None,
        description: str | None,
        parameters: JsonObject | None,
        is_default: bool | None,
    ) -> WebSearchProviderInfo:
        """Update mutable fields of an existing provider.

        The ``provider`` field is immutable post-creation; the web
        layer never sends it through this endpoint. The repository
        returns ``None`` when the row is missing or soft-deleted; the
        service surfaces that as ``ValidationError``.
        """
        _require_tenant_id(tenant_id)
        existing = await self._repo.get_by_id(tenant_id, provider_id)
        if existing is None:
            raise ValidationError(
                code=_NOT_FOUND_CODE,
                message=f"web search provider {provider_id} not found",
            )
        if name is not None:
            existing = existing.model_copy(update={"name": name})
        if description is not None:
            existing = existing.model_copy(update={"description": description})
        if parameters is not None:
            params = _validate_provider_parameters(existing.provider, parameters)
            existing = existing.model_copy(
                update={
                    "parameters": _parameters_to_json(params),
                    "updated_at": datetime.now(UTC),
                }
            )
        if is_default is not None:
            existing = existing.model_copy(update={"is_default": bool(is_default)})
            if is_default:
                await self._repo.clear_default(tenant_id, exclude_id=provider_id)
        else:
            existing = existing.model_copy(update={"updated_at": datetime.now(UTC)})
        column_to_update: BindParams = {
            "name": existing.name,
            "description": existing.description,
            "parameters": existing.parameters,
            "is_default": existing.is_default,
            "updated_at": datetime.now(UTC),
        }
        persisted = await self._repo.update_by_primary_key(
            {"id": provider_id, "tenant_id": tenant_id},
            column_to_update,
        )
        if persisted is None:
            raise ValidationError(
                code=_NOT_FOUND_CODE,
                message=f"web search provider {provider_id} not found",
            )
        return WebSearchProviderInfo.map_from_db(persisted)

    async def delete_provider(self, tenant_id: int, provider_id: str) -> None:
        """Soft-delete a provider. Raises when nothing matched."""
        _require_tenant_id(tenant_id)
        existing = await self._repo.get_by_id(tenant_id, provider_id)
        if existing is None:
            raise ValidationError(
                code=_NOT_FOUND_CODE,
                message=f"web search provider {provider_id} not found",
            )
        await self._repo.update_by_primary_key(
            {"id": provider_id, "tenant_id": tenant_id},
            BindParams(deleted_at=datetime.now(UTC)),
            exclude_deleted_or_archived=False,
        )

    # ── Connectivity test ───────────────────────────────────────────

    async def test_provider_by_id(
        self,
        tenant_id: int,
        provider_id: str,
        registry: WebSearchClientRegistry,
    ) -> None:
        """Run a one-shot sample search against the saved configuration."""
        info = await self.get_provider(tenant_id, provider_id)
        await _run_test_search(
            registry, info.provider, info.parameters or WebSearchProviderParameters()
        )

    async def test_provider_raw(
        self,
        provider: str,
        parameters: JsonObject,
        registry: WebSearchClientRegistry,
    ) -> None:
        """Run a one-shot sample search against an unsaved configuration."""
        _validate_provider_type(provider)
        params = _validate_provider_parameters(provider, parameters)
        await _run_test_search(registry, provider, params)


# ── Helpers ────────────────────────────────────────────────────────


def _require_tenant_id(tenant_id: int) -> None:
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="web_search_provider.tenant_required",
            message="tenant ID is required",
        )


def _validate_provider_type(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDER_TYPES:
        raise ValidationError(
            code="web_search_provider.invalid_provider_type",
            message=f"invalid provider type: {provider}",
        )


def _validate_provider_parameters(
    provider: str,
    raw: JsonObject | None,
) -> WebSearchProviderParameters:
    """Validate the parameters blob against the provider type's requirements.

    Mirrors the Go validator: Bing/Google/Tavily/Ollama/Baidu/Zhipu
    require an api_key; Google additionally requires cx;
    SearXNG requires a non-empty base_url; the rest are optional.
    """
    params = _parameters_from_json(raw)
    if provider in {"bing", "tavily", "ollama", "baidu", "zhipu"} and not params.api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message=f"API key is required for {provider} provider",
        )
    if provider == "google":
        if not params.api_key:
            raise ValidationError(
                code="web_search_provider.api_key_required",
                message="API key is required for Google provider",
            )
        if not params.cx:
            raise ValidationError(
                code="web_search_provider.cx_required",
                message="cx (Google Custom Search engine ID) is required for Google provider",
            )
    if provider == "searxng" and not (params.base_url and params.base_url.strip()):
        raise ValidationError(
            code="web_search_provider.base_url_required",
            message="base URL is required for SearXNG provider",
        )
    return params


def _parameters_from_json(raw: JsonObject | None) -> WebSearchProviderParameters:
    if raw is None:
        return WebSearchProviderParameters()
    extra = raw.get("extra_config")
    extra_dict: dict[str, str] | None = None
    if isinstance(extra, dict):
        extra_dict = {
            str(k): str(v) for k, v in extra.items() if isinstance(v, (str, int, float, bool))
        }
    api_key = raw.get("api_key")
    cx_raw = raw.get("cx") or raw.get("engine_id") or raw.get("engineId")
    base_url = raw.get("base_url")
    proxy_url = raw.get("proxy_url")
    return WebSearchProviderParameters(
        api_key=str(api_key) if isinstance(api_key, (str, int, float, bool)) else None,
        cx=str(cx_raw) if isinstance(cx_raw, (str, int, float, bool)) else None,
        base_url=str(base_url) if isinstance(base_url, (str, int, float, bool)) else None,
        proxy_url=str(proxy_url) if isinstance(proxy_url, (str, int, float, bool)) else None,
        extra_config=extra_dict,
    )


def _parameters_to_json(params: WebSearchProviderParameters) -> JsonObject:
    """Render the typed DTO as the JSONB blob the row carries."""
    raw: JsonObject = {}
    if params.api_key is not None:
        raw["api_key"] = params.api_key
    if params.cx is not None:
        raw["cx"] = params.cx
    if params.base_url is not None:
        raw["base_url"] = params.base_url
    if params.proxy_url is not None:
        raw["proxy_url"] = params.proxy_url
    if params.extra_config is not None:
        raw["extra_config"] = dict(params.extra_config)
    return raw


async def _run_test_search(
    registry: WebSearchClientRegistry,
    provider: str,
    params: WebSearchProviderParameters,
) -> None:
    """Run a connectivity test search via the registered client."""
    raw = _parameters_to_json(params)
    try:
        client = registry.create_provider(provider, raw)
    except KeyError as exc:
        raise ValidationError(
            code="web_search_provider.unknown_provider_type",
            message=f"web search provider type {provider!r} is not registered",
        ) from exc
    results = client.search("test", 1, False)
    if not results:
        raise ValidationError(
            code="web_search_provider.test_empty_results",
            message=f"{provider} returned 0 results; verify API key and configuration",
        )


__all__ = ["WebSearchClient", "WebSearchClientRegistry", "WebSearchProviderService"]
