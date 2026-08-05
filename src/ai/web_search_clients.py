"""Provider HTTP clients for the web-search domain.

Implements the ``WebSearchClientRegistry`` Protocol from
``src.core.infra.web_search.provider_service`` and a ``WebSearchClient``
type that satisfies ``WebSearchClient.search``.

This module lives under ``src/ai`` (not under ``core``) per the layered
architecture rule: HTTP clients depend on third-party SDKs / outbound
URLs, while the service layer that consumes them must not. The core
service treats the registry as a Protocol; concrete implementations
live here and are wired in by the lifespan.

The actual HTTP plumbing lands in a followup checkpoint — the public
surface is stable now (registry + client classes), so the search
service can be tested with fakes. The Go search clients (bing,
google, tavily, …) live in ``internal/infrastructure/web_search/*`` in
the upstream repo and port to this module one provider per followup.

The registry's public API uses only types from ``src.common.json`` (a
shared layer, not ``core``) so this module does not violate the
``ai`` <-> ``core`` boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

# Provider-type IDs are passed in as plain strings; the registry does
# NOT import the typed ``SUPPORTED_PROVIDER_TYPES`` frozenset from
# ``core.infra.web_search.types`` (ai must not import core). The
# registry accepts any non-empty string and only rejects unknown
# provider types when its ``create_provider`` consumer supplies a
# protocol-enforced list. Tests wire the actual set via
# ``register(known_type, ...)`` from a shared list declared elsewhere.

WebSearchClientFactory = Callable[[JsonObject], "WebSearchClient"]
"""Factory signature: ``(params: JsonObject) -> WebSearchClient``."""


class WebSearchClient:
    """Protocol-like base; concrete subclasses override :meth:`search`."""

    provider_type: str

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[dict[str, str]]:
        """Run a search; return a list of hits (may be empty)."""
        raise ExternalServiceError(
            code="web_search_provider.client_unimplemented",
            message=f"client for {self.provider_type!r} is not wired yet",
        )


@dataclass(slots=True)
class _NotImplementedClient(WebSearchClient):
    """Stand-in client that surfaces a clear error.

    The real HTTP clients (bing, google, …) land in followup PRs.
    Tests inject their own client via :meth:`WebSearchClientRegistry.register`.
    """

    def __init__(self, provider_type: str) -> None:
        self.provider_type = provider_type

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[dict[str, str]]:
        raise ExternalServiceError(
            code="web_search_provider.client_unimplemented",
            message=(
                f"client for {self.provider_type!r} is not wired yet; "
                "use a registry you populated via .register() for tests"
            ),
        )


@dataclass(slots=True)
class WebSearchClientRegistry:
    """Map ``provider_type`` → factory that builds a :class:`WebSearchClient`.

    Mirrors ``internal/infrastructure/web_search/registry.go::Registry``.
    """

    _factories: dict[str, WebSearchClientFactory] = field(default_factory=dict)

    def register(self, provider_type: str, factory: WebSearchClientFactory) -> None:
        """Register a factory for ``provider_type``. Idempotent overwrite."""
        if not provider_type:
            raise ValidationError(
                code="web_search_provider.unknown_provider_type",
                message="provider type must be a non-empty string",
            )
        self._factories[provider_type] = factory

    def is_registered(self, provider_type: str) -> bool:
        """True iff ``provider_type`` has a factory bound."""
        return provider_type in self._factories

    def create_provider(
        self,
        provider_type: str,
        params: JsonObject,
    ) -> WebSearchClient:
        """Build a client for ``provider_type`` with the given parameters."""
        if not provider_type:
            raise ValidationError(
                code="web_search_provider.unknown_provider_type",
                message="provider type must be a non-empty string",
            )
        factory = self._factories.get(provider_type)
        if factory is None:
            return _NotImplementedClient(provider_type=provider_type)
        return factory(params)


__all__ = [
    "WebSearchClient",
    "WebSearchClientFactory",
    "WebSearchClientRegistry",
]
