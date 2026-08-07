"""Concrete web-search client registry (provider type -> factory).

This is the concrete implementation of the ``WebSearchClientRegistry``
Protocol declared in ``src.core.infra.web_search.provider_service``. It
lives in the ``ai`` layer because the registered factories build
outbound HTTP clients; the service layer consumes it through the
Protocol.

Provider-type ids are plain strings; the registry accepts any non-empty
string and leaves the supported set to the caller (the typed catalog in
``src.core.infra.web_search.types``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.ai.web_search_clients._base import WebSearchClient
from src.common.exception import ValidationError
from src.common.json import JsonObject

WebSearchClientFactory = Callable[[JsonObject], WebSearchClient]
"""Factory signature: ``(params: JsonObject) -> WebSearchClient``."""


@dataclass(slots=True)
class WebSearchClientRegistry:
    """Map ``provider_type`` -> factory that builds a :class:`WebSearchClient`."""

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
        """Build a client for ``provider_type`` with the given parameters.

        An unregistered provider type raises ``ValidationError`` (the
        registry mirrors the upstream registry, which fails construction
        for an unknown type instead of returning a silent stand-in).
        """
        if not provider_type:
            raise ValidationError(
                code="web_search_provider.unknown_provider_type",
                message="provider type must be a non-empty string",
            )
        factory = self._factories.get(provider_type)
        if factory is None:
            raise ValidationError(
                code="web_search_provider.unknown_provider_type",
                message=f"web search provider type {provider_type!r} is not registered",
            )
        return factory(params)


__all__ = [
    "WebSearchClientFactory",
    "WebSearchClientRegistry",
]
