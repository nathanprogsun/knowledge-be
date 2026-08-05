"""Initialization-domain request-scoped service factory.

Follows ``src.core.system.factory``: the service is assembled in ``core``
so ``web/deps`` stays a one-line forwarder and ``web`` never imports
``db``.

This domain has no repositories — the probes are pure outbound HTTP —
so the factory takes no ``AsyncSession``. It does own the two collaborators
that must not be rebuilt per call:

- ``DownloadTaskStore``: process-wide, because pull progress outlives the
  request that started the pull.
- ``httpx.AsyncClient`` / ``OllamaClient``: pooled connections. These are
  APP-scope by nature; registering them on ``LifeSpanService`` is
  deferred (the DI registry is off-limits here), so they are memoized
  with the same ``lru_cache`` pattern ``get_settings()`` uses — no
  module-level mutable globals.
"""

from __future__ import annotations

from functools import lru_cache

import httpx

from src.core.infra.initialization.provider_detect import (
    OllamaClient,
    get_download_task_store,
)
from src.core.infra.initialization.service.initialization_service import InitializationService

_PROBE_TIMEOUT_SECONDS = 30.0


@lru_cache(maxsize=1)
def get_probe_http_client() -> httpx.AsyncClient:
    """Shared outbound client for model-provider probes."""
    return httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS)


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    """Shared Ollama REST client (base URL read from ``OLLAMA_BASE_URL``)."""
    return OllamaClient()


def build_initialization_service(
    *,
    ollama_client: OllamaClient | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> InitializationService:
    """Per-request ``InitializationService`` over shared HTTP clients.

    Both collaborators are injectable so tests can supply
    ``httpx.MockTransport``-backed clients without touching the network.
    """
    return InitializationService(
        ollama_client=ollama_client if ollama_client is not None else get_ollama_client(),
        task_store=get_download_task_store(),
        http_client=http_client if http_client is not None else get_probe_http_client(),
    )


__all__ = [
    "build_initialization_service",
    "get_ollama_client",
    "get_probe_http_client",
]
