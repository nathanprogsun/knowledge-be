"""Initialization-domain FastAPI dependency factory.

One-line forwarder to ``src.core.infra.initialization.factory``, per
AGENTS.md §1: collaborators (Ollama REST client, outbound probe client,
download-task store) are assembled in ``core``; ``web`` never imports
``db`` and never constructs them itself.

This domain is repository-free — the probes are pure outbound HTTP — so
the forwarder takes no ``SessionDep``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.initialization.factory import build_initialization_service
from src.core.infra.initialization.service.initialization_service import InitializationService


def get_initialization_service() -> InitializationService:
    """Build a per-request ``InitializationService`` over shared clients."""
    return build_initialization_service()


InitializationServiceDep = Annotated[InitializationService, Depends(get_initialization_service)]


__all__ = [
    "InitializationServiceDep",
    "get_initialization_service",
]
