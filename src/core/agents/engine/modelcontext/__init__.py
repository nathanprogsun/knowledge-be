"""Request-local model context: compact handles for durable identities.

``Registry`` is the only boundary request lifecycles use. The remaining exports
exist for tests and for tool wrappers that need the raw codecs.
"""

from __future__ import annotations

from src.core.agents.engine.modelcontext.handles import HandleTable
from src.core.agents.engine.modelcontext.model_output import ToolResult
from src.core.agents.engine.modelcontext.registry import (
    ARGUMENT_RESOLUTION_PARTIALLY_RESOLVED,
    ARGUMENT_RESOLUTION_RESOLVED,
    ARGUMENT_RESOLUTION_UNCHANGED,
    ARGUMENT_RESOLUTION_UNRESOLVED,
    Registry,
)
from src.core.agents.engine.modelcontext.sources import ChunkReference, SourceRegistry

__all__ = [
    "ARGUMENT_RESOLUTION_PARTIALLY_RESOLVED",
    "ARGUMENT_RESOLUTION_RESOLVED",
    "ARGUMENT_RESOLUTION_UNCHANGED",
    "ARGUMENT_RESOLUTION_UNRESOLVED",
    "ChunkReference",
    "HandleTable",
    "Registry",
    "SourceRegistry",
    "ToolResult",
]
