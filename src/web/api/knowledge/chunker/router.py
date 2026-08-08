"""Chunker debug endpoints - read-only chunking preview.

Registered by ``RegisterChunkerDebugRoutes``. The single endpoint runs
the adaptive chunker over supplied text without touching the database;
the handler and its response models live in ``views.py``.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.web.api.knowledge.chunker.views import (
    PreviewChunkingEnvelope,
    preview_chunking,
)

router = APIRouter(prefix="/chunker", tags=["chunker"])

# ``response_model_exclude_none`` mirrors the upstream ``omitempty`` tags:
# an empty ``context_header`` and an absent ``truncated_to`` stay off the
# wire instead of serializing as null.
router.add_api_route(
    "/preview",
    preview_chunking,
    methods=["POST"],
    response_model=PreviewChunkingEnvelope,
    response_model_exclude_none=True,
)


__all__ = ["router"]
