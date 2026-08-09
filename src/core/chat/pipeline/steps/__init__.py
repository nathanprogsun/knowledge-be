"""Chunk-merge pipeline steps.

Stages implementing the ``CHUNK_MERGE`` pipeline event: the merge
orchestrator (``merge``), parent-child resolution support, sequential
body merging, FAQ answer enrichment, short-context expansion, and
history-reference injection. Shared pure helpers live in ``merge_utils``.
"""

from __future__ import annotations

__all__: list[str] = []
