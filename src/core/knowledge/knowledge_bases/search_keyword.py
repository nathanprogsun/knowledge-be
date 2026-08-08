"""Keyword retrieval parameter construction.

Document-type KBs are the only keyword-retrieval participants; FAQ KBs
are retrieved exclusively via the FAQ vector index and have no keyword
index. The orchestrator partitions group KBs by type before calling
:func:`build_keyword_params`.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.ai.retrieval.types import RetrieveParams, RetrieverType
from src.core.knowledge.knowledge_bases.search_filter import scope_retrieve_params


def build_keyword_params(
    *,
    query: str,
    kb_ids: Sequence[str],
    top_k: int,
    threshold: float,
    knowledge_ids: Sequence[str] = (),
    tag_ids: Sequence[str] = (),
) -> RetrieveParams:
    """Build the keyword ``RetrieveParams`` for one store group.

    ``knowledge_ids`` / ``tag_ids`` narrow the index query via the shared
    scope filter.
    """
    base = RetrieveParams(
        query=query,
        knowledge_base_ids=list(kb_ids),
        top_k=top_k,
        threshold=threshold,
        retriever_type=RetrieverType.KEYWORDS,
    )
    return scope_retrieve_params(
        base,
        knowledge_ids=knowledge_ids,
        tag_ids=tag_ids,
    )


__all__ = ["build_keyword_params"]
