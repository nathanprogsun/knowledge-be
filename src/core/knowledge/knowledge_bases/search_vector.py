"""Vector retrieval parameter construction.

One ``RetrieveParams`` is built per vector index: document KBs hit the
default index (empty knowledge type), FAQ KBs hit the FAQ index
(``knowledge_type="faq"``). The orchestrator partitions each store
group's KBs by type before calling :func:`build_vector_params`, so a
group mixing both types yields one param per index and every KB is
queried against the index it was written to.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.ai.retrieval.types import RetrieveParams, RetrieverType
from src.core.knowledge.knowledge_bases.search_filter import scope_retrieve_params

#: Index-knowledge-type tag used by FAQ vector retrieval.
_FAQ_KNOWLEDGE_TYPE = "faq"


def build_vector_params(
    *,
    query: str,
    embedding: Sequence[float],
    doc_kb_ids: Sequence[str],
    faq_kb_ids: Sequence[str],
    top_k: int,
    threshold: float,
    knowledge_ids: Sequence[str] = (),
    tag_ids: Sequence[str] = (),
) -> list[RetrieveParams]:
    """Build one vector ``RetrieveParams`` per populated vector index.

    Empty KB lists yield no params for that index. ``knowledge_ids`` /
    ``tag_ids`` narrow the index query via the shared scope filter.
    """
    params: list[RetrieveParams] = []
    if doc_kb_ids:
        params.append(
            _vector_params(
                query=query,
                embedding=embedding,
                kb_ids=list(doc_kb_ids),
                knowledge_type="",
                top_k=top_k,
                threshold=threshold,
                knowledge_ids=knowledge_ids,
                tag_ids=tag_ids,
            )
        )
    if faq_kb_ids:
        params.append(
            _vector_params(
                query=query,
                embedding=embedding,
                kb_ids=list(faq_kb_ids),
                knowledge_type=_FAQ_KNOWLEDGE_TYPE,
                top_k=top_k,
                threshold=threshold,
                knowledge_ids=knowledge_ids,
                tag_ids=tag_ids,
            )
        )
    return params


def _vector_params(
    *,
    query: str,
    embedding: Sequence[float],
    kb_ids: list[str],
    knowledge_type: str,
    top_k: int,
    threshold: float,
    knowledge_ids: Sequence[str],
    tag_ids: Sequence[str],
) -> RetrieveParams:
    base = RetrieveParams(
        query=query,
        embedding=list(embedding),
        knowledge_base_ids=kb_ids,
        top_k=top_k,
        threshold=threshold,
        knowledge_type=knowledge_type,
        retriever_type=RetrieverType.VECTOR,
    )
    return scope_retrieve_params(
        base,
        knowledge_ids=knowledge_ids,
        tag_ids=tag_ids,
    )


__all__ = ["build_vector_params"]
