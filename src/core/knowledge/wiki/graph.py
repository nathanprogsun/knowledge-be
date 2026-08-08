"""Wiki link-graph analysis exposed as standalone functions.

This module layers the graph slice computation on top of the merged
page service and page repository. It exists so the web layer can compose
the graph endpoint without reaching into the page service internals:
:func:`get_graph` validates the request, loads the full live page list
once, and delegates to the shared pure :func:`compute_graph_subset` core
so every caller stays on the same overview / ego semantics.

Nothing here modifies the merged services; the web layer wires the
repository instance.
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.core.knowledge.wiki.page_service import compute_graph_subset
from src.core.knowledge.wiki.types import (
    WikiGraphData,
    WikiGraphEdge,
    WikiGraphRequest,
)
from src.db.dao.wiki_page_repository import WikiPageRepository

# Stable domain error code shared with the page service.
_ERROR_GRAPH_KB_REQUIRED = "wiki.graph_kb_required"


async def get_graph(*, page_repo: WikiPageRepository, request: WikiGraphRequest) -> WikiGraphData:
    """Return the requested slice of the KB's wiki link graph.

    The full live page list is loaded once and the subgraph is computed
    in memory (mirrors the graph-analysis semantics): the overview mode
    keeps the top ``limit`` pages by link count plus the edges that
    connect surviving nodes; the ego mode returns the undirected BFS
    neighborhood of ``center`` up to ``depth`` hops. ``types`` narrows
    the candidate node set and, in ego mode, the frontier expansion.
    """
    if not request.knowledge_base_id.strip():
        raise ValidationError(
            code=_ERROR_GRAPH_KB_REQUIRED,
            message="wiki graph request requires a knowledge base id",
        )
    pages = await page_repo.list_all(knowledge_base_id=request.knowledge_base_id)
    return compute_graph_subset(pages, request)


__all__ = [
    "WikiGraphData",
    "WikiGraphEdge",
    "WikiGraphRequest",
    "compute_graph_subset",
    "get_graph",
]
