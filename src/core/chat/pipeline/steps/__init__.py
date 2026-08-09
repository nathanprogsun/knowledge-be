"""Chat pipeline steps: rerank, filter-top-k, and web-page fetch.

Each step implements the ``Plugin`` protocol from the pipeline engine and
is constructed with the seams it needs (a rerank model service, a web-page
fetcher) so callers can wire them into the stage list for a run.
"""

from __future__ import annotations

from src.core.chat.pipeline.steps.filter_topk import (
    FilterTopKPlugin,
    sort_search_results_deterministically,
)
from src.core.chat.pipeline.steps.rerank import (
    RerankModelService,
    RerankPlugin,
    apply_mmr,
    clean_passage_for_rerank,
    composite_score,
    get_enriched_passage,
    rerank_fallback_min_score,
)
from src.core.chat.pipeline.steps.web_fetch import WebFetchPlugin

__all__ = [
    "FilterTopKPlugin",
    "RerankModelService",
    "RerankPlugin",
    "WebFetchPlugin",
    "apply_mmr",
    "clean_passage_for_rerank",
    "composite_score",
    "get_enriched_passage",
    "rerank_fallback_min_score",
    "sort_search_results_deterministically",
]
