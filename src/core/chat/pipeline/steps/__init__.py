"""Pipeline step implementations.

Each module ports one pipeline stage onto the merged ``Plugin`` protocol
from ``engine.py``. Steps consume the shared ``PipelineContext`` carrier
(``context.py``) and are wired into an ``EventManager`` at the
composition root; the step modules stay standalone so a run can be
assembled from whichever stages the request mode selects.
"""

from __future__ import annotations

from src.core.chat.pipeline.steps.query_understand import (
    QueryUnderstandPlugin,
    parse_structured_query_output,
)
from src.core.chat.pipeline.steps.search_entity import SearchEntityPlugin
from src.core.chat.pipeline.steps.search_parallel import SearchParallelPlugin
from src.core.chat.pipeline.steps.extract_entity import ExtractEntityStep
from src.core.chat.pipeline.steps.search import SearchStep
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
    "ExtractEntityStep",
    "FilterTopKPlugin",
    "QueryUnderstandPlugin",
    "RerankModelService",
    "RerankPlugin",
    "SearchEntityPlugin",
    "SearchParallelPlugin",
    "SearchStep",
    "WebFetchPlugin",
    "apply_mmr",
    "clean_passage_for_rerank",
    "composite_score",
    "get_enriched_passage",
    "parse_structured_query_output",
    "rerank_fallback_min_score",
    "sort_search_results_deterministically",
]
