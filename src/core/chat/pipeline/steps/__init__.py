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
from src.core.chat.pipeline.steps.chat_completion import (
    ChatCompletionStep,
    to_pipeline_chat_response,
)
from src.core.chat.pipeline.steps.data_analysis import (
    DataAnalysisStep,
    DataAnalysisTool,
    KnowledgeService,
    filter_out_table_chunks,
    is_data_file,
)
from src.core.chat.pipeline.steps.into_chat_message import (
    IntoChatMessageStep,
    RenderedContentMessageService,
)
from src.core.chat.pipeline.steps.model_context import (
    KNOWLEDGE_QA_PURPOSE,
    ModelService,
    chat_message_to_llm,
    first_pipeline_title,
    is_pipeline_web_reference,
    ordered_pipeline_references,
    pipeline_search_result_to_llm,
    prepare_chat_model_for_step,
    prepare_messages_with_model_context,
    with_prompt_cache_metadata,
)
from src.core.chat.pipeline.steps.passage import (
    CHUNK_TYPE_FAQ,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
    CHUNK_TYPE_WEB_SEARCH,
    build_document_header,
    enrich_content_with_image_info_for_chat,
    get_enriched_passage_for_chat,
)
from src.core.chat.pipeline.steps.stream import (
    ChatCompletionStreamStep,
    StreamBus,
)

__all__ = [
    "CHUNK_TYPE_FAQ",
    "CHUNK_TYPE_TABLE_COLUMN",
    "CHUNK_TYPE_TABLE_SUMMARY",
    "CHUNK_TYPE_WEB_SEARCH",
    "ChatCompletionStep",
    "ChatCompletionStreamStep",
    "DataAnalysisStep",
    "DataAnalysisTool",
    "ExtractEntityStep",
    "FilterTopKPlugin",
    "IntoChatMessageStep",
    "KNOWLEDGE_QA_PURPOSE",
    "KnowledgeService",
    "ModelService",
    "QueryUnderstandPlugin",
    "RenderedContentMessageService",
    "RerankModelService",
    "RerankPlugin",
    "SearchEntityPlugin",
    "SearchParallelPlugin",
    "SearchStep",
    "StreamBus",
    "WebFetchPlugin",
    "apply_mmr",
    "build_document_header",
    "chat_message_to_llm",
    "clean_passage_for_rerank",
    "composite_score",
    "enrich_content_with_image_info_for_chat",
    "filter_out_table_chunks",
    "first_pipeline_title",
    "get_enriched_passage",
    "get_enriched_passage_for_chat",
    "is_data_file",
    "is_pipeline_web_reference",
    "ordered_pipeline_references",
    "parse_structured_query_output",
    "pipeline_search_result_to_llm",
    "prepare_chat_model_for_step",
    "prepare_messages_with_model_context",
    "rerank_fallback_min_score",
    "sort_search_results_deterministically",
    "to_pipeline_chat_response",
    "with_prompt_cache_metadata",
]
