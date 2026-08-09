"""Pipeline completion steps.

The four stages that turn merged retrieval results into a model answer:
``DataAnalysisStep`` (tabular SQL), ``IntoChatMessageStep`` (context
template rendering), ``ChatCompletionStep`` (non-streaming completion)
and ``ChatCompletionStreamStep`` (streaming completion). Shared message
preparation and passage helpers live in sibling modules.
"""

from __future__ import annotations

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
    "KNOWLEDGE_QA_PURPOSE",
    "ChatCompletionStep",
    "ChatCompletionStreamStep",
    "DataAnalysisStep",
    "DataAnalysisTool",
    "IntoChatMessageStep",
    "KnowledgeService",
    "ModelService",
    "RenderedContentMessageService",
    "StreamBus",
    "build_document_header",
    "chat_message_to_llm",
    "enrich_content_with_image_info_for_chat",
    "filter_out_table_chunks",
    "first_pipeline_title",
    "get_enriched_passage_for_chat",
    "is_data_file",
    "is_pipeline_web_reference",
    "ordered_pipeline_references",
    "pipeline_search_result_to_llm",
    "prepare_chat_model_for_step",
    "prepare_messages_with_model_context",
    "to_pipeline_chat_response",
    "with_prompt_cache_metadata",
]
