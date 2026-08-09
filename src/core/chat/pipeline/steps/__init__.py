"""Auxiliary chat-pipeline steps: references, progress, history, wiki boost.

Standalone steps that plug into the pipeline engine (``EventManager``) or
provide shared helpers for it. Plugins implement the engine ``Plugin``
protocol; the pure functions stay dependency-free so downstream steps and
tests can reuse them.
"""

from __future__ import annotations

from src.core.chat.pipeline.steps.load_history import LoadHistoryPlugin
from src.core.chat.pipeline.steps.progress import (
    QUERY_UNDERSTAND_PROGRESS_TOOL,
    RETRIEVAL_PROGRESS_TOOL,
    RETRIEVAL_SOURCE_KNOWLEDGE,
    RETRIEVAL_SOURCE_MIXED,
    RETRIEVAL_SOURCE_WEB,
    ProgressEventBus,
    StageProgress,
    begin_query_understand_progress,
    begin_retrieval_progress,
    end_query_understand_progress,
    end_retrieval_progress,
    is_consolidated_retrieval_stage,
    last_consolidated_retrieval_stage,
    should_close_retrieval_progress,
    should_emit_query_understand_progress,
)
from src.core.chat.pipeline.steps.references import (
    enrich_content_with_image_info_for_chat,
    first_pipeline_title,
    get_enriched_passage_for_chat,
    is_pipeline_web_reference,
    ordered_pipeline_references,
    prepare_messages_with_model_context,
)
from src.core.chat.pipeline.steps.wiki_boost import (
    WIKI_BOOST_FACTOR,
    KnowledgeBaseService,
    WikiBoostPlugin,
)

__all__ = [
    "QUERY_UNDERSTAND_PROGRESS_TOOL",
    "RETRIEVAL_PROGRESS_TOOL",
    "RETRIEVAL_SOURCE_KNOWLEDGE",
    "RETRIEVAL_SOURCE_MIXED",
    "RETRIEVAL_SOURCE_WEB",
    "WIKI_BOOST_FACTOR",
    "KnowledgeBaseService",
    "LoadHistoryPlugin",
    "ProgressEventBus",
    "StageProgress",
    "WikiBoostPlugin",
    "begin_query_understand_progress",
    "begin_retrieval_progress",
    "end_query_understand_progress",
    "end_retrieval_progress",
    "enrich_content_with_image_info_for_chat",
    "first_pipeline_title",
    "get_enriched_passage_for_chat",
    "is_consolidated_retrieval_stage",
    "is_pipeline_web_reference",
    "last_consolidated_retrieval_stage",
    "ordered_pipeline_references",
    "prepare_messages_with_model_context",
    "should_close_retrieval_progress",
    "should_emit_query_understand_progress",
]
