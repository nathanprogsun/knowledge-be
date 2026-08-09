"""Chat-pipeline entity-extraction step (upstream ``PluginExtractEntity``).

``ExtractEntityStep`` handles the ``QUERY_UNDERSTAND`` event. When graph
extraction is enabled, it resolves the chat model, collects the knowledge
bases whose extract config is enabled, and asks the model to pull key
entities out of the user's original query. The surviving entity names are
stored on ``PipelineContext.entity``; the enabled knowledge bases and the
knowledge→KB mapping are stored for the later entity-search step.

The step never fails a run: every recoverable failure (model resolution,
KB loading, extraction errors) logs and continues the chain unchanged,
matching the upstream event contract. Dependencies arrive as injected
structural protocols.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

from loguru import logger

from src.ai.embedding.base import Context as RetrievalContext
from src.ai.llm.types import Chat
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import Context, EventType
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.chunk_extract import (
    GraphData,
    GraphNode,
    PromptTemplateStructured,
    StructureExtractor,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo


def env_graph_enabled() -> bool:
    """Read the environment gate that enables graph extraction."""
    return (os.getenv("NEO4J_ENABLE") or "").lower() == "true"


def extract_config_enabled(kb: KnowledgeBaseInfo) -> bool:
    """Report whether ``kb`` enables entity extraction."""
    config = kb.extract_config
    if not isinstance(config, dict):
        return False
    enabled = config.get("enabled")
    return enabled if isinstance(enabled, bool) else False


#: Default entity-extraction prompt template (description + one example).
DEFAULT_EXTRACT_ENTITY_TEMPLATE = PromptTemplateStructured(
    description=(
        "Based on the user's question, process the key information "
        "extraction task following these steps:\n"
        "1. Analyze logical connections: First, fully analyze the text "
        "content, identify its core logical relationships, and briefly "
        "annotate the core logic type;\n"
        "2. Extract key entities: Based on the identified logical "
        "relationships, precisely extract key information from the text and "
        "classify it into clear entities, ensuring no core information is "
        "omitted and no redundant content is added;\n"
        "3. Prioritize entities: Sort by the closeness of each entity's "
        "association with the core topic of the text, presenting the most "
        "important entities for understanding the main idea first;"
    ),
    examples=[
        GraphData(
            text="'Romeo and Juliet' is a tragedy written by William "
            "Shakespeare early in his career, and is one of the most "
            "frequently performed plays in world literature.",
            node=[
                GraphNode(name="Romeo and Juliet"),
                GraphNode(name="William Shakespeare"),
                GraphNode(name="world literature"),
            ],
            relation=[],
        )
    ],
)


# ── Dependency seams ───────────────────────────────────────────────────


@runtime_checkable
class ChatModelProvider(Protocol):
    """Resolves a chat-capable model client by id."""

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat: ...


@runtime_checkable
class KnowledgeBaseLoader(Protocol):
    """Loads knowledge-base records by id (authorization is the caller's job)."""

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]: ...


@runtime_checkable
class KnowledgeLoader(Protocol):
    """Loads knowledge documents for a tenant (shared-access aware)."""

    async def load_documents(self, tenant_id: int, knowledge_ids: list[str]) -> list[Knowledge]: ...


# ── Concrete adapters ──────────────────────────────────────────────────


class KBServiceKnowledgeBaseLoader:
    """``KnowledgeBaseLoader`` adapter over ``KBService``."""

    def __init__(self, service: KBService) -> None:
        self._service = service

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        return await self._service.get_knowledge_bases_by_ids(ids=ids)


class KnowledgeServiceLoader:
    """``KnowledgeLoader`` adapter over ``KnowledgeService``."""

    def __init__(self, service: KnowledgeService) -> None:
        self._service = service

    async def load_documents(self, tenant_id: int, knowledge_ids: list[str]) -> list[Knowledge]:
        return await self._service.get_documents(tenant_id=tenant_id, ids=knowledge_ids)


# ── Step ───────────────────────────────────────────────────────────────


class ExtractEntityStep:
    """Extracts key entities from the user query for the ``QUERY_UNDERSTAND``
    event."""

    def __init__(
        self,
        *,
        model_provider: ChatModelProvider,
        kb_loader: KnowledgeBaseLoader,
        knowledge_loader: KnowledgeLoader,
        template: PromptTemplateStructured | None = None,
        graph_enabled: bool | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._kb_loader = kb_loader
        self._knowledge_loader = knowledge_loader
        self._template = template if template is not None else DEFAULT_EXTRACT_ENTITY_TEMPLATE
        self._graph_enabled = graph_enabled

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.QUERY_UNDERSTAND,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        graph_enabled = (
            self._graph_enabled if self._graph_enabled is not None else env_graph_enabled()
        )
        if not graph_enabled:
            logger.debug("skipping extract entity, graph extraction is disabled")
            return await next()

        query = pipeline_ctx.query

        try:
            chat_model = await self._model_provider.get_chat_model(ctx, pipeline_ctx.chat_model_id)
        except Exception as exc:
            logger.error(
                "Failed to get model, session_id: {}, error: {}",
                pipeline_ctx.session_id,
                exc,
            )
            return await next()

        # Collect the knowledge-base scope, resolving documents to their
        # owning knowledge bases (including shared documents).
        kb_id_set: set[str] = set(pipeline_ctx.knowledge_base_ids)
        knowledge_to_kb: dict[str, str] = {}
        if pipeline_ctx.knowledge_ids:
            try:
                knowledges = await self._knowledge_loader.load_documents(
                    pipeline_ctx.tenant_id, pipeline_ctx.knowledge_ids
                )
            except Exception as exc:
                logger.error("failed to get knowledges: {}", exc)
                return await next()
            for knowledge in knowledges:
                kb_id_set.add(knowledge.knowledge_base_id)
                knowledge_to_kb[knowledge.id] = knowledge.knowledge_base_id

        all_kb_ids = list(kb_id_set)
        try:
            kbs = await self._kb_loader.load_by_ids(all_kb_ids)
        except Exception as exc:
            logger.error("failed to get knowledge bases: {}", exc)
            return await next()

        enabled_kb_set = {kb.id for kb in kbs if extract_config_enabled(kb)}
        if not enabled_kb_set:
            logger.debug("no knowledge base has extract config enabled")
            return await next()

        enabled_kb_ids = list(enabled_kb_set)
        pipeline_ctx.entity_kb_ids = enabled_kb_ids

        entity_knowledge = {
            knowledge_id: kb_id
            for knowledge_id, kb_id in knowledge_to_kb.items()
            if kb_id in enabled_kb_set
        }
        pipeline_ctx.entity_knowledge = entity_knowledge

        template = PromptTemplateStructured(
            description=self._template.description,
            examples=[
                GraphData(text=example.text, node=example.node, relation=example.relation)
                for example in self._template.examples
            ],
        )
        extractor = StructureExtractor(chat_model, template)
        try:
            graph = await extractor.extract(cast("RetrievalContext", ctx), query)
        except Exception as exc:
            logger.error(
                "Failed to extract entities, session_id: {}, error: {}",
                pipeline_ctx.session_id,
                exc,
            )
            return await next()

        nodes = [node.name for node in graph.node]
        logger.debug("extracted node: {}", nodes)
        pipeline_ctx.entity = nodes
        return await next()


__all__ = [
    "DEFAULT_EXTRACT_ENTITY_TEMPLATE",
    "ChatModelProvider",
    "ExtractEntityStep",
    "KBServiceKnowledgeBaseLoader",
    "KnowledgeBaseLoader",
    "KnowledgeLoader",
    "KnowledgeServiceLoader",
    "env_graph_enabled",
    "extract_config_enabled",
]
