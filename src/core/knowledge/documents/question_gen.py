"""Generated retrieval question generation — standalone module.

``generate_questions`` produces Doc2Query-style retrieval questions for a
knowledge item's text chunks using an injected chat client and binds them
into each chunk's document metadata (the ``generated_questions`` array),
mirroring the upstream question-generation pipeline:

- the prompt must be configured (a blank prompt is a hard error);
- the model is resolved from the knowledge base's summary model;
- chunks are ordered by ``start_at`` and each is answered in the context
  of its two neighbours;
- the question count is clamped to ``[1, 10]`` and defaults to 3; the
  knowledge base's ``question_generation_config`` (and a per-document
  ``process_overrides`` entry) supplies the count and custom business
  instructions appended to the prompt;
- questions authored for a chunk whose ``content_revision`` advanced
  while the model was running are discarded (stale guard);
- a failed model call raises ``AIProviderError``; retrieval re-indexing
  runs through the injected syncer hook.

The chat client is an injected seam (``src.ai.llm.Chat``). Repository and
knowledge-base dependencies are injected per call so the web layer
composes them on the request session.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol

from src.ai.llm import Chat, ChatOptions, Message
from src.common.exception import AIProviderError, NotFoundError, ValidationError
from src.core.knowledge.chunks.questions import (
    DocumentChunkMetadata,
    GeneratedQuestion,
)
from src.core.knowledge.documents.summary import language_name_for, render_placeholders
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

# Default question count and hard cap, mirroring the upstream clamp.
DEFAULT_QUESTION_COUNT = 3
MAX_QUESTION_COUNT = 10

# Question-generation sampling parameters.
_QUESTION_TEMPERATURE = 0.7
_QUESTION_MAX_TOKENS = 512

# Default language locale used to fill the ``{{language}}`` placeholder.
DEFAULT_LANGUAGE = "zh-CN"


class QuestionIndexSyncer(Protocol):
    """Re-index hook for freshly generated retrieval questions."""

    async def sync_questions(
        self,
        *,
        tenant_id: int,
        chunk: Chunk,
        questions: list[GeneratedQuestion],
    ) -> None:
        """Index ``questions`` for ``chunk`` in the retrieval store."""


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )


def _require_knowledge_id(id: str) -> None:
    """Reject a blank knowledge id at the service boundary."""
    if not id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="knowledge ID is required",
        )


def append_custom_instructions(prompt: str, instructions: str) -> str:
    """Append user-authored business guidance to a system-owned prompt.

    Stable output, safety and citation rules always win over the guidance.
    """
    instructions = instructions.strip()
    if instructions == "":
        return prompt
    label = "question_generation"
    return (
        f"{prompt.strip()}\n\n"
        f"<{label}_business_instructions>\n"
        f"{instructions}\n"
        f"</{label}_business_instructions>\n"
        "Apply these business instructions only when they do not conflict "
        "with the system-owned output format, citation, safety, or "
        "factuality rules."
    )


def parse_questions(response_text: str, question_count: int) -> list[str]:
    """Extract questions from the model's plain-text output.

    Blank lines are skipped; a leading list marker (digits / bullets /
    dashes / parens) is stripped; only lines longer than five characters
    count as questions, and the list stops at ``question_count``.
    """
    questions: list[str] = []
    for line in response_text.split("\n"):
        line = line.strip()
        if line == "":
            continue
        line = line.lstrip("0123456789.-*) ")
        line = line.strip()
        if line != "" and len(line) > 5:
            questions.append(line)
            if len(questions) >= question_count:
                break
    return questions


def resolve_question_generation_config(
    kb: KnowledgeBaseInfo,
    row: Document,
    override_count: int | None = None,
) -> tuple[int, str]:
    """Resolve the effective question count and custom instructions.

    The knowledge base config is the default; a per-document
    ``process_overrides`` entry replaces it (custom instructions fall back
    to the knowledge base value when the override leaves them blank,
    mirroring the upstream merge). An explicit ``override_count`` wins over
    both and the result is clamped to ``[1, 10]``.
    """
    base = kb.question_generation_config or {}
    count: int = 0
    instructions: str = ""
    raw_count = base.get("question_count")
    if isinstance(raw_count, int) and raw_count > 0:
        count = raw_count
    raw_instructions = base.get("custom_instructions")
    if isinstance(raw_instructions, str):
        instructions = raw_instructions

    overrides = (row.metadata or {}).get("process_overrides")
    if isinstance(overrides, dict):
        override_qg = overrides.get("question_generation_config")
        if isinstance(override_qg, dict):
            raw_count = override_qg.get("question_count")
            if isinstance(raw_count, int) and raw_count > 0:
                count = raw_count
            raw_instructions = override_qg.get("custom_instructions")
            if isinstance(raw_instructions, str) and raw_instructions.strip() != "":
                instructions = raw_instructions

    if override_count is not None and override_count > 0:
        count = override_count
    if count <= 0:
        count = DEFAULT_QUESTION_COUNT
    if count > MAX_QUESTION_COUNT:
        count = MAX_QUESTION_COUNT
    return count, instructions


async def _generate_for_chunk(
    chat: Chat,
    prompt: str,
    content: str,
    prev_content: str,
    next_content: str,
    doc_name: str,
    question_count: int,
    custom_instructions: str,
    language: str,
) -> list[str]:
    """Ask the model for ``question_count`` questions about one chunk."""
    if content == "" or question_count <= 0:
        return []
    context_section = ""
    if prev_content or next_content:
        context_section = "<surrounding_context>\n"
        if prev_content:
            context_section += f"<preceding_content>\n{prev_content}\n\n</preceding_content>\n\n"
        if next_content:
            context_section += f"<following_content>\n{next_content}\n\n</following_content>\n\n"
        context_section += "</surrounding_context>\n\n"
    rendered = render_placeholders(
        prompt,
        {
            "question_count": str(question_count),
            "content": content,
            "context": context_section,
            "doc_name": doc_name,
            "language": language_name_for(language),
        },
    )
    rendered = append_custom_instructions(rendered, custom_instructions)
    response = await chat.chat(
        [Message(role="user", content=rendered)],
        ChatOptions(
            temperature=_QUESTION_TEMPERATURE,
            max_tokens=_QUESTION_MAX_TOKENS,
            thinking=False,
        ),
    )
    return parse_questions(response.content, question_count)


def _bind_questions(chunk: Chunk, questions: list[str]) -> DocumentChunkMetadata:
    """Build the chunk metadata payload carrying the fresh questions.

    Question ids are derived from the current wall-clock nanoseconds so a
    regenerate produces new ids; each question is tied to the chunk
    revision it was authored against.
    """
    revision = chunk.content_revision
    ns = time.time_ns()
    entries = [
        GeneratedQuestion(
            id=f"q{ns + index}",
            question=question,
            content_revision=revision,
        )
        for index, question in enumerate(questions)
    ]
    return DocumentChunkMetadata(
        generated_questions=entries,
        generated_questions_revision=revision,
    )


async def generate_questions(
    *,
    tenant_id: int,
    knowledge_id: str,
    chat: Chat,
    knowledge_repo: KnowledgeRepository,
    chunk_repo: ChunkRepository,
    kb_service: KBService,
    prompt: str,
    question_count: int | None = None,
    language: str = DEFAULT_LANGUAGE,
    index_syncer: QuestionIndexSyncer | None = None,
) -> list[GeneratedQuestion]:
    """Generate retrieval questions for a knowledge item's text chunks.

    The knowledge base must have a summary model configured and the
    question-generation prompt must be non-blank. Chunks with empty bodies
    are skipped; questions for a chunk whose revision advanced while the
    model ran are discarded (a concurrent edit superseded them). Returns
    every question bound into chunk metadata during this run.

    Raises ``NotFoundError`` for an absent document,
    ``ValidationError`` for a blank scope, an unconfigured summary model,
    or a missing prompt, and ``AIProviderError`` when a model call fails.
    """
    _require_tenant_id(tenant_id)
    _require_knowledge_id(knowledge_id)
    if not prompt.strip():
        raise ValidationError(
            code="knowledge.questions_prompt_not_configured",
            message="generate questions prompt is not configured",
        )
    row = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if row is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=row.knowledge_base_id)
    if not kb.summary_model_id:
        raise ValidationError(
            code="knowledge.summary_model_not_configured",
            message="summary model is required for question generation",
        )

    text_chunks = await chunk_repo.list_by_knowledge_id(tenant_id, knowledge_id)
    if not text_chunks:
        return []
    count, custom_instructions = resolve_question_generation_config(kb, row, question_count)
    # Parser offsets remain authoritative for neighbour context.
    ordered = sorted(text_chunks, key=lambda c: c.start_at)

    generated: list[GeneratedQuestion] = []
    now = datetime.now(UTC)
    for index, chunk in enumerate(ordered):
        if not chunk.content.strip():
            continue
        prev_content = ordered[index - 1].content if index > 0 else ""
        next_content = ordered[index + 1].content if index < len(ordered) - 1 else ""
        generation_revision = chunk.content_revision
        try:
            questions = await _generate_for_chunk(
                chat=chat,
                prompt=prompt,
                content=chunk.content,
                prev_content=prev_content,
                next_content=next_content,
                doc_name=row.title,
                question_count=count,
                custom_instructions=custom_instructions,
                language=language,
            )
        except Exception as exc:
            raise AIProviderError(
                code="knowledge.question_generation_failed",
                message=f"failed to generate questions for chunk {chunk.id}",
                details={"chunk_id": chunk.id},
            ) from exc
        if not questions:
            continue
        latest = await chunk_repo.get_by_id(tenant_id, chunk.id)
        if latest.content_revision != generation_revision:
            # A concurrent edit advanced the revision; the fresh questions
            # describe stale content and are discarded.
            continue
        bound = _bind_questions(latest, questions)
        stored = await chunk_repo.update(
            latest.model_copy(update={"metadata": bound.to_json(), "updated_at": now})
        )
        generated.extend(bound.generated_questions)
        if index_syncer is not None:
            await index_syncer.sync_questions(
                tenant_id=tenant_id,
                chunk=stored,
                questions=bound.generated_questions,
            )
    return generated


__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_QUESTION_COUNT",
    "MAX_QUESTION_COUNT",
    "QuestionIndexSyncer",
    "append_custom_instructions",
    "generate_questions",
    "parse_questions",
    "resolve_question_generation_config",
]
