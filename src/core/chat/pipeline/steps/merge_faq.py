"""FAQ answer enrichment for the chunk-merge step.

FAQ chunks carry their canonical Q&A in chunk metadata; this step replaces
the retrieved snippet with the formatted answer text so the model sees the
authoritative answer rather than a partial match.
"""

from __future__ import annotations

from src.common.json import JsonObject
from src.core.chat.pipeline.common import pipeline_info, pipeline_warn
from src.core.chat.pipeline.steps.merge_utils import ChunkSource
from src.core.chat.pipeline.types import Context, SearchResult
from src.core.knowledge.chunks.types import CHUNK_TYPE_FAQ


def build_faq_answer_content(meta: JsonObject | None) -> str:
    """Format a FAQ chunk's metadata into an answer block.

    The question becomes a ``Q:`` line and the answers a ``Answer:`` list;
    an empty block is returned when neither is present.
    """
    if not meta:
        return ""
    question_value = meta.get("standard_question")
    question = question_value.strip() if isinstance(question_value, str) else ""
    answers: list[str] = []
    raw_answers = meta.get("answers")
    if isinstance(raw_answers, list):
        for answer in raw_answers:
            if isinstance(answer, str):
                trimmed = answer.strip()
                if trimmed:
                    answers.append(trimmed)
    if not question and not answers:
        return ""
    lines: list[str] = []
    if question:
        lines.append(f"Q: {question}")
    if answers:
        lines.append("Answer:")
        lines.extend(f"- {answer}" for answer in answers)
    return "\n".join(lines).strip()


async def populate_faq_answers(
    ctx: Context,
    chunk_repo: ChunkSource | None,
    tenant_id: int,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Replace FAQ hit content with the formatted answer from chunk metadata."""
    if not results or chunk_repo is None:
        return results
    if tenant_id == 0:
        pipeline_warn("Merge", "faq_enrich_skip", {"reason": "missing_tenant"})
        return results

    chunk_result_ids: dict[str, list[int]] = {}
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for index, result in enumerate(results):
        if not result.id or result.chunk_type != CHUNK_TYPE_FAQ:
            continue
        chunk_result_ids.setdefault(result.id, []).append(index)
        if result.id not in seen:
            seen.add(result.id)
            chunk_ids.append(result.id)
    if not chunk_ids:
        return results

    try:
        chunks = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, chunk_ids)
    except Exception as err:
        pipeline_warn("Merge", "faq_chunk_fetch_failed", {"error": str(err)})
        return results

    answer_by_chunk_id: dict[str, str] = {}
    for chunk in chunks:
        content = build_faq_answer_content(chunk.metadata)
        if content:
            answer_by_chunk_id[chunk.id] = content
    if not answer_by_chunk_id:
        return results

    out = list(results)
    enriched = 0
    for chunk_id, indexes in chunk_result_ids.items():
        answer = answer_by_chunk_id.get(chunk_id)
        if answer is None:
            continue
        for index in indexes:
            out[index] = out[index].model_copy(update={"content": answer})
            enriched += 1
    pipeline_info("Merge", "faq_content_enriched", {"chunk_cnt": enriched})
    return out


__all__ = ["build_faq_answer_content", "populate_faq_answers"]
