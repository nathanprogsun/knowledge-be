"""Pipeline step: rerank retrieved chunks with a dedicated rerank model.

Ports the upstream rerank plugin semantics: build enriched passages
(content plus image captions/OCR and generated questions), call the rerank
model — retrying with a degraded threshold when the first pass returns
nothing above the configured one — combine the model score with the
retrieval base score, optionally boost FAQ chunks, and finally reduce
redundancy with maximal marginal relevance before publishing
``rerank_result``.

The model resolver is injected as a structural seam
(:class:`RerankModelService`) so the step stays testable without a live
backend.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from src.ai.rerank.base import Reranker
from src.ai.rerank.remote_api import RankResult
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.text_utils import jaccard, tokenize_simple
from src.core.chat.pipeline.common import pipeline_error, pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_GET_RERANK_MODEL,
    ERR_SEARCH_NOTHING,
    Next,
    PluginError,
)
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    SearchResult,
    SearchTarget,
)

#: Maximal marginal relevance lambda (relevance vs. redundancy trade-off).
_MMR_LAMBDA = 0.7
#: Relevance a top candidate must carry to survive threshold filtering.
_DEFAULT_FALLBACK_MIN_SCORE = 0.15
#: A rerank threshold above this value is degraded when the first model
#: pass returns nothing (degraded = original * 0.7, floored here).
_DEGRADE_THRESHOLD = 0.3


class RerankModelService(Protocol):
    """Resolves a rerank model instance for a tenant by model id."""

    async def get_rerank_model(self, *, tenant_id: int, model_id: str) -> Reranker: ...


# ── Passage cleaning ───────────────────────────────────────────────────
#
# Rerank models work on semantic text similarity. Markdown formatting, raw
# URLs, image references, table separators and other structural syntax are
# noise that can dilute the semantic signal; the patterns below strip that
# noise before passages are sent to the rerank model.

#: ``![alt](url)`` — the whole construct is noise; the URL group supports
#: one level of balanced parentheses.
_RE_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)")
#: ``[![alt](img_url)](link_url)`` — unwrap to ``![alt](img_url)`` so the
#: image pass can remove the whole construct.
_RE_LINKED_IMAGE = re.compile(
    r"\[!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)\]"
    r"\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)"
)
#: ``[text](url)`` — keep the display text, drop the URL.
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)")
#: Standalone http(s) URLs.
_RE_RAW_URL = re.compile(r"https?://[^\s)\]>]+")
#: Fenced code blocks (````` ``` ... ``` `````).
_RE_CODE_BLOCK = re.compile(r"(?s)```(?:\w*)\n?.*?```")
#: Block-level LaTeX (``$$...$$``).
_RE_LATEX_BLOCK = re.compile(r"(?s)\$\$.*?\$\$")
#: Table separator rows like ``|---|---|``. Uses ``[ \t]`` so newlines are
#: never consumed across rows.
_RE_TABLE_SEP = re.compile(r"(?m)^[ \t]*\|[ \t:|-]+\|[ \t]*$")
#: Markdown table data rows like ``| col1 | col2 |``.
_RE_TABLE_ROW = re.compile(r"(?m)^[ \t]*\|(.+?)\|[ \t]*$")
#: Leading ``#`` markers in headings.
_RE_HEADING_PREFIX = re.compile(r"(?m)^#{1,6}\s+")
#: Leading ``>`` blockquote markers.
_RE_BLOCKQUOTE = re.compile(r"(?m)^>\s?")
#: ``***text***`` wrappers (must run before the 2- and 1-star patterns).
_RE_BOLD_ITALIC_3 = re.compile(r"\*{3}(.+?)\*{3}")
#: ``**text**`` wrappers.
_RE_BOLD_ITALIC_2 = re.compile(r"\*{2}(.+?)\*{2}")
#: ``*text*`` wrappers.
_RE_BOLD_ITALIC_1 = re.compile(r"\*(.+?)\*")
#: Three or more consecutive newlines.
_RE_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
#: Unordered (``-``, ``*``, ``+``) and ordered (``1.``) list prefixes.
_RE_LIST_MARKER = re.compile(r"(?m)^[\t ]*(?:[-*+]|\d+\.)\s+")
#: HTML tags like ``<br>``, ``<div class="...">``.
_RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _convert_table_row(match: re.Match[str]) -> str:
    """Turn one table data row into comma-joined plain text."""
    inner = match.group(1)
    cells = [cell.strip() for cell in inner.split("|") if cell.strip()]
    return ", ".join(cells)


def clean_passage_for_rerank(text: str) -> str:
    """Strip markdown/structural noise from ``text`` for the rerank model.

    All meaningful natural-language content is preserved; formatting that
    would confuse text-similarity scoring is removed. Block-level patterns
    run before inline ones so partial matches cannot corrupt the result.
    """
    text = _RE_CODE_BLOCK.sub("", text)
    text = _RE_LATEX_BLOCK.sub("", text)
    text = _RE_HTML_TAG.sub("", text)
    text = _RE_LINKED_IMAGE.sub(r"![$1]($2)", text)
    text = _RE_MARKDOWN_IMAGE.sub("", text)
    text = _RE_MARKDOWN_LINK.sub(r"\1", text)
    text = _RE_RAW_URL.sub("", text)
    text = _RE_TABLE_SEP.sub("", text)
    text = _RE_TABLE_ROW.sub(_convert_table_row, text)
    text = _RE_HEADING_PREFIX.sub("", text)
    text = _RE_BLOCKQUOTE.sub("", text)
    text = _RE_BOLD_ITALIC_3.sub(r"\1", text)
    text = _RE_BOLD_ITALIC_2.sub(r"\1", text)
    text = _RE_BOLD_ITALIC_1.sub(r"\1", text)
    text = _RE_LIST_MARKER.sub("", text)
    text = _RE_EXCESSIVE_NEWLINES.sub("\n\n", text)
    return text.strip()


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _generated_question_strings(chunk_metadata: JsonObject) -> list[str]:
    """Extract the generated-question texts from chunk metadata."""
    questions = chunk_metadata.get("generated_questions")
    if not isinstance(questions, list):
        return []
    result: list[str] = []
    for entry in questions:
        if not isinstance(entry, dict):
            continue
        question = entry.get("question")
        if isinstance(question, str) and question:
            result.append(question)
    return result


def get_enriched_passage(result: SearchResult) -> str:
    """Merge content, image captions/OCR text and generated questions.

    ``image_info`` carries a JSON list of image objects; ``chunk_metadata``
    may carry AI-generated questions. Malformed payloads degrade to the
    cleaned content alone rather than failing the step.
    """
    combined_text = clean_passage_for_rerank(result.content)
    enrichments: list[str] = []

    if result.image_info:
        try:
            parsed = json.loads(result.image_info)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            for img in parsed:
                if not isinstance(img, dict):
                    continue
                caption = _as_str(img.get("caption"))
                if caption:
                    enrichments.append(caption)
                ocr_text = _as_str(img.get("ocr_text"))
                if ocr_text:
                    enrichments.append(ocr_text)

    if isinstance(result.chunk_metadata, dict):
        question_strings = _generated_question_strings(result.chunk_metadata)
        if question_strings:
            enrichments.append("; ".join(question_strings))

    if not enrichments:
        return combined_text
    if combined_text:
        combined_text += "\n\n"
    combined_text += "\n".join(enrichments)
    return combined_text


# ── Scoring helpers ────────────────────────────────────────────────────


def composite_score(result: SearchResult, model_score: float, base_score: float) -> float:
    """Combine the model score, the base score and a source prior.

    Web-search provenance is discounted slightly (0.95) to mirror the
    upstream weighting; the result is clamped to ``[0, 1]``.
    """
    source_weight = 0.95 if result.knowledge_source.lower() == "web_search" else 1.0
    composite = 0.6 * model_score + 0.3 * base_score + 0.1 * source_weight
    return min(max(composite, 0.0), 1.0)


def rerank_fallback_min_score(search_targets: Sequence[SearchTarget]) -> float:
    """Return the minimum top score that justifies the fallback top-1.

    An explicitly constrained tag/document scope preserves its best
    candidate instead of letting a global rerank threshold erase the entire
    authoritative scope, so the floor is 0 there.
    """
    if any(target.disable_recall_thresholds for target in search_targets):
        return 0.0
    return _DEFAULT_FALLBACK_MIN_SCORE


def _safe_top_score(results: Sequence[RankResult]) -> float:
    if not results:
        return 0.0
    return results[0].relevance_score


def _average_redundancy(token_sets: Sequence[set[str]]) -> float:
    if len(token_sets) <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            total += jaccard(token_sets[i], token_sets[j])
            pairs += 1
    if pairs == 0:
        return 0.0
    return total / pairs


def apply_mmr(
    results: list[SearchResult],
    k: int,
    lambda_value: float,
) -> list[SearchResult]:
    """Reduce redundancy with maximal marginal relevance.

    Greedily selects ``k`` candidates balancing each one's relevance
    against its similarity to the already-selected passages. Token sets are
    computed up front so the redundancy scan stays cheap.
    """
    if k <= 0 or not results:
        return []
    pipeline_info(
        "Rerank",
        "mmr_start",
        {"lambda": lambda_value, "k": k, "candidates": len(results)},
    )
    all_token_sets = [tokenize_simple(get_enriched_passage(result)) for result in results]

    selected: list[SearchResult] = []
    selected_token_sets: list[set[str]] = []
    selected_indices: set[int] = set()

    while len(selected) < k and len(selected_indices) < len(results):
        best_index = -1
        best_score = -1.0
        for index, result in enumerate(results):
            if index in selected_indices:
                continue
            relevance = result.score
            redundancy = 0.0
            for selected_tokens in selected_token_sets:
                similarity = jaccard(all_token_sets[index], selected_tokens)
                if similarity > redundancy:
                    redundancy = similarity
            mmr = lambda_value * relevance - (1.0 - lambda_value) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_index = index
        if best_index < 0:
            break
        selected.append(results[best_index])
        selected_token_sets.append(all_token_sets[best_index])
        selected_indices.add(best_index)

    avg_redundancy = _average_redundancy(selected_token_sets)
    pipeline_info(
        "Rerank",
        "mmr_done",
        {"selected": len(selected), "avg_redundancy": f"{avg_redundancy:.4f}"},
    )
    return selected


# ── Pipeline step ──────────────────────────────────────────────────────


class RerankPlugin:
    """Pipeline step that reranks retrieved chunks (``CHUNK_RERANK``)."""

    def __init__(self, model_service: RerankModelService) -> None:
        self._model_service = model_service

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_RERANK]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.needs_retrieval():
            return await next()
        pipeline_info(
            "Rerank",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "candidate_cnt": len(pipeline_ctx.search_result),
                "rerank_model": pipeline_ctx.rerank_model_id,
                "rerank_thresh": pipeline_ctx.rerank_threshold,
                "rewrite_query": pipeline_ctx.rewrite_query,
            },
        )
        if not pipeline_ctx.search_result:
            pipeline_info("Rerank", "skip", {"reason": "empty_search_result"})
            return await next()
        if not pipeline_ctx.rerank_model_id:
            pipeline_warn("Rerank", "skip", {"reason": "empty_model_id"})
            return await next()

        try:
            rerank_model = await self._model_service.get_rerank_model(
                tenant_id=pipeline_ctx.tenant_id,
                model_id=pipeline_ctx.rerank_model_id,
            )
        except Exception as exc:
            pipeline_error(
                "Rerank",
                "get_model",
                {"model_id": pipeline_ctx.rerank_model_id, "error": str(exc)},
            )
            return ERR_GET_RERANK_MODEL.with_error(exc)

        # Build the passages, dropping candidates whose cleaned passage is
        # empty (markdown-image-only or pure-noise chunks are not worth a
        # model call).
        passages: list[str] = []
        candidates: list[SearchResult] = []
        for result in pipeline_ctx.search_result:
            passage = get_enriched_passage(result)
            if not passage.strip():
                pipeline_info("Rerank", "empty_passage_skip", {"chunk_id": result.id})
                continue
            passages.append(passage)
            candidates.append(result)

        pipeline_info(
            "Rerank",
            "build_passages",
            {
                "total_cnt": len(pipeline_ctx.search_result),
                "candidate_cnt": len(candidates),
            },
        )

        rerank_resp: list[RankResult] = []
        if candidates:
            original_threshold = pipeline_ctx.rerank_threshold
            try:
                rerank_resp = await self._rerank(
                    rerank_model,
                    pipeline_ctx,
                    pipeline_ctx.rewrite_query,
                    passages,
                    candidates,
                )
            except Exception as exc:
                # The model call failed — fall back to the original retrieval
                # results so the pipeline can still return something useful.
                pipeline_warn(
                    "Rerank",
                    "api_error_fallback",
                    {"error": str(exc), "candidate_cnt": len(candidates)},
                )
                pipeline_ctx.search_result = list(candidates)
                return await next()

            # If nothing passed the threshold and it was high enough, retry
            # once with a degraded threshold before giving up.
            if not rerank_resp and original_threshold > _DEGRADE_THRESHOLD:
                degraded_threshold = max(original_threshold * 0.7, _DEGRADE_THRESHOLD)
                pipeline_warn(
                    "Rerank",
                    "threshold_degrade",
                    {
                        "original": original_threshold,
                        "degraded": degraded_threshold,
                        "candidate_cnt": len(candidates),
                        "reason": "no results above original threshold, retrying with lower threshold",
                    },
                )
                pipeline_ctx.rerank_threshold = degraded_threshold
                try:
                    rerank_resp = await self._rerank(
                        rerank_model,
                        pipeline_ctx,
                        pipeline_ctx.rewrite_query,
                        passages,
                        candidates,
                    )
                except Exception as exc:
                    pipeline_ctx.rerank_threshold = original_threshold
                    pipeline_warn(
                        "Rerank",
                        "api_error_fallback",
                        {"error": str(exc), "candidate_cnt": len(candidates)},
                    )
                    pipeline_ctx.search_result = list(candidates)
                    return await next()
                pipeline_ctx.rerank_threshold = original_threshold

        pipeline_info("Rerank", "model_response", {"result_cnt": len(rerank_resp)})

        reranked: list[SearchResult] = []
        updated_by_id: dict[str, SearchResult] = {}
        for rank_result in rerank_resp:
            if rank_result.index >= len(candidates):
                continue
            result = candidates[rank_result.index]
            base_score = result.score
            model_score = rank_result.relevance_score
            metadata = dict(result.metadata)
            metadata["base_score"] = f"{base_score:.4f}"
            metadata["model_score"] = f"{model_score:.4f}"
            updated = result.model_copy(
                update={
                    "score": composite_score(result, model_score, base_score),
                    "metadata": metadata,
                }
            )
            if (
                pipeline_ctx.faq_priority_enabled
                and pipeline_ctx.faq_score_boost > 1.0
                and updated.chunk_type == "faq"
            ):
                original_score = updated.score
                boosted_score = min(updated.score * pipeline_ctx.faq_score_boost, 1.0)
                metadata = dict(updated.metadata)
                metadata["faq_boosted"] = "true"
                metadata["faq_original_score"] = f"{original_score:.4f}"
                updated = updated.model_copy(
                    update={"score": boosted_score, "metadata": metadata}
                )
                pipeline_info(
                    "Rerank",
                    "faq_boost",
                    {
                        "chunk_id": updated.id,
                        "original_score": f"{original_score:.4f}",
                        "boosted_score": f"{boosted_score:.4f}",
                        "boost_factor": pipeline_ctx.faq_score_boost,
                    },
                )
            reranked.append(updated)
            updated_by_id[updated.id] = updated

        for rank, item in enumerate(reranked[:3], start=1):
            pipeline_info(
                "Rerank",
                "composite_top",
                {
                    "rank": rank,
                    "chunk_id": item.id,
                    "base_score": item.metadata.get("base_score"),
                    "final_score": f"{item.score:.4f}",
                },
            )

        top_k = min(len(reranked), max(1, pipeline_ctx.rerank_top_k))
        final = apply_mmr(reranked, top_k, _MMR_LAMBDA)
        pipeline_ctx.rerank_result = final

        # The upstream step mutates the shared candidates in place; publish
        # the updated copies back into ``search_result`` so later stages
        # observe the composite scores.
        pipeline_ctx.search_result = [
            updated_by_id.get(result.id, result) for result in pipeline_ctx.search_result
        ]

        if not pipeline_ctx.rerank_result:
            pipeline_warn("Rerank", "output", {"filtered_cnt": 0})
            return ERR_SEARCH_NOTHING

        pipeline_info("Rerank", "output", {"filtered_cnt": len(pipeline_ctx.rerank_result)})
        return await next()

    async def _rerank(
        self,
        rerank_model: Reranker,
        pipeline_ctx: PipelineContext,
        query: str,
        passages: list[str],
        candidates: list[SearchResult],
    ) -> list[RankResult]:
        """Call the rerank model and filter its results by threshold.

        Blank passages are dropped before the call; results below the
        (possibly degraded) threshold are filtered out. When the threshold
        wiped everything but the top score is still meaningful, the top
        candidate is kept as a safety net.
        """
        pipeline_info(
            "Rerank",
            "model_call",
            {"query_variant": query, "passages": len(passages)},
        )
        clean_passages: list[str] = []
        clean_candidates: list[SearchResult] = []
        for index, passage in enumerate(passages):
            if passage.strip():
                clean_passages.append(passage)
                if index < len(candidates):
                    clean_candidates.append(candidates[index])
        if not clean_passages:
            pipeline_info("Rerank", "model_call_skip", {"reason": "all_passages_empty"})
            return []

        resp = await rerank_model.rerank(query, clean_passages)

        threshold = pipeline_ctx.rerank_threshold
        rank_filter: list[RankResult] = []
        for result in resp:
            if result.index >= len(clean_candidates):
                continue
            if result.relevance_score >= threshold:
                rank_filter.append(result)

        fallback_min_score = rerank_fallback_min_score(pipeline_ctx.search_targets)
        if not rank_filter and resp and resp[0].relevance_score >= fallback_min_score:
            rank_filter = [resp[0]]
            pipeline_info(
                "Rerank",
                "fallback_top1",
                {
                    "reason": "all_below_threshold",
                    "threshold": threshold,
                    "top_score": resp[0].relevance_score,
                },
            )
        elif not rank_filter:
            pipeline_info(
                "Rerank",
                "fallback_skip",
                {
                    "reason": "top_score_too_low",
                    "threshold": threshold,
                    "top_score": _safe_top_score(resp),
                },
            )
        return rank_filter


__all__ = [
    "RerankModelService",
    "RerankPlugin",
    "apply_mmr",
    "clean_passage_for_rerank",
    "composite_score",
    "get_enriched_passage",
    "rerank_fallback_min_score",
]
