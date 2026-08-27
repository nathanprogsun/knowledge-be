"""FAQ metadata parsing and XML / JSON helpers for the retrieval tools.

FAQ entries are stored as ``faq``-type chunks whose ``metadata`` JSON
carries the standard question, similar questions, and answers. This module
parses that metadata leniently and renders it into tool output XML and
structured result maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import Pattern

from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.text_utils import (
    SNIPPET_MAX_ANSWER_RUNES,
    SNIPPET_MAX_TOTAL_RUNES,
    extract_snippet_regex,
    regex_matches_any,
    search_query_tokens,
    text_matches_search_queries,
    truncate_runes,
    xml_escape,
)
from src.db.models.chunk import Chunk

#: Chunk type string for FAQ entries.
FAQ_CHUNK_TYPE = "faq"

#: Caps similar-question rows in tool output.
FAQ_MAX_SIMILAR_QUESTIONS_DISPLAY = 5


@dataclass(frozen=True, slots=True)
class FAQChunkMetadata:
    """Parsed FAQ metadata carried in a chunk's ``metadata`` JSON."""

    standard_question: str = ""
    similar_questions: tuple[str, ...] = ()
    negative_questions: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    answer_strategy: str = ""
    version: int = 0
    source: str = ""


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _as_int(value: JsonValue) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def faq_metadata_from_json(raw: JsonObject | None) -> FAQChunkMetadata | None:
    """Parse FAQ metadata from a chunk ``metadata`` JSON object."""
    if not isinstance(raw, dict):
        return None
    return FAQChunkMetadata(
        standard_question=_as_str(raw.get("standard_question")),
        similar_questions=_as_str_list(raw.get("similar_questions")),
        negative_questions=_as_str_list(raw.get("negative_questions")),
        answers=_as_str_list(raw.get("answers")),
        answer_strategy=_as_str(raw.get("answer_strategy")),
        version=_as_int(raw.get("version")),
        source=_as_str(raw.get("source")),
    )


def faq_metadata_from_chunk(chunk: Chunk) -> FAQChunkMetadata | None:
    """Parse FAQ metadata from a chunk row, or ``None`` for non-FAQ chunks."""
    if chunk is None or chunk.chunk_type != FAQ_CHUNK_TYPE:
        return None
    return faq_metadata_from_json(chunk.metadata)


def faq_standard_question(chunk: Chunk) -> str:
    """Return the FAQ standard question, or ``""`` when unavailable."""
    if chunk is None or chunk.chunk_type != FAQ_CHUNK_TYPE:
        return ""
    meta = faq_metadata_from_chunk(chunk)
    if meta is None:
        return ""
    return meta.standard_question.strip()


def faq_match_snippet(chunk: Chunk, compiled: list[Pattern[str]]) -> str:
    """Build a "Q: … | A: …" snippet for grep regex hits."""
    if chunk is None:
        return ""
    meta = faq_metadata_from_chunk(chunk)
    if meta is None:
        return ""
    question = _faq_matched_question_from_regex(meta, compiled)
    if not question:
        question = meta.standard_question.strip()
    if not question:
        return ""
    return format_faq_match_snippet(question, meta.answers)


def extract_chunk_match_snippet(chunk: Chunk, compiled: list[Pattern[str]]) -> str:
    """Return a preview for tool output.

    FAQ chunks surface the matched question plus answers from metadata
    (answers are not stored in chunk content for question-only index mode);
    other chunk types use regex context around the first body match.
    """
    if chunk is not None and chunk.chunk_type == FAQ_CHUNK_TYPE:
        snippet = faq_match_snippet(chunk, compiled)
        if snippet:
            return snippet
    if chunk is None:
        return ""
    return extract_snippet_regex(chunk.content, compiled)


def faq_match_snippet_from_queries(
    meta: FAQChunkMetadata | None,
    queries: list[str],
) -> str:
    """Build a "Q: … | A: …" snippet for knowledge-search hits."""
    if meta is None:
        return ""
    question = _faq_matched_question_from_queries(meta, queries)
    if not question:
        question = meta.standard_question.strip()
    return format_faq_match_snippet(question, meta.answers)


def format_faq_match_snippet(question: str, answers: tuple[str, ...]) -> str:
    """Format ``Q: question | A: answer`` with a bounded answer snippet."""
    question = question.strip()
    if not question:
        return ""
    answer = faq_answers_for_snippet(answers)
    snippet = f"Q: {question} | A: {answer}" if answer else f"Q: {question}"
    snippet = snippet.strip()
    if len(snippet) > SNIPPET_MAX_TOTAL_RUNES:
        snippet = truncate_runes(snippet, SNIPPET_MAX_TOTAL_RUNES)
    return snippet


def faq_answers_for_snippet(answers: tuple[str, ...]) -> str:
    """Join the non-empty answers into one bounded snippet string."""
    parts = [answer.strip() for answer in answers if answer.strip()]
    if not parts:
        return ""
    return truncate_runes(" | ".join(parts), SNIPPET_MAX_ANSWER_RUNES)


def _faq_matched_question_from_regex(
    meta: FAQChunkMetadata,
    compiled: list[Pattern[str]],
) -> str:
    for similar in meta.similar_questions:
        if regex_matches_any(similar, compiled):
            return similar
    if regex_matches_any(meta.standard_question, compiled):
        return meta.standard_question
    return meta.standard_question


def _faq_matched_question_from_queries(
    meta: FAQChunkMetadata,
    queries: list[str],
) -> str:
    tokens = search_query_tokens(queries)
    for similar in meta.similar_questions:
        if text_matches_search_queries(similar, queries, tokens):
            return similar
    if text_matches_search_queries(meta.standard_question, queries, tokens):
        return meta.standard_question
    return meta.standard_question


def _truncate_similar_questions(
    questions: tuple[str, ...] | list[str],
) -> tuple[list[str], int]:
    items = list(questions)
    if len(items) <= FAQ_MAX_SIMILAR_QUESTIONS_DISPLAY:
        return items, 0
    return items[:FAQ_MAX_SIMILAR_QUESTIONS_DISPLAY], len(items) - FAQ_MAX_SIMILAR_QUESTIONS_DISPLAY


def write_similar_questions_xml(
    parts: list[str],
    questions: tuple[str, ...] | list[str],
) -> None:
    """Emit ``<similar_question>`` rows (bounded) into ``parts``."""
    display, omitted = _truncate_similar_questions(questions)
    for similar in display:
        parts.append(f"<similar_question>{xml_escape(similar)}</similar_question>\n")
    if omitted > 0:
        parts.append(f'<similar_questions_omitted count="{omitted}" />\n')


def append_similar_questions_to_chunk_data(
    data: JsonObject,
    questions: tuple[str, ...] | list[str],
) -> None:
    """Add the bounded similar-question list to a structured result map."""
    display, omitted = _truncate_similar_questions(questions)
    if not display:
        return
    data["faq_similar_questions"] = list(display)
    if omitted > 0:
        data["faq_similar_questions_omitted"] = omitted


def write_faq_fields_xml(parts: list[str], meta: FAQChunkMetadata | None) -> None:
    """Emit question / similar-question / answer children (no wrapper)."""
    if meta is None:
        return
    if meta.standard_question:
        parts.append(f"<question>{xml_escape(meta.standard_question)}</question>\n")
    write_similar_questions_xml(parts, meta.similar_questions)
    for answer in meta.answers:
        if not answer.strip():
            continue
        parts.append(f"<answer>{xml_escape(answer)}</answer>\n")


def faq_fields_empty(meta: FAQChunkMetadata | None) -> bool:
    """Whether the metadata carries no displayable FAQ fields."""
    if meta is None:
        return True
    return not meta.standard_question and not meta.similar_questions and not meta.answers


def write_faq_metadata_xml(parts: list[str], meta: FAQChunkMetadata | None) -> None:
    """Emit a nested ``<faq>`` block (used inside a knowledge-search chunk)."""
    if faq_fields_empty(meta):
        return
    parts.append("<faq>\n")
    write_faq_fields_xml(parts, meta)
    parts.append("</faq>\n")


def write_faq_entry_xml(parts: list[str], chunk: Chunk) -> None:
    """Emit a top-level FAQ entry (not wrapped in ``<chunk>``)."""
    if chunk is None or chunk.chunk_type != FAQ_CHUNK_TYPE:
        return
    meta = faq_metadata_from_chunk(chunk)
    question = faq_standard_question(chunk)
    question_attr = f' question="{xml_escape(question)}"' if question else ""
    parts.append(
        f'<faq faq_id="{xml_escape(chunk.id)}" index="{chunk.chunk_index}"{question_attr}>\n'
    )
    if not faq_fields_empty(meta):
        write_faq_fields_xml(parts, meta)
    elif question:
        parts.append(f"<question>{xml_escape(question)}</question>\n")
    parts.append("</faq>\n")


def normalize_faq_chunk_data_map(data: JsonObject, chunk: Chunk) -> None:
    """Use ``faq_id`` / ``index`` instead of ``chunk_id`` / ``chunk_index``.

    Mutates the freshly-built per-chunk result map in place (the map is
    constructed by the caller for exactly this chunk).
    """
    if chunk is None or chunk.chunk_type != FAQ_CHUNK_TYPE or data is None:
        return
    data["faq_id"] = chunk.id
    data["index"] = chunk.chunk_index
    data.pop("chunk_id", None)
    data.pop("chunk_index", None)


def append_faq_chunk_data(data: JsonObject, chunk: Chunk) -> None:
    """Add FAQ metadata fields to a structured chunk result map."""
    if chunk is None or chunk.chunk_type != FAQ_CHUNK_TYPE:
        return
    meta = faq_metadata_from_chunk(chunk)
    if meta is None:
        return
    question = meta.standard_question.strip()
    if question:
        data["faq_question"] = question
    append_similar_questions_to_chunk_data(data, meta.similar_questions)
    if meta.answers:
        data["faq_answers"] = list(meta.answers)


__all__ = [
    "FAQ_CHUNK_TYPE",
    "FAQ_MAX_SIMILAR_QUESTIONS_DISPLAY",
    "FAQChunkMetadata",
    "append_faq_chunk_data",
    "append_similar_questions_to_chunk_data",
    "extract_chunk_match_snippet",
    "faq_answers_for_snippet",
    "faq_fields_empty",
    "faq_match_snippet",
    "faq_match_snippet_from_queries",
    "faq_metadata_from_chunk",
    "faq_metadata_from_json",
    "faq_standard_question",
    "format_faq_match_snippet",
    "normalize_faq_chunk_data_map",
    "write_faq_entry_xml",
    "write_faq_fields_xml",
    "write_faq_metadata_xml",
    "write_similar_questions_xml",
]
