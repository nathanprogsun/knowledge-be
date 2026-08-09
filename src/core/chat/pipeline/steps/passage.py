"""Passage helpers shared by the completion-side pipeline steps.

Pure text transforms operating on a single search hit: HTML-escaped
document headers, chunk-type predicates, and the image-info enrichment
that splices per-image caption / OCR metadata back into a retrieved
passage for the model to see.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence

from src.common.json import JsonObject
from src.core.chat.pipeline.types import SearchResult

#: Chunk-type strings the completion steps key on.
CHUNK_TYPE_FAQ = "faq"
CHUNK_TYPE_WEB_SEARCH = "web_search"
CHUNK_TYPE_TABLE_SUMMARY = "table_summary"
CHUNK_TYPE_TABLE_COLUMN = "table_column"

#: Matches a Markdown image link ``![alt](url)``.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

#: Matches an HTML ``<img>`` tag with a quoted ``src`` attribute. The ``src``
#: must be preceded by whitespace so ``data-src``-style hyphenated attributes
#: are not mistaken for it.
_HTML_IMG_SRC_RE = re.compile(r'(?i)<img\b([^>]*?)\ssrc\s*=\s*[\'"]([^\'"]+)[\'"]([^>]*)>')

#: Submatch index of the ``src`` value in ``_HTML_IMG_SRC_RE``.
_HTML_IMG_SRC_URL_GROUP = 2


def get_enriched_passage_for_chat(result: SearchResult) -> str:
    """Return a hit's content merged with its image caption / OCR metadata.

    When the hit has no image info, the raw content is returned unchanged;
    when both are empty, the passage is empty.
    """
    if result.content == "" and result.image_info == "":
        return ""
    if result.image_info == "":
        return result.content
    return enrich_content_with_image_info_for_chat(result.content, result.image_info)


def enrich_content_with_image_info_for_chat(content: str, image_info_json: str) -> str:
    """Splice image metadata blocks into ``content`` after matching images.

    Every image referenced by a Markdown link or an ``<img src=...>`` tag
    whose URL appears in ``image_info_json`` gets a blockquote (caption +
    OCR text) injected immediately after it. Injections are applied
    right-to-left so splice positions stay valid.
    """
    if not image_info_json:
        return content
    try:
        parsed = json.loads(image_info_json)
    except ValueError:
        return content
    if not isinstance(parsed, list) or not parsed:
        return content

    image_info_map: dict[str, JsonObject] = {}
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url:
            image_info_map.setdefault(url, entry)
        original_url = entry.get("original_url")
        if isinstance(original_url, str) and original_url:
            image_info_map.setdefault(original_url, entry)

    injections: list[tuple[int, str]] = []
    for match in _MARKDOWN_IMAGE_RE.finditer(content):
        _collect_injection(match.end(), match.group(2), False, image_info_map, injections)
    for match in _HTML_IMG_SRC_RE.finditer(content):
        _collect_injection(
            match.end(), match.group(_HTML_IMG_SRC_URL_GROUP), True, image_info_map, injections
        )

    injections.sort(key=lambda pair: pair[0], reverse=True)
    for at, text in injections:
        content = content[:at] + text + content[at:]
    return content


def _collect_injection(
    at: int,
    raw_key: str,
    trim: bool,
    image_info_map: dict[str, JsonObject],
    injections: list[tuple[int, str]],
) -> None:
    """Append one metadata injection when ``raw_key`` resolves to an image."""
    key = raw_key.strip() if trim else raw_key
    info = image_info_map.get(key)
    if info is None:
        return
    metadata = _image_info_markdown_metadata(info)
    if metadata == "":
        return
    injections.append((at, "\n\n" + metadata))


def _image_info_markdown_metadata(info: JsonObject) -> str:
    """Render caption + OCR text as a Markdown blockquote, if any is present."""
    lines: list[str] = []
    caption = str(info.get("caption") or "").strip()
    if caption:
        lines.append("**Image caption:** " + caption)
    ocr = str(info.get("ocr_text") or "").strip()
    if ocr:
        lines.append("**Image text (OCR):** " + ocr)
    if not lines:
        return ""
    joined = "\n\n".join(lines)
    return "> " + joined.replace("\n", "\n> ")


def build_document_header(results: Sequence[SearchResult]) -> str:
    """Render a metadata section listing each unique knowledge document.

    One entry per knowledge id (title + description + custom metadata, all
    HTML-escaped); returns an empty string when no document has meaningful
    metadata.
    """
    seen: set[str] = set()
    documents: list[tuple[str, str, str]] = []
    for result in results:
        if result.knowledge_id == "" or result.knowledge_id in seen:
            continue
        seen.add(result.knowledge_id)
        title = result.knowledge_title or result.knowledge_filename
        if title == "":
            continue
        documents.append(
            (title, result.knowledge_description, result.knowledge_custom_metadata)
        )
    if not documents:
        return ""

    parts = ["<documents>\n"]
    for title, description, metadata in documents:
        parts.append("<document>\n")
        parts.append(f"<title>{html.escape(title)}</title>\n")
        if description:
            parts.append(f"<description>{html.escape(description)}</description>\n")
        if metadata:
            parts.append(f"<metadata>{html.escape(metadata)}</metadata>\n")
        parts.append("</document>\n")
    parts.append("</documents>")
    return "".join(parts)


__all__ = [
    "CHUNK_TYPE_FAQ",
    "CHUNK_TYPE_TABLE_COLUMN",
    "CHUNK_TYPE_TABLE_SUMMARY",
    "CHUNK_TYPE_WEB_SEARCH",
    "build_document_header",
    "enrich_content_with_image_info_for_chat",
    "get_enriched_passage_for_chat",
]
