"""Public-citation surface of the model context.

Owns the system protocol prompt, the regex vocabulary used to expand private
``<ref/>`` handles into canonical ``<kb/>`` / ``<web/>`` tags, the
word-boundary tag-start helpers, and the citation stream expander that keeps
partial tags off the wire. The registry-dependent compaction methods
(``compact_public_citations`` / ``expand_text`` / ...) live on the source
registry and reuse the constants and helpers declared here.
"""

from __future__ import annotations

import html
import re
from typing import Protocol

SOURCE_HANDLE_PROTOCOL_PROMPT = """

## Source handling protocol (system-owned)
Retrieved content uses request-local source handles: cN identifies a knowledge chunk, wN a web page, dN a document, and bN a knowledge base.
- Use dN and bN only as tool arguments when a tool requests a document or knowledge base.
- Never reveal raw chunk IDs, knowledge IDs, knowledge-base IDs, or private source handles in user-visible output. This does not change separate instructions to preserve retrieved Markdown image URLs."""

CITATION_ENABLED_PROTOCOL_PROMPT = """
- Source citations are enabled for this answer. Cite a knowledge chunk with exactly <ref id="cN"/> and a web page with exactly <ref id="wN"/>.
- Copy only cN/wN handles that appeared in supplied context or tool results. Never cite dN/bN.
- Never output <kb> or <web> tags yourself; the system expands valid <ref/> tags after generation.
- Keep each <ref/> inline on the same line as the claim it supports. Do not group citations at the end.
- These rules supersede earlier, saved, or custom prompt instructions about citation syntax."""

CITATION_DISABLED_PROTOCOL_PROMPT = """
- Source citations are disabled for this answer. Do not output <ref>, <kb>, <web>, raw source URLs, or source-handle citations.
- These rules supersede earlier, saved, or custom prompt instructions that require source citations."""

RESOURCE_HANDLE_PROTOCOL_PROMPT = """

## Resource handle protocol (system-owned)
Some durable resources and high-entropy Wiki slugs are represented by request-local res://NNNN handles. Wiki issues may use iN handles.
- Copy only handles that appeared in supplied context or tool results, preserving them exactly in links, images, and tool arguments.
- Never invent, edit, or expand any handle. The system restores it after generation."""


def source_protocol_prompt(citations_enabled: bool) -> str:
    """Return the internal, non-user-editable source protocol for a model call."""
    if citations_enabled:
        return SOURCE_HANDLE_PROTOCOL_PROMPT + CITATION_ENABLED_PROTOCOL_PROMPT
    return SOURCE_HANDLE_PROTOCOL_PROMPT + CITATION_DISABLED_PROTOCOL_PROMPT


_PUBLIC_KB_TAG_RE = re.compile(r"<kb\b[^>]*>", re.IGNORECASE | re.DOTALL)
_PUBLIC_WEB_TAG_RE = re.compile(r"<web\b[^>]*>", re.IGNORECASE | re.DOTALL)
_DOC_ATTR_RE = re.compile(r'\bdoc\s*=\s*"([^"]*)"', re.IGNORECASE)
_CHUNK_ATTR_RE = re.compile(r'\bchunk_id\s*=\s*"([^"]+)"', re.IGNORECASE)
_PUBLIC_KB_ATTR_RE = re.compile(r'\bkb_id\s*=\s*"([^"]*)"', re.IGNORECASE)
_URL_ATTR_RE = re.compile(r'\burl\s*=\s*"([^"]+)"', re.IGNORECASE)
_TITLE_ATTR_RE = re.compile(r'\btitle\s*=\s*"([^"]*)"', re.IGNORECASE)
_LEGACY_CHUNK_RE = re.compile(r"<(?:chunk|faq)\b[^>]*>", re.IGNORECASE | re.DOTALL)
_FAQ_ATTR_RE = re.compile(r'\bfaq_id\s*=\s*"([^"]+)"', re.IGNORECASE)
_KNOWLEDGE_TITLE_ATTR_RE = re.compile(r'\bknowledge_title\s*=\s*"([^"]*)"', re.IGNORECASE)

_REF_TAG_RE = re.compile(r'<ref\s+id\s*=\s*"([^"]+)"\s*/?>', re.IGNORECASE)
_REF_CANDIDATE_RE = re.compile(r"<ref(?:\s|$)[^>]*(?:>|$)", re.IGNORECASE | re.DOTALL)
_MODEL_KB_TAG_RE = re.compile(r"<kb(?:\s|$)[^>]*(?:>|$)", re.IGNORECASE | re.DOTALL)
_MODEL_WEB_TAG_RE = re.compile(r"<web(?:\s|$)[^>]*(?:>|$)", re.IGNORECASE | re.DOTALL)

_DOCUMENT_ATTR_RE = re.compile(r'\bknowledge_id\s*=\s*"([^"]+)"', re.IGNORECASE)
_DOCUMENT_ELEMENT_RE = re.compile(
    r"<knowledge_id>\s*([^<]+?)\s*</knowledge_id>", re.IGNORECASE | re.DOTALL
)
_KB_ATTR_RE = re.compile(r'\b(?:knowledge_base_id|kb_id)\s*=\s*"([^"]+)"', re.IGNORECASE)
_KB_ELEMENT_RE = re.compile(
    r"<(?:knowledge_base_id|kb_id)>\s*([^<]+?)\s*</(?:knowledge_base_id|kb_id)>",
    re.IGNORECASE | re.DOTALL,
)


def public_attr(expression: re.Pattern[str], tag: str) -> str:
    """Extract the captured attribute from ``tag``, unescaping HTML entities."""
    match = expression.search(tag)
    if match is None:
        return ""
    return html.unescape(match.group(1))


def escape_attr(value: str) -> str:
    """Escape a string for use inside a double-quoted attribute value."""
    return html.escape(value, quote=True)


def is_ref_tag_start(value: str) -> bool:
    return is_named_tag_start(value, "ref")


def is_named_tag_start(value: str, name: str) -> bool:
    """Return whether ``value`` starts the opening ``<name ...>`` tag."""
    prefix = f"<{name}"
    if not value.startswith(prefix):
        return False
    if len(value) == len(prefix):
        return True
    next_char = value[len(prefix)]
    return next_char in " \t\r\n>"


def is_source_tag_pending(value: str) -> bool:
    """Return whether ``value`` could still grow into a source tag prefix."""
    for name in ("ref", "kb", "web"):
        prefix = f"<{name}"
        if (len(value) <= len(prefix) and prefix.startswith(value)) or is_named_tag_start(
            value, name
        ):
            return True
    return False


class _CitationCodec(Protocol):
    """Structural interface the stream expander needs from the source registry."""

    def expand_text(self, text: str) -> str:
        """Expand private source handles into canonical public tags."""
        ...


class CitationStreamExpander:
    """Prevents partial private ``<ref/>`` tags from reaching a stream.

    Non-tag content streams normally; only a trailing run that could still grow
    into a source tag is withheld until the next chunk or the stream flush.
    """

    def __init__(self, registry: _CitationCodec) -> None:
        self._registry = registry
        self._pending = ""

    def feed(self, chunk: str) -> str:
        """Consume one stream chunk and return the text safe to emit."""
        data = self._pending + chunk
        self._pending = ""
        out: list[str] = []
        while data != "":
            idx = data.find("<")
            if idx < 0:
                out.append(data)
                break
            out.append(data[:idx])
            data = data[idx:]
            lower = data.lower()
            if is_source_tag_pending(lower) and ">" not in data:
                self._pending = data
                break
            if is_ref_tag_start(lower):
                end = data.find(">")
                if end < 0:
                    self._pending = data
                    break
                tag = data[: end + 1]
                if _REF_TAG_RE.search(tag):
                    out.append(self._registry.expand_text(tag))
                data = data[end + 1 :]
                continue
            if is_named_tag_start(lower, "kb") or is_named_tag_start(lower, "web"):
                end = data.find(">")
                if end < 0:
                    self._pending = data
                    break
                data = data[end + 1 :]
                continue
            out.append("<")
            data = data[1:]
        return "".join(out)

    def flush(self) -> str:
        """Decide what happens to a suffix still held when the stream closes."""
        pending = self._pending
        self._pending = ""
        if is_source_tag_pending(pending.lower()):
            return ""
        return self._registry.expand_text(pending)


__all__ = [
    "CITATION_DISABLED_PROTOCOL_PROMPT",
    "CITATION_ENABLED_PROTOCOL_PROMPT",
    "RESOURCE_HANDLE_PROTOCOL_PROMPT",
    "SOURCE_HANDLE_PROTOCOL_PROMPT",
    "CitationStreamExpander",
    "escape_attr",
    "is_named_tag_start",
    "is_ref_tag_start",
    "is_source_tag_pending",
    "public_attr",
    "source_protocol_prompt",
]
