"""Source-reference half of the model-context registry.

Owns the request-local ``cN``/``dN``/``bN``/``wN`` handles for chunks,
documents, knowledge bases and web pages, the tool-argument codec that maps
them back to durable identifiers, and the public-citation compaction methods
that reuse the citation vocabulary declared in ``citations``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast
from urllib.parse import urlparse, urlunparse

from src.ai.llm.types import LLMToolCall, Message, SearchResult
from src.common.json import JsonValue
from src.core.agents.engine.modelcontext.citations import (
    _CHUNK_ATTR_RE,
    _DOC_ATTR_RE,
    _DOCUMENT_ATTR_RE,
    _DOCUMENT_ELEMENT_RE,
    _FAQ_ATTR_RE,
    _KB_ATTR_RE,
    _KB_ELEMENT_RE,
    _KNOWLEDGE_TITLE_ATTR_RE,
    _LEGACY_CHUNK_RE,
    _MODEL_KB_TAG_RE,
    _MODEL_WEB_TAG_RE,
    _PUBLIC_KB_ATTR_RE,
    _PUBLIC_KB_TAG_RE,
    _PUBLIC_WEB_TAG_RE,
    _REF_CANDIDATE_RE,
    _REF_TAG_RE,
    _TITLE_ATTR_RE,
    _URL_ATTR_RE,
    escape_attr,
    public_attr,
    source_protocol_prompt,
)
from src.core.agents.engine.modelcontext.handle_table import HandleTable as _HandleTable
from src.core.agents.engine.modelcontext.tool_policy import (
    ArgumentPolicy,
    ResultPolicy,
    SourceKeySpace,
    source_key_spaces,
)

_S = TypeVar("_S")

#: Handle-shaped source values; a model-emitted handle is echoed back only when
#: it already exists and is never accepted as a new durable identity.
_SHORT_SOURCE_HANDLE_RE = re.compile(r"^[cdbw][1-9][0-9]*$", re.IGNORECASE)
_SHORT_SOURCE_HANDLE_IN_TEXT_RE = re.compile(r"\b[cdbw][1-9][0-9]*\b", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkReference:
    """Metadata registered alongside a chunk handle."""

    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    document_title: str = ""
    chunk_index: int = 0
    chunk_type: str = ""


@dataclass(frozen=True)
class WebMeta:
    """Per-web-page metadata stored next to the raw URL."""

    title: str = ""


def merge_chunk_reference(dst: ChunkReference, src: ChunkReference) -> ChunkReference:
    """Fold non-empty metadata of ``src`` into ``dst`` (first-wins)."""
    return ChunkReference(
        chunk_id=dst.chunk_id,
        knowledge_id=dst.knowledge_id or src.knowledge_id,
        knowledge_base_id=dst.knowledge_base_id or src.knowledge_base_id,
        document_title=dst.document_title or src.document_title,
        chunk_index=dst.chunk_index or src.chunk_index,
        chunk_type=dst.chunk_type or src.chunk_type,
    )


def _merge_web_meta(dst: WebMeta, src: WebMeta) -> WebMeta:
    """Keep the first non-empty title seen for a canonical URL."""
    return WebMeta(title=dst.title or src.title)


def first_non_empty(*values: str) -> str:
    """Return the first value whose trimmed content is non-empty."""
    for value in values:
        if value.strip() != "":
            return value
    return ""


def _known_handle(table: _HandleTable[_S], id: str) -> str:
    """Echo a model-emitted handle back only when it already exists."""
    handle = id.lower()
    if table.has(handle):
        return handle
    return ""


def canonical_web_url(raw: str) -> str:
    """Return the dedup key for a web reference (fragment-stripped, normalized).

    The raw URL the model was shown is stored as the decode value; only the
    canonical form is used for handle allocation.
    """
    value = raw.strip()
    parsed = urlparse(value)
    if parsed.scheme == "" or parsed.netloc == "":
        return value
    return urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.params, parsed.query, "")
    )


def _loads_json(raw: str) -> JsonValue | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return cast(JsonValue, value)


class SourceRegistry:
    """Request-scoped source handles, scoped to one assistant response.

    Handles are never persisted or accepted across requests.
    """

    def __init__(self, citations_enabled: bool = True) -> None:
        self.citations_enabled = citations_enabled
        self.chunks: _HandleTable[ChunkReference] = _HandleTable[ChunkReference]("c", 0, 1)
        self.docs: _HandleTable[None] = _HandleTable[None]("d", 0, 1)
        self.kbs: _HandleTable[None] = _HandleTable[None]("b", 0, 1)
        self.webs: _HandleTable[WebMeta] = _HandleTable[WebMeta]("w", 0, 1)

    def protocol_prompt(self) -> str:
        """Return the source protocol configured for this registry."""
        return source_protocol_prompt(self.citations_enabled)

    def count(self) -> int:
        """Return the number of chunk and web handles allocated."""
        return self.chunks.size() + self.webs.size()

    # ── Registration ────────────────────────────────────────────────────────

    def register_chunk(self, ref: ChunkReference) -> str:
        ref = ChunkReference(
            chunk_id=ref.chunk_id.strip(),
            knowledge_id=ref.knowledge_id,
            knowledge_base_id=ref.knowledge_base_id,
            document_title=ref.document_title,
            chunk_index=ref.chunk_index,
            chunk_type=ref.chunk_type,
        )
        if ref.chunk_id == "":
            return ""
        if _SHORT_SOURCE_HANDLE_RE.match(ref.chunk_id):
            return _known_handle(self.chunks, ref.chunk_id)
        return self.chunks.register(ref.chunk_id, ref.chunk_id, ref, merge_chunk_reference)

    def register_document(self, id: str) -> str:
        id = id.strip()
        if id == "":
            return ""
        if _SHORT_SOURCE_HANDLE_RE.match(id):
            return _known_handle(self.docs, id)
        return self.docs.register(id, id, None, None)

    def register_knowledge_base(self, id: str) -> str:
        id = id.strip()
        if id == "":
            return ""
        if _SHORT_SOURCE_HANDLE_RE.match(id):
            return _known_handle(self.kbs, id)
        return self.kbs.register(id, id, None, None)

    def register_web(self, raw_url: str, title: str) -> str:
        raw_url = raw_url.strip()
        if raw_url == "":
            return ""
        if _SHORT_SOURCE_HANDLE_RE.match(raw_url):
            return _known_handle(self.webs, raw_url)
        return self.webs.register(
            canonical_web_url(raw_url), raw_url, WebMeta(title=title), _merge_web_meta
        )

    def register_search_results(self, results: list[SearchResult]) -> None:
        """Register every durable identifier present in knowledge-search hits."""
        for result in results:
            self.register_document(result.knowledge_id)
            self.register_knowledge_base(result.knowledge_base_id)
            self.register_chunk(
                ChunkReference(
                    chunk_id=result.id,
                    knowledge_id=result.knowledge_id,
                    knowledge_base_id=result.knowledge_base_id,
                    document_title=first_non_empty(
                        result.knowledge_title, result.knowledge_filename
                    ),
                    chunk_index=result.chunk_index,
                    chunk_type=result.chunk_type,
                )
            )

    def chunk_handle(self, id: str) -> str:
        handle = self.chunks.handle_for_key(id)
        if handle is None:
            return ""
        return handle

    # ── Public-citation compaction / expansion ─────────────────────────────

    def register_legacy_tool_references(self, text: str) -> None:
        """Register durable IDs found inside legacy ``<chunk>`` / ``<faq>`` tags."""
        if text == "":
            return
        self.register_labeled_references(text)
        for tag in _LEGACY_CHUNK_RE.findall(text):
            chunk_id = first_non_empty(
                public_attr(_CHUNK_ATTR_RE, tag), public_attr(_FAQ_ATTR_RE, tag)
            )
            if chunk_id == "":
                continue
            self.register_chunk(
                ChunkReference(
                    chunk_id=chunk_id,
                    knowledge_id=public_attr(_DOCUMENT_ATTR_RE, tag),
                    knowledge_base_id=first_non_empty(
                        public_attr(_KB_ATTR_RE, tag), public_attr(_PUBLIC_KB_ATTR_RE, tag)
                    ),
                    document_title=first_non_empty(
                        public_attr(_KNOWLEDGE_TITLE_ATTR_RE, tag), public_attr(_DOC_ATTR_RE, tag)
                    ),
                )
            )

    def compact_public_citations(self, text: str) -> str:
        """Fold canonical citations from prior turns back into private refs.

        This prevents durable chunk IDs and web URLs in conversation history
        from becoming model-visible again.
        """
        if text == "":
            return text

        def kb_repl(tag: str) -> str:
            chunk_id = public_attr(_CHUNK_ATTR_RE, tag)
            if chunk_id == "":
                return tag
            handle = self.register_chunk(
                ChunkReference(
                    chunk_id=chunk_id,
                    knowledge_base_id=public_attr(_PUBLIC_KB_ATTR_RE, tag),
                    document_title=public_attr(_DOC_ATTR_RE, tag),
                )
            )
            return f'<ref id="{handle}"/>'

        text = _PUBLIC_KB_TAG_RE.sub(lambda match: kb_repl(match.group(0)), text)

        def web_repl(tag: str) -> str:
            raw_url = public_attr(_URL_ATTR_RE, tag)
            if raw_url == "":
                return tag
            handle = self.register_web(raw_url, public_attr(_TITLE_ATTR_RE, tag))
            return f'<ref id="{handle}"/>'

        return _PUBLIC_WEB_TAG_RE.sub(lambda match: web_repl(match.group(0)), text)

    def register_labeled_references(self, text: str) -> None:
        """Register explicit ID labels from metadata-oriented tool output.

        Only explicit ID labels are recognized; UUID-like text in retrieved
        content is never guessed to be a source identifier.
        """
        if text == "":
            return
        for expression in (_DOCUMENT_ATTR_RE, _DOCUMENT_ELEMENT_RE):
            for match in expression.finditer(text):
                self.register_document(match.group(1).strip())
        for expression in (_KB_ATTR_RE, _KB_ELEMENT_RE):
            for match in expression.finditer(text):
                self.register_knowledge_base(match.group(1).strip())

    def expand_text(self, text: str) -> str:
        """Convert the private model protocol into the public citation contract.

        Unknown handles fail closed and disappear; model-written public tags are
        dropped before canonical tags are created from registered handles.
        """
        if text == "":
            return text
        text = _MODEL_KB_TAG_RE.sub("", text)
        text = _MODEL_WEB_TAG_RE.sub("", text)
        if not self.citations_enabled:
            return _REF_CANDIDATE_RE.sub("", text)

        def repl(tag: str) -> str:
            match = _REF_TAG_RE.search(tag)
            if match is None:
                return ""
            handle = match.group(1).lower()
            chunk = self.chunks.resolve(handle)
            if chunk is not None:
                chunk_id, chunk_ref = chunk
                attrs = f'doc="{escape_attr(chunk_ref.document_title)}" chunk_id="{escape_attr(chunk_id)}"'
                if chunk_ref.knowledge_base_id != "":
                    attrs += f' kb_id="{escape_attr(chunk_ref.knowledge_base_id)}"'
                return f"<kb {attrs} />"
            web = self.webs.resolve(handle)
            if web is not None:
                raw_url, web_meta = web
                return f'<web url="{escape_attr(raw_url)}" title="{escape_attr(web_meta.title)}" />'
            return ""

        return _REF_CANDIDATE_RE.sub(lambda match: repl(match.group(0)), text)

    # ── Tool-call codec ─────────────────────────────────────────────────────

    def decode_tool_calls_with_policy(
        self, tool_calls: list[LLMToolCall], policy: ArgumentPolicy | None
    ) -> None:
        """Restore handles only for fields explicitly owned by the named tool."""
        for call in tool_calls:
            tool_name = call.function.name
            allowed = _allowed_key(policy, tool_name)
            call.function = call.function.model_copy(
                update={
                    "arguments": self._decode_json_with_policy(
                        call.function.arguments, False, allowed
                    )
                }
            )

    def unresolved_tool_handles_with_policy(
        self, tool_name: str, raw: str, policy: ArgumentPolicy | None
    ) -> list[str]:
        """Report unknown handles only in fields owned by the named tool."""
        if raw.strip() == "":
            return []
        value = _loads_json(raw)
        if value is None:
            return []
        seen: set[str] = set()
        self._collect_unresolved_tool_handles("", value, seen, _allowed_key(policy, tool_name))
        return sorted(seen)

    def _collect_unresolved_tool_handles(
        self,
        key: str,
        value: JsonValue,
        seen: set[str],
        allowed: Callable[[str], bool],
    ) -> None:
        if isinstance(value, str):
            lowered = key.lower()
            if lowered not in source_key_spaces or not allowed(lowered):
                return
            handle = value.strip()
            if _SHORT_SOURCE_HANDLE_RE.match(handle) and self._durable_for_handle(handle) == "":
                seen.add(handle)
        elif isinstance(value, list):
            for item in value:
                self._collect_unresolved_tool_handles(key, item, seen, allowed)
        elif isinstance(value, dict):
            for child_key, item in value.items():
                self._collect_unresolved_tool_handles(child_key, item, seen, allowed)

    def encode_messages_with_policies(
        self,
        messages: list[Message],
        argument_policy: ArgumentPolicy | None,
        result_policy: ResultPolicy | None,
    ) -> list[Message]:
        """Compact known identifiers in replayed messages, gated by tool name.

        The two-pass shape registers every durable identifier present in
        historical tool calls and canonical citations before compaction, so an
        early tool message can reuse metadata that appears only in the turn's
        final answer.
        """
        if not messages:
            return messages
        out = list(messages)
        for i, message in enumerate(out):
            process_tool_result = message.role == "tool" and (
                result_policy is None or result_policy(message.name)
            )
            if message.tool_calls:
                for call in message.tool_calls:
                    tool_name = call.function.name
                    self._register_tool_arguments(
                        call.function.arguments, _allowed_key(argument_policy, tool_name)
                    )
            if message.role == "assistant" or process_tool_result:
                out[i] = message.model_copy(
                    update={
                        "content": self.compact_public_citations(message.content),
                        "reasoning_content": self.compact_public_citations(
                            message.reasoning_content
                        ),
                        "multi_content": [
                            part.model_copy(
                                update={"text": self.compact_public_citations(part.text)}
                            )
                            if part.type == "text"
                            else part
                            for part in message.multi_content
                        ],
                    }
                )
        for i, message in enumerate(out):
            if message.role == "tool" and (result_policy is None or result_policy(message.name)):
                self.register_legacy_tool_references(message.content)
                out[i] = message.model_copy(
                    update={"content": self.compact_known_text(message.content)}
                )
            if not message.tool_calls:
                continue
            tool_calls = list(message.tool_calls)
            for j, call in enumerate(tool_calls):
                tool_name = call.function.name
                arguments = self._decode_json_with_policy(
                    call.function.arguments, True, _allowed_key(argument_policy, tool_name)
                )
                tool_calls[j] = call.model_copy(
                    update={"function": call.function.model_copy(update={"arguments": arguments})}
                )
            out[i] = message.model_copy(update={"tool_calls": tool_calls})
        return out

    def _register_tool_arguments(self, raw: str, allowed: Callable[[str], bool]) -> None:
        if raw.strip() == "":
            return
        value = _loads_json(raw)
        if value is None:
            return
        self._register_tool_argument_value("", value, allowed)

    def _register_tool_argument_value(
        self, key: str, value: JsonValue, allowed: Callable[[str], bool]
    ) -> None:
        if isinstance(value, str):
            if allowed(key.lower()):
                self.register_source_id_by_key(key, value)
        elif isinstance(value, list):
            for item in value:
                self._register_tool_argument_value(key, item, allowed)
        elif isinstance(value, dict):
            for child_key, item in value.items():
                self._register_tool_argument_value(child_key, item, allowed)

    def register_source_id_by_key(self, key: str, value: str) -> None:
        """Register a durable ID by key dispatch, mirroring the tool-argument rules.

        Only public web pages become web references; internal schemes
        (``res://``, storage providers) never enter the web handle space, where
        compaction would rewrite them a second time.
        """
        value = value.strip()
        if value == "" or _SHORT_SOURCE_HANDLE_RE.match(value):
            return
        space = source_key_spaces.get(key.lower())
        if space is None:
            return
        match space:
            case SourceKeySpace.CHUNK:
                self.register_chunk(ChunkReference(chunk_id=value))
            case SourceKeySpace.DOCUMENT:
                self.register_document(value)
            case SourceKeySpace.DOCUMENT_REF:
                self.register_document(value.split("|", 1)[0].strip())
            case SourceKeySpace.KNOWLEDGE_BASE:
                self.register_knowledge_base(value)
            case SourceKeySpace.WEB:
                parsed = urlparse(value)
                if parsed.scheme in ("http", "https"):
                    self.register_web(value, "")

    # ── Known-text codecs ───────────────────────────────────────────────────

    def decode_known_text(self, text: str) -> str:
        """Restore registered source handles embedded in structured text.

        Only for structured expressions such as a built-in SQL tool argument;
        arbitrary prose must never be rewritten this way.
        """
        if text == "":
            return text
        return _SHORT_SOURCE_HANDLE_IN_TEXT_RE.sub(
            lambda match: self._durable_for_handle(match.group(0)) or match.group(0), text
        )

    def decode_known_quoted_text(self, text: str) -> str:
        """Restore source handles only inside quoted segments.

        Unquoted tokens in structured expressions are left untouched so a
        legitimate table/column handle that happens to look like ``d1`` or
        ``b2`` is never corrupted.
        """
        if text == "":
            return text
        return rewrite_quoted_text(text, lambda segment: self._decode_quoted_segment(segment))

    def _decode_quoted_segment(self, segment: str) -> str:
        return _SHORT_SOURCE_HANDLE_IN_TEXT_RE.sub(
            lambda match: self._durable_for_handle(match.group(0)) or match.group(0), segment
        )

    def unresolved_quoted_text_handles(self, text: str) -> list[str]:
        """Report handle-shaped values inside quoted segments that do not exist."""
        if text == "":
            return []
        seen: set[str] = set()

        def collect(segment: str) -> str:
            for handle in _SHORT_SOURCE_HANDLE_IN_TEXT_RE.findall(segment):
                if self._durable_for_handle(handle) == "":
                    seen.add(handle)
            return segment

        rewrite_quoted_text(text, collect)
        return sorted(seen)

    # ── JSON codec ──────────────────────────────────────────────────────────

    def _decode_json_with_policy(
        self, raw: str, encode: bool, allowed: Callable[[str], bool]
    ) -> str:
        if raw.strip() == "":
            return raw
        value = _loads_json(raw)
        if value is None:
            return raw
        value = self._walk_json("", value, encode, allowed)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _walk_json(
        self,
        key: str,
        value: JsonValue,
        encode: bool,
        allowed: Callable[[str], bool],
    ) -> JsonValue:
        if isinstance(value, str):
            if not allowed(key.lower()):
                return value
            if encode:
                handle = self._handle_for_durable(value)
                if handle != "":
                    return handle
                return value
            if key.lower() not in source_key_spaces:
                return value
            if not _SHORT_SOURCE_HANDLE_RE.match(value.strip()):
                return value
            real = self._durable_for_handle(value)
            if real != "":
                return real
            return value
        if isinstance(value, list):
            return [self._walk_json(key, item, encode, allowed) for item in value]
        if isinstance(value, dict):
            return {
                child_key: self._walk_json(child_key, item, encode, allowed)
                for child_key, item in value.items()
            }
        return value

    def _handle_for_durable(self, real: str) -> str:
        handle = self.chunks.handle_for_key(real)
        if handle is not None:
            return handle
        handle = self.docs.handle_for_key(real)
        if handle is not None:
            return handle
        handle = self.kbs.handle_for_key(real)
        if handle is not None:
            return handle
        handle = self.webs.handle_for_key(canonical_web_url(real))
        if handle is not None:
            return handle
        return ""

    def _durable_for_handle(self, handle: str) -> str:
        handle = handle.lower().strip()
        chunk = self.chunks.resolve(handle)
        if chunk is not None:
            return chunk[0]
        doc = self.docs.resolve(handle)
        if doc is not None:
            return doc[0]
        kb = self.kbs.resolve(handle)
        if kb is not None:
            return kb[0]
        web = self.webs.resolve(handle)
        if web is not None:
            return web[0]
        return ""

    def compact_known_text(self, text: str) -> str:
        """Compact identifiers already registered from structured runtime data.

        The snapshot spans all four source tables and is sorted longest-value
        first globally: a web URL may contain a registered document UUID as a
        substring, so per-table passes could corrupt the longer value.
        """
        if text == "":
            return text
        pairs = [*self.chunks.pairs(), *self.docs.pairs(), *self.kbs.pairs(), *self.webs.pairs()]
        pairs.sort(key=lambda item: len(item.value), reverse=True)
        for item in pairs:
            if item.value != "":
                text = text.replace(item.value, item.handle)
        return text


def _allowed_key(policy: ArgumentPolicy | None, tool_name: str) -> Callable[[str], bool]:
    """Return the per-tool key gate; a nil policy allows every key."""
    return lambda key: policy is None or policy(tool_name, key)


def rewrite_quoted_text(text: str, rewrite: Callable[[str], str]) -> str:
    """Rewrite single/double/backtick-quoted segments of structured text.

    Backslash escapes and doubled quotes (SQL escaping) are kept inside the
    same literal instead of terminating it early.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        quote = text[i]
        if quote not in "'\"`":
            out.append(text[i])
            i += 1
            continue
        start = i
        i += 1
        while i < n:
            if text[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if text[i] != quote:
                i += 1
                continue
            if i + 1 < n and text[i + 1] == quote:
                i += 2
                continue
            i += 1
            break
        out.append(rewrite(text[start:i]))
    return "".join(out)


__all__ = [
    "ChunkReference",
    "SourceRegistry",
    "WebMeta",
    "canonical_web_url",
    "first_non_empty",
    "merge_chunk_reference",
    "rewrite_quoted_text",
]
