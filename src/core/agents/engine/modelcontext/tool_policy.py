"""Per-tool policy layer: which JSON keys are ID-bearing and how handles decode.

``source_key_spaces`` is the single table of ID-bearing keys the source codec
understands. It drives both handle registration and the handle-shaped-value
decode gate, so registration and decoding cannot drift apart. Per-tool
``source_id_keys`` contracts gate which of those keys each built-in tool may
use; dynamic tools remain fully opaque.

The JSON rewrite helpers operate on parsed JSON values and re-serialize with
compact separators, mirroring the reference implementation's marshal output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias, cast

from src.common.json import JsonValue

TOOL_DATABASE_QUERY = "database_query"
TOOL_DATA_ANALYSIS = "data_analysis"
TOOL_WIKI_READ_ISSUE = "wiki_read_issue"
TOOL_WIKI_UPDATE_ISSUE = "wiki_update_issue"

ISSUE_HANDLE_SHAPE_RE = re.compile(r"^i[1-9][0-9]*$")


class SourceKeySpace(IntEnum):
    """Identifies which source-handle space an ID-bearing JSON key belongs to."""

    CHUNK = 0
    DOCUMENT = 1
    DOCUMENT_REF = 2  # "knowledgeID|title" stored refs; only the ID is durable
    KNOWLEDGE_BASE = 3
    WEB = 4


source_key_spaces: dict[str, SourceKeySpace] = {
    "chunk_id": SourceKeySpace.CHUNK,
    "faq_id": SourceKeySpace.CHUNK,
    "chunk_ids": SourceKeySpace.CHUNK,
    "faq_ids": SourceKeySpace.CHUNK,
    "knowledge_id": SourceKeySpace.DOCUMENT,
    "knowledge_ids": SourceKeySpace.DOCUMENT,
    "suspected_knowledge_ids": SourceKeySpace.DOCUMENT,
    "source_refs": SourceKeySpace.DOCUMENT_REF,
    "knowledge_base": SourceKeySpace.KNOWLEDGE_BASE,
    "knowledge_base_id": SourceKeySpace.KNOWLEDGE_BASE,
    "knowledge_base_ids": SourceKeySpace.KNOWLEDGE_BASE,
    "kb_id": SourceKeySpace.KNOWLEDGE_BASE,
    "kb_ids": SourceKeySpace.KNOWLEDGE_BASE,
    "url": SourceKeySpace.WEB,
    "urls": SourceKeySpace.WEB,
}


@dataclass(frozen=True)
class ToolHandlePolicy:
    """Model-handle contract for one built-in tool family."""

    source_id_keys: frozenset[str] = frozenset()
    source_text_keys: frozenset[str] = frozenset()
    source_output: bool = False
    decoded_issue_id_keys: frozenset[str] = frozenset()
    encoded_issue_id_keys: frozenset[str] = frozenset()
    encode_known_issue_ids: bool = False


#: Complete allowlist for built-in tool fields whose identifiers need more than
#: the generic resource/source codecs. Field names alone are deliberately
#: insufficient: a dynamic tool may use the same name with unrelated semantics.
tool_handle_policies: dict[str, ToolHandlePolicy] = {
    "knowledge_search": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_base_ids"}), source_output=True
    ),
    "grep_chunks": ToolHandlePolicy(source_output=True),
    "list_knowledge_chunks": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_id", "faq_id", "chunk_id"}), source_output=True
    ),
    "get_document_info": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_ids", "faq_ids"}), source_output=True
    ),
    "query_knowledge_graph": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_base_ids"}), source_output=True
    ),
    TOOL_DATABASE_QUERY: ToolHandlePolicy(source_text_keys=frozenset({"sql"}), source_output=True),
    TOOL_DATA_ANALYSIS: ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_id"}), source_text_keys=frozenset({"sql"})
    ),
    "data_schema": ToolHandlePolicy(source_id_keys=frozenset({"knowledge_id"})),
    "web_fetch": ToolHandlePolicy(source_id_keys=frozenset({"url", "urls"}), source_output=True),
    "web_search": ToolHandlePolicy(source_output=True),
    "wiki_read_page": ToolHandlePolicy(source_output=True),
    "wiki_read_source_doc": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_id"}), source_output=True
    ),
    "wiki_write_page": ToolHandlePolicy(source_id_keys=frozenset({"source_refs"})),
    "wiki_replace_text": ToolHandlePolicy(source_id_keys=frozenset({"source_refs"})),
    "wiki_flag_issue": ToolHandlePolicy(source_id_keys=frozenset({"suspected_knowledge_ids"})),
    "wiki_search": ToolHandlePolicy(
        source_id_keys=frozenset({"knowledge_base_id"}), source_output=True
    ),
    TOOL_WIKI_READ_ISSUE: ToolHandlePolicy(
        decoded_issue_id_keys=frozenset({"issue_id"}),
        encoded_issue_id_keys=frozenset({"id"}),
        encode_known_issue_ids=True,
        source_output=True,
    ),
    TOOL_WIKI_UPDATE_ISSUE: ToolHandlePolicy(
        decoded_issue_id_keys=frozenset({"issue_id"}), encode_known_issue_ids=True
    ),
    # Mutation tools echo the slug they acted on and surface validation errors
    # that can quote a durable knowledge-base or document ID. They own no source
    # arguments, but their output must still be compacted before the model sees
    # it.
    "wiki_rename_page": ToolHandlePolicy(source_output=True),
    "wiki_delete_page": ToolHandlePolicy(source_output=True),
    # Agent-private bookkeeping tools echo model-authored text. They need
    # compaction so a durable ID quoted back by the model is re-compacted, but
    # never structured source rendering.
    "thinking": ToolHandlePolicy(),
    "todo_write": ToolHandlePolicy(),
    # Skill output is user-authored content of unknown shape. Compact known
    # durable IDs, but do not mine it for source keys: a skill's JSON may use
    # "url"/"knowledge_id" with unrelated semantics.
    "read_skill": ToolHandlePolicy(),
    "execute_skill_script": ToolHandlePolicy(),
}


def has_tool_policy(tool_name: str) -> bool:
    """Report whether a tool declares an explicit model-handle policy.

    Built-in tools must declare one (even an empty policy) so that adding a
    tool is a deliberate decision about what the model may see, rather than a
    silent fall-through to fully opaque output.
    """
    return tool_name in tool_handle_policies


def source_argument_allowed(tool_name: str, key: str) -> bool:
    """Return whether ``key`` is a declared source-ID argument of ``tool_name``."""
    policy = tool_handle_policies.get(tool_name)
    if policy is None:
        return False
    return key.lower() in policy.source_id_keys


def source_output_allowed(tool_name: str) -> bool:
    """Return whether ``tool_name`` gets structured source rendering."""
    if tool_name == "":
        return True
    policy = tool_handle_policies.get(tool_name)
    return policy is not None and policy.source_output


def source_compaction_allowed(tool_name: str) -> bool:
    """Return whether ``tool_name`` output may be compacted after rendering."""
    if tool_name == "":
        return True
    return tool_name in tool_handle_policies


ArgumentPolicy: TypeAlias = Callable[[str, str], bool]
ResultPolicy: TypeAlias = Callable[[str], bool]


def _loads_json(raw: str) -> JsonValue | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return cast(JsonValue, value)


def walk_json_value(key: str, value: JsonValue, rewrite: Callable[[str, str], str]) -> JsonValue:
    """Apply ``rewrite`` to every JSON string value, keyed by its field path."""
    if isinstance(value, str):
        return rewrite(key.lower(), value)
    if isinstance(value, list):
        return [walk_json_value(key, item, rewrite) for item in value]
    if isinstance(value, dict):
        return {
            child_key: walk_json_value(child_key, item, rewrite)
            for child_key, item in value.items()
        }
    return value


def rewrite_json_string_values(raw: str, rewrite: Callable[[str, str], str]) -> str:
    """Rewrite JSON string values, returning the original when input is not JSON."""
    value = _loads_json(raw)
    if value is None:
        return raw
    value = walk_json_value("", value, rewrite)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def walk_json_string_values(raw: str, rewrite: Callable[[str, str], str]) -> None:
    """Walk JSON string values purely for side effects; no-op on non-JSON input."""
    value = _loads_json(raw)
    if value is None:
        return
    walk_json_value("", value, rewrite)


__all__ = [
    "TOOL_DATABASE_QUERY",
    "TOOL_DATA_ANALYSIS",
    "TOOL_WIKI_READ_ISSUE",
    "TOOL_WIKI_UPDATE_ISSUE",
    "ArgumentPolicy",
    "ResultPolicy",
    "SourceKeySpace",
    "ToolHandlePolicy",
    "has_tool_policy",
    "rewrite_json_string_values",
    "source_argument_allowed",
    "source_compaction_allowed",
    "source_key_spaces",
    "source_output_allowed",
    "tool_handle_policies",
    "walk_json_string_values",
    "walk_json_value",
]
