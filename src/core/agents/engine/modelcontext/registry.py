"""Registry facade — the single request-scoped boundary between durable
application identities and temporary model handles.

Resource references are encoded before source identifiers: a wiki slug such as
``summary/<knowledge-id>`` must be protected as one resource-like handle before
the embedded document ID can be compacted to ``dN``. Callers therefore cannot
accidentally reverse the codecs.
"""

from __future__ import annotations

import json
from typing import cast

from src.ai.llm.types import ChatResponse, LLMToolCall, Message, SearchResult, ToolCall
from src.common.json import JsonValue
from src.core.agents.engine.modelcontext.citations import (
    RESOURCE_HANDLE_PROTOCOL_PROMPT,
    CitationStreamExpander,
)
from src.core.agents.engine.modelcontext.handles import HandleTable
from src.core.agents.engine.modelcontext.model_output import (
    ToolResult,
    render_model_output,
)
from src.core.agents.engine.modelcontext.resources import ResourceRegistry
from src.core.agents.engine.modelcontext.sources import ChunkReference, SourceRegistry
from src.core.agents.engine.modelcontext.stream import (
    HandleStreamDecoder,
    OrphanResourceStreamFilter,
    ResourceStreamDecoder,
    StreamDecoder,
)
from src.core.agents.engine.modelcontext.tool_policy import (
    ISSUE_HANDLE_SHAPE_RE,
    ToolHandlePolicy,
    rewrite_json_string_values,
    source_argument_allowed,
    source_compaction_allowed,
    source_output_allowed,
    tool_handle_policies,
    walk_json_string_values,
)

ARGUMENT_RESOLUTION_UNCHANGED = "unchanged"
ARGUMENT_RESOLUTION_RESOLVED = "resolved"
ARGUMENT_RESOLUTION_PARTIALLY_RESOLVED = "partially_resolved"
ARGUMENT_RESOLUTION_UNRESOLVED = "unresolved"


def json_equivalent(left: str, right: str) -> bool:
    """Compare two JSON strings by parsed value, falling back to raw equality."""
    try:
        left_value = cast(JsonValue, json.loads(left))
        right_value = cast(JsonValue, json.loads(right))
    except ValueError:
        return left == right
    return left_value == right_value


def unique_sorted(values: list[str]) -> list[str]:
    """Return the distinct non-empty values in sorted order."""
    seen: set[str] = set()
    for value in values:
        if value != "":
            seen.add(value)
    return sorted(seen)


class Registry:
    """Request-scoped boundary between durable values and model handles."""

    def __init__(self, citations_enabled: bool) -> None:
        self._sources = SourceRegistry(citations_enabled)
        self._resources = ResourceRegistry()
        self._issues = HandleTable("i", 0, 1)

    def protocol_prompt(self) -> str:
        """Return the system-owned model handle and citation protocol."""
        return self._sources.protocol_prompt() + RESOURCE_HANDLE_PROTOCOL_PROMPT

    # ── Encode ──────────────────────────────────────────────────────────────

    def encode_messages(self, messages: list[Message]) -> list[Message]:
        """Return a model-facing copy with every temporary handle encoded.

        The scan is intentionally order-independent, matching the source
        codec's two-pass replay behavior.
        """
        messages = self._resources.encode_messages(messages)
        out = list(messages)
        for i, message in enumerate(out):
            if message.role == "tool":
                content = self._encode_tool_private_result(message.name, message.content)
                out[i] = message.model_copy(update={"content": content})
        messages = self._sources.encode_messages_with_policies(
            out, source_argument_allowed, source_output_allowed
        )
        out = list(messages)
        for i, message in enumerate(out):
            if not message.tool_calls:
                continue
            tool_calls = list(message.tool_calls)
            for j, call in enumerate(tool_calls):
                tool_calls[j] = self._encode_replayed_tool_policies(call)
            out[i] = message.model_copy(update={"tool_calls": tool_calls})
        return out

    # ── Decode ──────────────────────────────────────────────────────────────

    def decode_tool_calls(self, tool_calls: list[LLMToolCall]) -> None:
        """Restore all temporary handles in tool-call arguments and classify them."""
        for call in tool_calls:
            if call.model_arguments == "":
                call.model_arguments = call.function.arguments
        self._resources.decode_tool_calls(tool_calls)
        self._sources.decode_tool_calls_with_policy(tool_calls, source_argument_allowed)
        for call in tool_calls:
            self._decode_tool_policies(call)
            resolved = call.function.arguments
            unresolved = self._resources.orphan_handles(resolved)
            unresolved = [
                *unresolved,
                *self._sources.unresolved_tool_handles_with_policy(
                    call.function.name, resolved, source_argument_allowed
                ),
            ]
            unresolved = [
                *unresolved,
                *self._unresolved_private_tool_handles(call.function.name, resolved),
            ]
            call.unresolved_handles = unique_sorted(unresolved)
            changed = not json_equivalent(call.model_arguments, resolved)
            if changed and len(call.unresolved_handles) > 0:
                call.argument_resolution = ARGUMENT_RESOLUTION_PARTIALLY_RESOLVED
            elif len(call.unresolved_handles) > 0:
                call.argument_resolution = ARGUMENT_RESOLUTION_UNRESOLVED
            elif changed:
                call.argument_resolution = ARGUMENT_RESOLUTION_RESOLVED
            else:
                call.argument_resolution = ARGUMENT_RESOLUTION_UNCHANGED

    def decode_response(self, response: ChatResponse) -> None:
        """Restore resources, expand citations, and decode tool arguments."""
        response.content = self.decode_output_text(response.content)
        response.reasoning_content = self.decode_output_text(response.reasoning_content)
        self.decode_tool_calls(response.tool_calls)

    def stream_decoder(self) -> StreamDecoder:
        """Create one ordered decoder for a response text channel."""
        return StreamDecoder(
            resources=ResourceStreamDecoder(self._resources),
            sources=CitationStreamExpander(self._sources),
            issues=HandleStreamDecoder(self._issues),
            orphans=OrphanResourceStreamFilter(),
        )

    def orphan_resource_handles(self, decoded: str) -> list[str]:
        """Report model-generated resource handles with no backing reference."""
        return self._resources.orphan_handles(decoded)

    # ── Registration ────────────────────────────────────────────────────────

    def register_chunk(self, ref: ChunkReference) -> str:
        return self._sources.register_chunk(ref)

    def register_document(self, id: str) -> str:
        return self._sources.register_document(id)

    def register_knowledge_base(self, id: str) -> str:
        return self._sources.register_knowledge_base(id)

    def register_web(self, raw_url: str, title: str) -> str:
        return self._sources.register_web(raw_url, title)

    def register_search_results(self, results: list[SearchResult]) -> None:
        self._sources.register_search_results(results)

    def chunk_handle(self, id: str) -> str:
        return self._sources.chunk_handle(id)

    def compact_known_text(self, text: str) -> str:
        """Replace only previously registered durable source IDs."""
        text = self._resources.encode_text(text)
        return self._sources.compact_known_text(text)

    # ── Tool-result rendering ───────────────────────────────────────────────

    def model_tool_result(self, result: ToolResult) -> str:
        """Render a tool result using registered model handles."""
        return self.model_tool_result_for_tool("", result)

    def model_tool_result_for_tool(self, tool_name: str, result: ToolResult) -> str:
        """Render a result, applying any explicit private-ID policy for the tool.

        Error text is encoded on the same path as output: a failed tool call
        routinely echoes the offending argument, so a raw durable ID would leak
        through the error branch of an otherwise handle-only tool.
        """
        copy = result.model_copy(
            update={
                "output": self._resources.encode_text(
                    self._encode_tool_private_result(tool_name, result.output)
                ),
                "error": self._resources.encode_text(
                    self._encode_tool_private_result(tool_name, result.error)
                ),
            }
        )
        if source_output_allowed(tool_name):
            rendered = render_model_output(self._sources, copy)
        elif copy.success:
            rendered = copy.output
        elif copy.error != "":
            rendered = "Error: " + copy.error
        else:
            rendered = "Error: tool call failed"
        if source_compaction_allowed(tool_name):
            rendered = self._sources.compact_known_text(rendered)
        return self._resources.encode_text(rendered)

    def decode_output_text(self, text: str) -> str:
        """Apply the public citation policy to complete text.

        Primarily used by non-streaming cleanup/fallback paths.
        """
        text = self._resources.decode_text(text)
        text = self._resources.strip_orphan_handles(text)
        text = self._sources.expand_text(text)
        return self._issues.decode_known_text(text)

    # ── Tool-private policy application ─────────────────────────────────────

    def _encode_tool_private_result(self, tool_name: str, output: str) -> str:
        if output == "":
            return output
        policy = tool_handle_policies.get(tool_name)
        if policy is None:
            return output
        if policy.encoded_issue_id_keys:
            output = rewrite_json_string_values(
                output,
                lambda key, value: self._encode_issue_id_value(policy, key, value),
            )
        if policy.encode_known_issue_ids:
            output = self._issues.encode_known_text(output)
        return output

    def _encode_issue_id_value(self, policy: ToolHandlePolicy, key: str, value: str) -> str:
        if key in policy.encoded_issue_id_keys and value.strip() != "":
            # Model-facing tool messages can be replayed across several rounds.
            # Never treat an existing temporary handle as a new durable issue
            # identity and allocate i2/i3 drift.
            if ISSUE_HANDLE_SHAPE_RE.match(value.strip()):
                return value
            return self._issues.register(value)
        return value

    def _encode_replayed_tool_policies(self, call: ToolCall) -> ToolCall:
        policy = tool_handle_policies.get(call.function.name)
        if policy is None:
            return call
        arguments = rewrite_json_string_values(
            call.function.arguments,
            lambda key, value: self._encode_replayed_value(policy, key, value),
        )
        return call.model_copy(
            update={"function": call.function.model_copy(update={"arguments": arguments})}
        )

    def _encode_replayed_value(self, policy: ToolHandlePolicy, key: str, value: str) -> str:
        if key in policy.source_text_keys:
            return self._sources.compact_known_text(value)
        if key in policy.decoded_issue_id_keys:
            return self._issues.encode_known_text(value)
        return value

    def _decode_tool_policies(self, call: LLMToolCall) -> None:
        policy = tool_handle_policies.get(call.function.name)
        if policy is None:
            return
        arguments = rewrite_json_string_values(
            call.function.arguments,
            lambda key, value: self._decode_policy_value(policy, key, value),
        )
        call.function = call.function.model_copy(update={"arguments": arguments})

    def _decode_policy_value(self, policy: ToolHandlePolicy, key: str, value: str) -> str:
        if key in policy.source_text_keys:
            return self._sources.decode_known_quoted_text(value)
        if key in policy.decoded_issue_id_keys and ISSUE_HANDLE_SHAPE_RE.match(value.strip()):
            resolved = self._issues.resolve(value)
            if resolved is not None:
                return resolved
        return value

    def _unresolved_private_tool_handles(self, tool_name: str, raw: str) -> list[str]:
        if raw == "":
            return []
        policy = tool_handle_policies.get(tool_name)
        if policy is None or not (policy.decoded_issue_id_keys or policy.source_text_keys):
            return []
        unresolved: list[str] = []

        def rewrite(key: str, value: str) -> str:
            value = value.strip()
            if key in policy.source_text_keys:
                unresolved.extend(self._sources.unresolved_quoted_text_handles(value))
            if (
                key in policy.decoded_issue_id_keys
                and ISSUE_HANDLE_SHAPE_RE.match(value)
                and (self._issues.resolve(value) is None)
            ):
                unresolved.append(value)
            return value

        walk_json_string_values(raw, rewrite)
        return unresolved


__all__ = [
    "ARGUMENT_RESOLUTION_PARTIALLY_RESOLVED",
    "ARGUMENT_RESOLUTION_RESOLVED",
    "ARGUMENT_RESOLUTION_UNCHANGED",
    "ARGUMENT_RESOLUTION_UNRESOLVED",
    "Registry",
    "json_equivalent",
    "unique_sorted",
]
