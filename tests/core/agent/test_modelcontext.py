"""Unit tests for the agent model context.

Covers the handle table, source/resource codecs, citation compaction and
stream expansion, per-tool argument decoding, and the registry facade.
No database is required.
"""

from __future__ import annotations

import json

from src.ai.llm.types import (
    ChatResponse,
    FunctionCall,
    LLMToolCall,
    Message,
    ToolCall,
)
from src.core.agents.engine.modelcontext import (
    ARGUMENT_RESOLUTION_RESOLVED,
    ARGUMENT_RESOLUTION_UNCHANGED,
    ARGUMENT_RESOLUTION_UNRESOLVED,
    ChunkReference,
    HandleTable,
    Registry,
    SourceRegistry,
    ToolResult,
)
from src.core.agents.engine.modelcontext.citations import CitationStreamExpander
from src.core.agents.engine.modelcontext.model_output import render_model_output
from src.core.agents.engine.modelcontext.resources import ResourceRegistry
from src.core.agents.engine.modelcontext.sources import rewrite_quoted_text
from src.core.agents.engine.modelcontext.stream import (
    HandleStreamDecoder,
    OrphanResourceStreamFilter,
    ResourceStreamDecoder,
)

# ── HandleTable (exported wrapper) ───────────────────────────────────────


def test_handle_table_register_resolve() -> None:
    table = HandleTable("i", 0, 1)
    assert table.register("issue-42") == "i1"
    assert table.register("issue-42") == "i1"
    assert table.register("issue-43") == "i2"
    assert table.handle("issue-42") == "i1"
    assert table.handle("missing") is None
    assert table.resolve("i1") == "issue-42"
    assert table.resolve("i9") is None
    assert table.len() == 2
    assert table.empty() is False


def test_handle_table_register_blank() -> None:
    table = HandleTable("i", 0, 1)
    assert table.register("") == ""
    assert table.register("  ") == ""
    assert table.len() == 0


def test_handle_table_zero_padded() -> None:
    table = HandleTable("c", 3, 0)
    assert table.register("a") == "c000"
    assert table.register("b") == "c001"


def test_handle_table_encode_decode_known_text() -> None:
    table = HandleTable("i", 0, 1)
    table.register("issue-42")
    table.register("issue-7")
    assert table.encode_known_text("issue-42 and issue-7") == "i1 and i2"
    assert table.decode_known_text("i1 and i2") == "issue-42 and issue-7"


def test_handle_table_decode_word_boundary() -> None:
    table = HandleTable("i", 0, 1)
    table.register("issue-42")
    # "i1" inside "fi1e" must not be replaced.
    assert table.decode_known_text("see i1, not fi1e or i12") == "see issue-42, not fi1e or i12"


# ── SourceRegistry registration ──────────────────────────────────────────


def test_source_registry_handle_shapes() -> None:
    registry = SourceRegistry(citations_enabled=True)
    assert registry.register_document("kid-1") == "d1"
    assert registry.register_knowledge_base("kb-1") == "b1"
    assert registry.register_chunk(ChunkReference(chunk_id="chunk-1", knowledge_id="kid-1")) == "c1"
    assert registry.register_web("https://example.com/page", "Title") == "w1"
    assert registry.count() == 2  # one chunk + one web


def test_register_chunk_merges_metadata() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_chunk(ChunkReference(chunk_id="chunk-1", knowledge_id="kid-1"))
    registry.register_chunk(
        ChunkReference(chunk_id="chunk-1", knowledge_base_id="kb-1", document_title="Doc")
    )
    value, meta = registry.chunks.resolve("c1")
    assert value == "chunk-1"
    assert meta.knowledge_id == "kid-1"
    assert meta.knowledge_base_id == "kb-1"
    assert meta.document_title == "Doc"


def test_register_web_dedup_on_canonical_url() -> None:
    registry = SourceRegistry(citations_enabled=True)
    first = registry.register_web("https://Example.com/page", "First")
    second = registry.register_web("https://example.com/page#fragment", "")
    assert first == second == "w1"
    # The raw URL originally shown is what decodes back.
    assert registry.webs.resolve("w1")[0] == "https://Example.com/page"


def test_register_handle_shaped_echo() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_chunk(ChunkReference(chunk_id="chunk-1"))
    # A handle-shaped input is echoed back only when that handle already exists.
    assert registry.register_chunk(ChunkReference(chunk_id="c1")) == "c1"
    assert registry.register_chunk(ChunkReference(chunk_id="c9")) == ""
    assert registry.register_document("d1") == ""
    assert registry.register_document("") == ""


def test_chunk_handle_lookup() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_chunk(ChunkReference(chunk_id="chunk-1"))
    assert registry.chunk_handle("chunk-1") == "c1"
    assert registry.chunk_handle("missing") == ""


# ── Known-text compaction and decoding ───────────────────────────────────


def test_compact_known_text_longest_first() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_document("kid-1")
    registry.register_web("https://example.com/kid-1", "")
    text = "see https://example.com/kid-1 and kid-1"
    compacted = registry.compact_known_text(text)
    # The URL contains the document UUID; the longer URL wins first.
    assert compacted == "see w1 and d1"


def test_decode_known_text_structured() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_document("kid-1")
    registry.register_knowledge_base("kb-1")
    assert registry.decode_known_text("WHERE knowledge_id = 'd1' AND kb = 'b1'") == (
        "WHERE knowledge_id = 'kid-1' AND kb = 'kb-1'"
    )
    # Unknown handle-shaped tokens stay untouched.
    assert registry.decode_known_text("d9") == "d9"


def test_decode_known_quoted_text_only_quotes() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_document("kid-1")
    sql = "SELECT 'd1' AS alias, d1 FROM docs"
    assert registry.decode_known_quoted_text(sql) == "SELECT 'kid-1' AS alias, d1 FROM docs"
    assert registry.unresolved_quoted_text_handles("WHERE id IN ('d9')") == ["d9"]


def test_rewrite_quoted_text_respects_escapes() -> None:
    assert rewrite_quoted_text("a 'it\\'s' b", lambda s: s.upper()) == "a 'IT\\'S' b"
    assert rewrite_quoted_text("x '' y", lambda s: s.upper()) == "x '' y"


# ── Citations: compaction and expansion ──────────────────────────────────


def test_compact_public_citations() -> None:
    registry = SourceRegistry(citations_enabled=True)
    text = '<kb doc="Doc" chunk_id="chunk-1" kb_id="kb-1" /> and <web url="https://example.com" title="Ex" />'
    compacted = registry.compact_public_citations(text)
    assert '<ref id="c1"/>' in compacted
    assert '<ref id="w1"/>' in compacted
    # Both handles were registered from the compacted tags.
    assert registry.chunks.resolve("c1")[0] == "chunk-1"
    assert registry.webs.resolve("w1")[0] == "https://example.com"


def test_expand_text_citations_enabled() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_chunk(
        ChunkReference(chunk_id="chunk-1", knowledge_id="kid-1", knowledge_base_id="kb-1")
    )
    registry.register_web("https://example.com", "Ex")
    assert registry.expand_text('<ref id="c1"/>') == '<kb doc="" chunk_id="chunk-1" kb_id="kb-1" />'
    assert registry.expand_text('<ref id="w1"/>') == '<web url="https://example.com" title="Ex" />'
    # Unknown handles fail closed.
    assert registry.expand_text('<ref id="c9"/>') == ""
    # Model-written public tags are dropped before canonical expansion.
    assert (
        registry.expand_text('<kb x="1" /> <ref id="c1"/>')
        == ' <kb doc="" chunk_id="chunk-1" kb_id="kb-1" />'
    )


def test_expand_text_citations_disabled_drops_refs() -> None:
    registry = SourceRegistry(citations_enabled=False)
    registry.register_chunk(ChunkReference(chunk_id="chunk-1"))
    assert registry.expand_text('text <ref id="c1"/> more') == "text  more"
    assert registry.expand_text('<kb x="1" />') == ""


def test_protocol_prompts() -> None:
    enabled = Registry(citations_enabled=True).protocol_prompt()
    assert "Source handling protocol" in enabled
    assert '<ref id="cN"' in enabled
    disabled = Registry(citations_enabled=False).protocol_prompt()
    assert "citations are disabled" in disabled
    assert "res://NNNN" in disabled


# ── Citation stream expander ─────────────────────────────────────────────


def test_citation_stream_expander_across_chunks() -> None:
    registry = SourceRegistry(citations_enabled=True)
    registry.register_chunk(
        ChunkReference(chunk_id="chunk-1", knowledge_id="kid-1", document_title="Doc")
    )
    expander = CitationStreamExpander(registry)
    out = expander.feed('plain text <ref id="c')
    out += expander.feed('1"/> done')
    out += expander.flush()
    assert out == 'plain text <kb doc="Doc" chunk_id="chunk-1" /> done'


def test_citation_stream_expander_holds_partial_prefix() -> None:
    registry = SourceRegistry(citations_enabled=True)
    expander = CitationStreamExpander(registry)
    assert expander.feed("just a <") == "just a "
    assert expander.feed('ref id="c9"/>') == ""
    assert expander.flush() == ""


def test_citation_stream_expander_plain_prose_passthrough() -> None:
    registry = SourceRegistry(citations_enabled=True)
    expander = CitationStreamExpander(registry)
    assert expander.feed("no tags here") == "no tags here"
    assert expander.flush() == ""


# ── Resource codec ───────────────────────────────────────────────────────


_STORED_REF = "resource://" + "A" * 22


def test_resource_registry_round_trip() -> None:
    registry = ResourceRegistry()
    encoded = registry.encode_text(f"see {_STORED_REF} and nothing")
    assert encoded == "see res://0001 and nothing"
    assert registry.decode_text(encoded) == f"see {_STORED_REF} and nothing"
    assert registry.orphan_handles(registry.decode_text("res://0009")) == ["res://0009"]


def test_resource_registry_strip_orphans() -> None:
    registry = ResourceRegistry()
    assert registry.strip_orphan_handles("x res://0003 y") == "x  y"


def test_resource_registry_known_handles_survive_orphan_scan() -> None:
    registry = ResourceRegistry()
    registry.encode_text(_STORED_REF)
    assert registry.orphan_handles("res://0001") == []


# ── Registry: encode_messages ────────────────────────────────────────────


def test_encode_messages_tool_private_issue_handles() -> None:
    registry = Registry(citations_enabled=True)
    tool_message = Message(
        role="tool",
        name="wiki_read_issue",
        content='{"id": "issue-42", "status": "open"}',
    )
    encoded = registry.encode_messages([tool_message])
    payload = json.loads(encoded[0].content)
    assert payload["id"] == "i1"
    assert registry._issues.resolve("i1") == "issue-42"


def test_encode_messages_replayed_tool_call() -> None:
    registry = Registry(citations_enabled=True)
    tool_message = Message(role="tool", name="wiki_read_issue", content='{"id": "issue-42"}')
    assistant = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="wiki_read_issue", arguments='{"issue_id": "issue-42"}'),
            )
        ],
    )
    encoded = registry.encode_messages([tool_message, assistant])
    args = json.loads(encoded[1].tool_calls[0].function.arguments)
    assert args["issue_id"] == "i1"


def test_encode_messages_compacts_tool_arguments() -> None:
    registry = Registry(citations_enabled=True)
    assistant = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(
                    name="web_fetch", arguments='{"urls": ["https://example.com/page"]}'
                ),
            )
        ],
    )
    encoded = registry.encode_messages([assistant])
    args = json.loads(encoded[0].tool_calls[0].function.arguments)
    assert args["urls"] == ["w1"]


# ── Registry: decode_tool_calls ──────────────────────────────────────────


def _decoded_call(registry: Registry, name: str, arguments: str) -> LLMToolCall:
    call = LLMToolCall(function=FunctionCall(name=name, arguments=arguments))
    registry.decode_tool_calls([call])
    return call


def test_decode_tool_calls_resolves_source_handles() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_knowledge_base("kb-123")
    call = _decoded_call(registry, "knowledge_search", '{"knowledge_base_ids": ["b1"]}')
    assert json.loads(call.function.arguments)["knowledge_base_ids"] == ["kb-123"]
    assert call.argument_resolution == ARGUMENT_RESOLUTION_RESOLVED
    assert call.unresolved_handles == []


def test_decode_tool_calls_reports_unresolved() -> None:
    registry = Registry(citations_enabled=True)
    call = _decoded_call(registry, "knowledge_search", '{"knowledge_base_ids": ["b9"]}')
    assert call.unresolved_handles == ["b9"]
    assert call.argument_resolution == ARGUMENT_RESOLUTION_UNRESOLVED


def test_decode_tool_calls_unchanged_without_handles() -> None:
    registry = Registry(citations_enabled=True)
    call = _decoded_call(registry, "knowledge_search", '{"knowledge_base_ids": ["kb-9"]}')
    assert call.argument_resolution == ARGUMENT_RESOLUTION_UNCHANGED
    assert json.loads(call.function.arguments)["knowledge_base_ids"] == ["kb-9"]


def test_decode_tool_calls_private_issue() -> None:
    registry = Registry(citations_enabled=True)
    registry._issues.register("issue-42")  # i1
    call = _decoded_call(registry, "wiki_read_issue", '{"issue_id": "i1"}')
    assert json.loads(call.function.arguments)["issue_id"] == "issue-42"
    assert call.unresolved_handles == []


def test_decode_tool_calls_sql_text_quoted() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_document("kid-1")
    call = _decoded_call(
        registry,
        "data_analysis",
        '{"knowledge_id": "d1", "sql": "SELECT \'d1\' FROM t"}',
    )
    args = json.loads(call.function.arguments)
    assert args["knowledge_id"] == "kid-1"
    assert args["sql"] == "SELECT 'kid-1' FROM t"


# ── Registry: tool-result rendering ──────────────────────────────────────


def test_model_tool_result_knowledge_rendering() -> None:
    registry = Registry(citations_enabled=True)
    result = ToolResult(
        success=True,
        output="fallback",
        data={
            "display_type": "search_results",
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "knowledge_id": "kid-1",
                    "knowledge_title": "Doc",
                    "content": "body",
                }
            ],
        },
    )
    rendered = registry.model_tool_result_for_tool("knowledge_search", result)
    assert rendered == (
        '<retrieval type="knowledge" mode="semantic">\n'
        '  <document id="d1" title="Doc">\n'
        '    <chunk id="c1" index="0" view="full">\n'
        "      <content>body</content>\n"
        "    </chunk>\n"
        "  </document>\n"
        "</retrieval>"
    )


def test_model_tool_result_error_compacts_durable_id() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_document("kid-1")
    result = ToolResult(success=False, output="", error="failed to load kid-1")
    rendered = registry.model_tool_result_for_tool("knowledge_search", result)
    assert rendered == "Error: failed to load d1"


def test_model_tool_result_opaque_for_dynamic_tool() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_document("kid-1")
    result = ToolResult(success=True, output="mentions kid-1", data={"display_type": "other"})
    # Dynamic MCP output stays fully opaque: no compaction, no source rendering.
    rendered = registry.model_tool_result_for_tool("mcp_custom", result)
    assert rendered == "mentions kid-1"


def test_model_tool_result_generic_facade() -> None:
    registry = Registry(citations_enabled=True)
    result = ToolResult(success=True, output="plain")
    assert registry.model_tool_result(result) == "plain"


def test_render_web_search_output() -> None:
    registry = SourceRegistry(citations_enabled=True)
    result = ToolResult(
        success=True,
        output="fb",
        data={
            "display_type": "web_search_results",
            "results": [{"url": "https://example.com", "title": "Ex", "snippet": "snip"}],
        },
    )
    rendered = render_model_output(registry, result)
    assert 'mode="search"' in rendered
    assert '<page id="w1" title="Ex">' in rendered
    assert "<match>snip</match>" in rendered


# ── Registry: decode_response and compact ────────────────────────────────


def test_decode_response_expands_citations() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_chunk(ChunkReference(chunk_id="chunk-1", knowledge_id="kid-1"))
    response = ChatResponse(content='answer <ref id="c1"/>')
    registry.decode_response(response)
    assert response.content == 'answer <kb doc="" chunk_id="chunk-1" />'


def test_compact_known_text_via_registry() -> None:
    registry = Registry(citations_enabled=True)
    registry.register_document("kid-1")
    assert registry.compact_known_text("load kid-1") == "load d1"


def test_orphan_resource_handles() -> None:
    registry = Registry(citations_enabled=True)
    assert registry.orphan_resource_handles("x res://0007 y") == ["res://0007"]


# ── Stream decoder composition ───────────────────────────────────────────


def test_stream_decoder_restores_resources_across_chunks() -> None:
    registry = Registry(citations_enabled=True)
    encoded = registry.compact_known_text(f"see {_STORED_REF} now")
    decoder = registry.stream_decoder()
    out = decoder.feed(encoded[:8])
    out += decoder.feed(encoded[8:16])
    out += decoder.feed(encoded[16:])
    out += decoder.flush()
    assert out == f"see {_STORED_REF} now"


def test_stream_decoder_drops_unknown_resource_handles() -> None:
    registry = Registry(citations_enabled=True)
    decoder = registry.stream_decoder()
    out = decoder.feed("text res://")
    out += decoder.feed("9999 text")
    out += decoder.flush()
    assert out == "text  text"


def test_resource_stream_decoder_split_handle() -> None:
    registry = ResourceRegistry()
    registry.encode_text(_STORED_REF)
    decoder = ResourceStreamDecoder(registry)
    out = decoder.feed("x res:")
    out += decoder.feed("//")
    out += decoder.feed("0001 y")
    assert out == f"x {_STORED_REF} y"


def test_handle_stream_decoder_split() -> None:
    table = HandleTable("i", 0, 1)
    table.register("issue-42")
    decoder = HandleStreamDecoder(table)
    out = decoder.feed("issue i")
    out += decoder.feed("1 now")
    out += decoder.flush()
    assert out == "issue issue-42 now"


def test_orphan_filter_holds_partial_handle() -> None:
    filter_ = OrphanResourceStreamFilter()
    assert filter_.feed("keep res:") == "keep "
    assert filter_.feed("//5 end") == " end"
    assert filter_.flush() == ""


def test_orphan_filter_preserves_prose() -> None:
    filter_ = OrphanResourceStreamFilter()
    assert filter_.feed("re") == ""
    assert filter_.flush() == "re"
