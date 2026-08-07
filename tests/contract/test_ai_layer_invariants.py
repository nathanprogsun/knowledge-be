"""AI-layer frozen-contract invariants.

Compares the AI-layer data models against the expected upstream contract
(shape, field names, JSON serialization names, enum values). The reference
values are derived from the read-only upstream Go source in the workspace
root and the frozen contract documentation in the agent workspace.

These checks are intentionally read-only and model-only — they exercise
no I/O, no SDK calls, no network. They serve as the milestone gate that
blocks any future drift between the Python port and the contract.
"""

from __future__ import annotations

import pytest

from src.ai.llm.types import (
    ChatResponse,
    LLMToolCall,
    ResponseType,
    StreamResponse,
    TokenUsage,
)
from src.ai.provider.registry import ALL_PROVIDERS, PROVIDER_ANTHROPIC
from src.ai.retrieval.types import RetrieverEngineType


# ---------------------------------------------------------------------------
# LLM wire contracts — mirror `internal/types/chat.go`
# ---------------------------------------------------------------------------


class TestChatResponseContract:
    def test_required_fields_present(self) -> None:
        fields = set(ChatResponse.model_fields.keys())
        for required in ("content", "finish_reason", "usage"):
            assert required in fields, f"ChatResponse missing {required}"

    def test_no_alias_name_drift(self) -> None:
        # JSON serialization names mirror the Go field tags exactly.
        for name, field in ChatResponse.model_fields.items():
            assert field.alias is None or field.alias == name, (
                f"ChatResponse.{name} has unexpected alias {field.alias}"
            )


class TestStreamResponseContract:
    def test_required_fields_present(self) -> None:
        fields = set(StreamResponse.model_fields.keys())
        for required in ("response_type", "content", "done"):
            assert required in fields, f"StreamResponse missing {required}"

    def test_response_type_values(self) -> None:
        actual = {member.value for member in ResponseType}
        expected = {
            "answer",
            "references",
            "thinking",
            "tool_call",
            "tool_result",
            "error",
            "reflection",
            "session_title",
            "agent_query",
            "complete",
            "tool_approval_required",
            "tool_approval_resolved",
            "mcp_oauth_required",
            "mcp_oauth_resolved",
        }
        assert actual == expected, f"ResponseType drift: missing={expected-actual} extra={actual-expected}"


class TestTokenUsageContract:
    def test_required_fields_present(self) -> None:
        fields = set(TokenUsage.model_fields.keys())
        for required in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_miss_tokens",
        ):
            assert required in fields, f"TokenUsage missing {required}"


class TestLLMToolCallContract:
    def test_required_fields_present(self) -> None:
        fields = set(LLMToolCall.model_fields.keys())
        for required in ("id", "type", "function"):
            assert required in fields, f"LLMToolCall missing {required}"


# ---------------------------------------------------------------------------
# Provider registry — mirror `internal/models/provider/provider.go`
# ---------------------------------------------------------------------------


EXPECTED_PROVIDER_VALUES = (
    "generic",
    "weknoracloud",
    "aliyun",
    "zhipu",
    "volcengine",
    "hunyuan",
    "siliconflow",
    "deepseek",
    "minimax",
    "moonshot",
    "modelscope",
    "qianfan",
    "qiniu",
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "requesty",
    "jina",
    "mimo",
    "longcat",
    "lkeap",
    "gpustack",
    "nvidia",
    "novita",
    "azure_openai",
)


class TestProviderRegistryContract:
    def test_provider_count(self) -> None:
        assert len(ALL_PROVIDERS) == 26, (
            f"ProviderName count drift: got {len(ALL_PROVIDERS)}, expected 26"
        )

    def test_provider_values_in_order(self) -> None:
        actual = tuple(ALL_PROVIDERS)
        assert actual == EXPECTED_PROVIDER_VALUES, (
            f"ProviderName drift (order matters — used by chat/embedding/rerank routing):\n"
            f"  missing: {set(EXPECTED_PROVIDER_VALUES) - set(actual)}\n"
            f"  extra:   {set(actual) - set(EXPECTED_PROVIDER_VALUES)}"
        )

    def test_anthropic_constant(self) -> None:
        assert PROVIDER_ANTHROPIC == "anthropic"


# ---------------------------------------------------------------------------
# Retrieval engine types — mirror `internal/types/retriever.go`
# ---------------------------------------------------------------------------


EXPECTED_RETRIEVER_ENGINE_TYPES = {
    "postgres",
    "elasticsearch",
    "qdrant",
    "milvus",
    "weaviate",
    "doris",
    "sqlite",
    "tencent_vectordb",
    "opensearch",
}


class TestRetrieverEngineTypeContract:
    def test_required_values_present(self) -> None:
        actual = {member.value for member in RetrieverEngineType}
        missing = EXPECTED_RETRIEVER_ENGINE_TYPES - actual
        assert not missing, f"RetrieverEngineType missing values: {missing}"

    def test_engine_modules_defined(self) -> None:
        """Every concrete engine in the plan must have a retrieval module."""
        # Sanity import guard — the modules are the wires that env_registry
        # and factory.py dispatch on; if any is missing the import fails.
        import importlib

        engine_modules = (
            "src.ai.retrieval.milvus",
            "src.ai.retrieval.elasticsearch_v7",
            "src.ai.retrieval.elasticsearch_v8",
            "src.ai.retrieval.opensearch",
            "src.ai.retrieval.pgvector",
            "src.ai.retrieval.qdrant",
            "src.ai.retrieval.weaviate",
            "src.ai.retrieval.doris",
            "src.ai.retrieval.tencent_vectordb",
            "src.ai.retrieval.sqlite_vec",
        )
        for mod in engine_modules:
            importlib.import_module(mod)