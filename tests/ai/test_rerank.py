"""Tests for the rerank service base, OpenAI-compatible backend, and transport.

All HTTP is faked through ``httpx.MockTransport`` — no network. The wire
parsing (``RankResult`` with string/object documents and the ``score``
fallback), the request shape (no ``truncate_prompt_tokens`` unless
configured, no empty ``additional_data``), the factory routing, the SSRF
gate, redirect re-validation, and the ``RerankService`` wrapper are
pinned here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.ai.rerank import (
    DocumentInfo,
    OpenAIReranker,
    RankResult,
    RerankerConfig,
    config_from_model,
    new_openai_reranker,
    new_reranker,
    validate_rerank_base_url,
)
from src.ai.rerank.aliyun import AliyunReranker
from src.ai.rerank.cloud import CloudReranker
from src.ai.rerank.jina import JinaReranker
from src.ai.rerank.lkeap import LKEAPReranker
from src.ai.rerank.nvidia import NvidiaReranker
from src.ai.rerank.remote_api import RerankRequest
from src.ai.rerank.volcengine import VolcengineReranker
from src.ai.rerank.zhipu import ZhipuReranker
from src.common.exception import ExternalServiceError, ValidationError
from src.core.contracts.infra import Model as WireModel
from src.core.contracts.infra import ModelParameters
from src.core.infra.models.rerank_service import RerankService
from src.db.dao.model_repository import ModelRepository
from src.db.models.infra.model import Model

_BASE_URL = "http://rerank.test/v1"

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow the fake upstream host so the SSRF gate does not DNS-resolve."""
    monkeypatch.setenv("SSRF_WHITELIST", "rerank.test")


def _json_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── Wire models ──────────────────────────────────────────────────────


def test_rank_result_parses_document_as_string() -> None:
    result = RankResult.model_validate(
        {"index": 0, "document": "This is a document", "relevance_score": 0.95}
    )
    assert result.index == 0
    assert result.document.text == "This is a document"
    assert result.relevance_score == 0.95


def test_rank_result_parses_document_as_object() -> None:
    result = RankResult.model_validate(
        {"index": 1, "document": {"text": "This is a document"}, "relevance_score": 0.87}
    )
    assert result.index == 1
    assert result.document.text == "This is a document"
    assert result.relevance_score == 0.87


def test_rank_result_score_field_fallback() -> None:
    result = RankResult.model_validate(
        {"index": 2, "document": "This is a document", "score": 0.92}
    )
    assert result.relevance_score == 0.92


def test_rank_result_relevance_score_wins_over_score() -> None:
    result = RankResult.model_validate(
        {"index": 4, "document": "This is a document", "relevance_score": 0.95, "score": 0.80}
    )
    assert result.relevance_score == 0.95


def test_rank_result_missing_score_defaults_zero() -> None:
    result = RankResult.model_validate({"index": 6, "document": "This is a document"})
    assert result.relevance_score == 0.0


def test_rank_result_round_trip() -> None:
    result = RankResult(
        index=1,
        document=DocumentInfo(text="Test document"),
        relevance_score=0.95,
    )
    parsed = RankResult.model_validate(result.model_dump(mode="json"))
    assert parsed.index == 1
    assert parsed.document.text == "Test document"
    assert parsed.relevance_score == 0.95


def test_document_info_serializes_to_text_object() -> None:
    dumped = RankResult(
        index=1,
        document=DocumentInfo(text="Test document"),
        relevance_score=0.95,
    ).model_dump(mode="json")
    assert dumped["document"] == {"text": "Test document"}


# ── RerankRequest shape ──────────────────────────────────────────────


def test_rerank_request_omits_empty_optionals() -> None:
    payload = RerankRequest(
        model="m",
        query="q",
        documents=["a"],
        truncate_prompt_tokens=None,
    ).model_dump(mode="json", exclude_none=True)
    assert payload == {"model": "m", "query": "q", "documents": ["a"]}


def test_rerank_request_keeps_configured_truncate() -> None:
    payload = RerankRequest(
        model="m",
        query="q",
        documents=["a"],
        truncate_prompt_tokens=511,
    ).model_dump(mode="json", exclude_none=True)
    assert payload["truncate_prompt_tokens"] == 511


# ── OpenAIReranker.rerank ────────────────────────────────────────────


async def test_rerank_sends_expected_request_and_returns_results() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers.get("Authorization")
        captured["content_type"] = request.headers.get("Content-Type")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "rerank-ok",
                "results": [
                    {"index": 2, "relevance_score": 0.998, "document": {"text": "C"}},
                    {"index": 0, "relevance_score": 0.51, "document": {"text": "A"}},
                    {"index": 1, "relevance_score": 0.006, "document": {"text": "B"}},
                ],
                "usage": {"total_tokens": 42},
            },
        )

    client = _json_client(handler)
    reranker = await new_openai_reranker(
        model_name="Qwen/Qwen3-VL-Reranker-8B",
        model_id="rr-1",
        api_key="sk-test",
        base_url=_BASE_URL,
        client=client,
    )
    results = await reranker.rerank("query", ["A", "B", "C"])

    assert captured["path"] == "/v1/rerank"
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["content_type"] == "application/json"
    body = captured["body"]
    assert body["model"] == "Qwen/Qwen3-VL-Reranker-8B"
    assert body["query"] == "query"
    assert body["documents"] == ["A", "B", "C"]
    assert "truncate_prompt_tokens" not in body
    assert "additional_data" not in body

    assert len(results) == 3
    assert results[0].index == 2
    assert results[0].relevance_score == 0.998
    assert results[0].document.text == "C"


async def test_rerank_omits_truncate_prompt_tokens_by_default() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.9, "document": "A"},
                    {"index": 1, "relevance_score": 0.1, "document": "B"},
                ]
            },
        )

    client = _json_client(handler)
    reranker = await new_openai_reranker(
        model_name="bge-reranker-base",
        model_id="rr-2",
        api_key="sk-test",
        base_url=_BASE_URL,
        client=client,
    )
    await reranker.rerank("query", ["A", "B", "C"])
    assert "truncate_prompt_tokens" not in captured["body"]


async def test_rerank_sends_truncate_prompt_tokens_when_configured() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": []})

    client = _json_client(handler)
    reranker = await new_openai_reranker(
        model_name="bge-reranker-base",
        model_id="rr-3",
        api_key="sk-test",
        base_url=_BASE_URL,
        extra_config={"truncate_prompt_tokens": "511"},
        client=client,
    )
    await reranker.rerank("query", ["A", "B", "C"])
    assert captured["body"]["truncate_prompt_tokens"] == 511


@pytest.mark.parametrize("raw", ["abc", "-1", "0"])
async def test_new_openai_reranker_rejects_invalid_truncate_prompt_tokens(
    raw: str,
) -> None:
    with pytest.raises(ValidationError, match="invalid truncate_prompt_tokens"):
        await new_openai_reranker(
            model_name="rerank-test",
            model_id="rr-4",
            api_key="sk-test",
            base_url=_BASE_URL,
            extra_config={"truncate_prompt_tokens": raw},
        )


async def test_rerank_raises_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _json_client(handler)
    reranker = await new_openai_reranker(
        model_name="m",
        model_id="rr-5",
        api_key="sk-test",
        base_url=_BASE_URL,
        client=client,
    )
    with pytest.raises(ExternalServiceError, match="Rerank API error: Http Status"):
        await reranker.rerank("query", ["A"])


# ── Factory routing ──────────────────────────────────────────────────


async def test_new_reranker_defaults_to_openai_for_openai_provider() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="bge-reranker-v2-m3",
            model_id="rr-6",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="openai",
        )
    )
    assert isinstance(reranker, OpenAIReranker)
    assert reranker.get_model_name() == "bge-reranker-v2-m3"
    assert reranker.get_model_id() == "rr-6"


async def test_new_reranker_detects_provider_from_base_url() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="m",
            model_id="rr-7",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="",
        )
    )
    assert isinstance(reranker, OpenAIReranker)


async def test_new_reranker_routes_aliyun() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="gte-rerank",
            model_id="rr-8",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="aliyun",
        )
    )
    assert isinstance(reranker, AliyunReranker)


async def test_new_reranker_routes_zhipu() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="rerank",
            model_id="rr-8",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="zhipu",
        )
    )
    assert isinstance(reranker, ZhipuReranker)


async def test_new_reranker_routes_jina() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="jina-reranker-v2",
            model_id="rr-8",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="jina",
        )
    )
    assert isinstance(reranker, JinaReranker)


async def test_new_reranker_routes_nvidia() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="nvidia-rerank",
            model_id="rr-8",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="nvidia",
        )
    )
    assert isinstance(reranker, NvidiaReranker)


async def test_new_reranker_routes_cloud() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="rerank",
            model_id="rr-8",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="cloud",
            app_id="app",
            app_secret="secret",
        )
    )
    assert isinstance(reranker, CloudReranker)


async def test_new_reranker_routes_lkeap() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="lke-reranker-base",
            model_id="rr-8",
            api_key="secret-id",
            base_url=_BASE_URL,
            provider="lkeap",
            app_secret="secret-key",
        )
    )
    assert isinstance(reranker, LKEAPReranker)


async def test_new_reranker_routes_volcengine() -> None:
    reranker = await new_reranker(
        RerankerConfig(
            model_name="doubao-seed-rerank",
            model_id="rr-8",
            api_key="ak",
            base_url=_BASE_URL,
            provider="volcengine",
            app_secret="sk",
        )
    )
    assert isinstance(reranker, VolcengineReranker)


async def test_new_reranker_applies_custom_headers() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["x_gateway"] = request.headers.get("X-Gateway")
        return httpx.Response(200, json={"results": []})

    client = _json_client(handler)
    reranker = await new_reranker(
        RerankerConfig(
            model_name="m",
            model_id="rr-9",
            api_key="sk-test",
            base_url=_BASE_URL,
            provider="openai",
            custom_headers={"X-Gateway": "g"},
        ),
        client=client,
    )
    await reranker.rerank("query", ["A"])
    assert captured["x_gateway"] == "g"


# ── config_from_model ────────────────────────────────────────────────


def test_config_from_model_maps_storage_row() -> None:
    model = Model(
        id="rr-1",
        tenant_id=1,
        name="bge-reranker-v2-m3",
        type="Rerank",
        source="remote",
        parameters={
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-xxx",
            "provider": "siliconflow",
            "extra_config": {"flag": "on"},
            "custom_headers": {"X-Gateway": "g"},
        },
        created_at=_NOW,
        updated_at=_NOW,
    )
    cfg = config_from_model(model, "app", "secret")
    assert cfg is not None
    assert cfg.model_id == "rr-1"
    assert cfg.model_name == "bge-reranker-v2-m3"
    assert cfg.source == "remote"
    assert cfg.api_key == "sk-xxx"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.provider == "siliconflow"
    assert cfg.extra_config == {"flag": "on"}
    assert cfg.custom_headers == {"X-Gateway": "g"}
    assert cfg.app_id == "app"
    assert cfg.app_secret == "secret"


def test_config_from_model_accepts_dumped_wire_parameters() -> None:
    wire = WireModel(
        id="rr-2",
        tenant_id=1,
        name="m",
        type="Rerank",
        source="remote",
        parameters=ModelParameters(base_url="https://api.example.com/v1", api_key="sk-xxx"),
        created_at=_NOW,
        updated_at=_NOW,
    )
    model = Model(
        id=wire.id,
        tenant_id=wire.tenant_id,
        name=wire.name,
        type=wire.type,
        source=wire.source,
        parameters=wire.parameters.model_dump(mode="json", exclude_none=True),
        created_at=_NOW,
        updated_at=_NOW,
    )
    cfg = config_from_model(model)
    assert cfg is not None
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.api_key == "sk-xxx"


def test_config_from_model_none_model_yields_none() -> None:
    assert config_from_model(None) is None


# ── SSRF gate + transport ────────────────────────────────────────────


async def test_validate_rerank_base_url_rejects_internal_url() -> None:
    with pytest.raises(ValidationError, match="base URL SSRF check failed"):
        await validate_rerank_base_url("http://169.254.169.254/latest/meta-data/")


async def test_new_reranker_rejects_internal_base_url() -> None:
    with pytest.raises(ValidationError, match="base URL SSRF check failed"):
        await new_reranker(
            RerankerConfig(
                model_name="m",
                model_id="rr-10",
                base_url="http://169.254.169.254/latest/meta-data/",
                provider="openai",
            )
        )


async def test_rerank_blocks_redirect_to_internal_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

    client = _json_client(handler)
    reranker = await new_openai_reranker(
        model_name="m",
        model_id="rr-11",
        api_key="sk-test",
        base_url=_BASE_URL,
        client=client,
    )
    with pytest.raises(ValidationError, match="base URL SSRF check failed"):
        await reranker.rerank("query", ["A"])


# ── RerankService ────────────────────────────────────────────────────


def _model_row() -> Model:
    return Model(
        id="rr-1",
        tenant_id=1,
        name="bge-reranker-v2-m3",
        type="Rerank",
        source="remote",
        parameters={
            "base_url": _BASE_URL,
            "api_key": "sk-xxx",
            "provider": "siliconflow",
            "app_id": "app",
            "app_secret": "secret",
        },
        created_at=_NOW,
        updated_at=_NOW,
    )


def _repo_with_row() -> AsyncMock:
    repo = AsyncMock(spec=ModelRepository)
    repo.find_by_tenant_and_id_or_fail = AsyncMock(return_value=_model_row())
    return repo


async def test_rerank_service_get_rerank_model() -> None:
    service = RerankService(models_repo=_repo_with_row())
    reranker = await service.get_rerank_model(tenant_id=1, model_id="rr-1")
    assert reranker.get_model_name() == "bge-reranker-v2-m3"
    assert reranker.get_model_id() == "rr-1"


async def test_rerank_service_get_rerank_model_rejects_bad_tenant() -> None:
    service = RerankService(models_repo=_repo_with_row())
    with pytest.raises(ValidationError, match="Tenant ID must be positive"):
        await service.get_rerank_model(tenant_id=0, model_id="rr-1")


async def test_rerank_service_rerank_drives_the_reranker() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9, "document": {"text": "B"}},
                    {"index": 0, "relevance_score": 0.1, "document": {"text": "A"}},
                ]
            },
        )

    client = _json_client(handler)
    service = RerankService(models_repo=_repo_with_row(), http_client=client)
    results = await service.rerank(
        tenant_id=1,
        model_id="rr-1",
        query="query",
        documents=["A", "B"],
    )
    assert [r.index for r in results] == [1, 0]
    assert captured["body"]["model"] == "bge-reranker-v2-m3"
