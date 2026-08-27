"""Unit tests for the ARQ ``chunk:extract`` worker task.

The handler is a thin shim over
:class:`src.core.knowledge.documents.chunk_extract.ChunkExtractor`; the
tests inject stub seams (a chat resolver and an extractor) and, where
the delegation contract is the target, patch the module-level
``_run_extraction`` helper so they run without a database / AI provider.
They cover:

- payload validation (required ids, optional fields, defaults),
- registry wiring (the handler registers under ``"chunk:extract"``),
- delegation: the parsed fields reach the extraction helper with the
  right names and types,
- model resolution: an unresolvable model short-circuits with a skip,
- result serialisation: the returned dict matches
  :class:`ExtractionOutcome` semantics.

No real ARQ broker, no real DB, no real provider calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.knowledge.documents.chunk_extract import ChunkExtractor, ExtractionOutcome
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, get_task
from src.workers.tasks import chunk_extract as chunk_extract_module
from src.workers.tasks.chunk_extract import (
    TASK_NAME,
    ChatResolver,
    parse_payload,
    task_chunk_extract,
)


def _make_ctx() -> WorkerContext:
    """Build the minimal ARQ context the handler receives."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


def _base_payload() -> dict[str, JsonValue]:
    """Required-field payload shared across delegation tests."""
    return {
        "tenant_id": 1,
        "chunk_id": "chunk-1",
        "model_id": "model-1",
    }


class _StubChat:
    """Object standing in for a resolved chat client.

    The extraction stub only forwards the value; it never calls into the
    client, so an empty object is sufficient.
    """


class _StubResolver:
    """Fake :class:`ChatResolver` returning a fixed chat (or ``None``)."""

    def __init__(self, chat: _StubChat | None) -> None:
        self._chat = chat
        self.called_model_id: str | None = None

    async def resolve_chat(self, *, model_id: str) -> _StubChat | None:
        self.called_model_id = model_id
        return self._chat


class _StubExtractor:
    """Fake core extractor recording every extraction call."""

    def __init__(self, outcome: ExtractionOutcome) -> None:
        self._outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def extract_chunk(self, **kwargs: Any) -> ExtractionOutcome:
        self.calls.append(kwargs)
        return self._outcome


# ── Registry ─────────────────────────────────────────────────────────


def test_handler_registered_under_task_name() -> None:
    """The decorator registers the handler at import time."""
    assert get_task(TASK_NAME) is task_chunk_extract


# ── Payload model ────────────────────────────────────────────────────


def test_parse_payload_accepts_minimum_required_fields() -> None:
    parsed = parse_payload(_base_payload())
    assert parsed.tenant_id == 1
    assert parsed.chunk_id == "chunk-1"
    assert parsed.model_id == "model-1"
    assert parsed.knowledge_id == ""
    assert parsed.attempt == 0
    assert parsed.chunk_index == 0


def test_parse_payload_accepts_all_optional_fields() -> None:
    parsed = parse_payload(
        {
            **_base_payload(),
            "knowledge_id": "k-1",
            "attempt": 3,
            "chunk_index": 7,
        }
    )
    assert parsed.knowledge_id == "k-1"
    assert parsed.attempt == 3
    assert parsed.chunk_index == 7


def test_parse_payload_rejects_missing_tenant_id() -> None:
    payload = _base_payload()
    payload.pop("tenant_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_chunk_id() -> None:
    payload = _base_payload()
    payload.pop("chunk_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_model_id() -> None:
    payload = _base_payload()
    payload.pop("model_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_ignores_unknown_and_tracing_fields() -> None:
    parsed = parse_payload(
        {
            **_base_payload(),
            "lf_traceparent": "00-abc-def-01",
            "extra": "ignored",
        }
    )
    assert parsed.tenant_id == 1


def test_payload_is_frozen() -> None:
    parsed = parse_payload(_base_payload())
    with pytest.raises(ValidationError):
        parsed.chunk_id = "tampered"


# ── Delegation ───────────────────────────────────────────────────────


async def test_task_delegates_to_core_extraction() -> None:
    captured: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> ExtractionOutcome:
        captured.update(kwargs)
        return ExtractionOutcome(node_count=2, relation_count=1)

    with patch.object(
        chunk_extract_module,
        "_run_extraction",
        side_effect=_fake_run,
    ):
        result = await task_chunk_extract(
            _make_ctx(),
            extractor=cast(ChunkExtractor, _StubExtractor(ExtractionOutcome())),
            chat_resolver=cast(ChatResolver, _StubResolver(None)),
            **_base_payload(),
        )

    assert captured["tenant_id"] == 1
    assert captured["chunk_id"] == "chunk-1"
    assert captured["model_id"] == "model-1"
    assert captured["knowledge_id"] == ""
    assert captured["chunk_index"] == 0

    assert result == {
        "skipped": False,
        "reason": "",
        "node_count": 2,
        "relation_count": 1,
    }


async def test_task_forwards_injected_seams() -> None:
    extractor = _StubExtractor(ExtractionOutcome())
    resolver = _StubResolver(None)
    captured: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> ExtractionOutcome:
        captured.update(kwargs)
        return ExtractionOutcome()

    with patch.object(chunk_extract_module, "_run_extraction", side_effect=_fake_run):
        await task_chunk_extract(
            _make_ctx(),
            extractor=cast(ChunkExtractor, extractor),
            chat_resolver=cast(ChatResolver, resolver),
            **_base_payload(),
        )

    assert captured["extractor"] is extractor
    assert captured["chat_resolver"] is resolver


async def test_task_skips_when_seams_not_wired() -> None:
    result = await task_chunk_extract(_make_ctx(), **_base_payload())

    assert result == {
        "skipped": True,
        "reason": "not_wired",
        "node_count": 0,
        "relation_count": 0,
    }


async def test_task_skips_when_model_unresolved() -> None:
    extractor = _StubExtractor(ExtractionOutcome())
    resolver = _StubResolver(None)

    result = await task_chunk_extract(
        _make_ctx(),
        extractor=cast(ChunkExtractor, extractor),
        chat_resolver=cast(ChatResolver, resolver),
        **_base_payload(),
    )

    assert resolver.called_model_id == "model-1"
    assert extractor.calls == []
    assert result == {
        "skipped": True,
        "reason": "model_unavailable",
        "node_count": 0,
        "relation_count": 0,
    }


async def test_task_calls_core_with_resolved_chat() -> None:
    chat = _StubChat()
    extractor = _StubExtractor(ExtractionOutcome(node_count=3, relation_count=4))
    resolver = _StubResolver(chat)

    result = await task_chunk_extract(
        _make_ctx(),
        extractor=cast(ChunkExtractor, extractor),
        chat_resolver=cast(ChatResolver, resolver),
        **_base_payload(),
        knowledge_id="k-1",
        chunk_index=2,
    )

    assert resolver.called_model_id == "model-1"
    assert len(extractor.calls) == 1
    call = extractor.calls[0]
    assert call["tenant_id"] == 1
    assert call["chunk_id"] == "chunk-1"
    assert call["chat"] is chat
    assert call["knowledge_id"] == "k-1"
    assert call["chunk_index"] == 2
    assert call["ctx"].is_background_task is True

    assert result == {
        "skipped": False,
        "reason": "",
        "node_count": 3,
        "relation_count": 4,
    }


async def test_task_rejects_invalid_payload() -> None:
    """Invalid payloads surface as Pydantic validation errors."""
    with pytest.raises(ValidationError):
        await task_chunk_extract(_make_ctx(), tenant_id="not-an-int")


# ── Re-registration guard ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the module leave the registration in place, but a
    future test that re-registers under the same name would silently
    overwrite it. The fixture is defensive — a no-op today.
    """
    import src.workers.tasks  # noqa: F401 — pre-warm so the snapshot captures registered handlers.
    from src.workers import registry as registry_module

    snapshot = dict(registry_module.all_tasks())
    # Drop any test-only handler that was registered by ``@register_task``
    # decorators at import time of this module (e.g. ``test_base``'s
    # ``test_task``). The canonical handler set is what downstream
    # invariant tests assert against.
    baseline = {name: handler for name, handler in snapshot.items() if not name.startswith("test_")}
    yield
    # Restore the baseline after the test: drop anything newly added
    # (incl. handlers the test itself registered) and re-assert the
    # canonical handlers from the snapshot.
    current = registry_module.all_tasks()
    for name in list(current.keys()):
        if name not in baseline:
            current.pop(name, None)
    for name, handler in baseline.items():
        current[name] = handler
