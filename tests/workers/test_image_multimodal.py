"""Unit tests for the ``image_multimodal`` worker task.

Covers the worker-side surface: the registered handler is the expected
function, the payload parses cleanly into the contract model, the
handler delegates to the core seam with the parsed arguments, and the
core seam dispatches to an injected :class:`ImageMultimodalService` (or
raises ``NotImplementedError`` until the wiring layer lands). The core
dispatch is exercised through a stubbed service so no real database or
AI provider is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.ai.embedding import Context
from src.core.knowledge.documents.image_multimodal import (
    ImageMultimodalOutcome,
)
from src.core.knowledge.documents.image_multimodal import (
    ImageMultimodalPayload as CoreImageMultimodalPayload,
)
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import image_multimodal as image_multimodal_module
from src.workers.tasks.image_multimodal import (
    ImageMultimodalTaskPayload,
    parse_payload,
    process_image_multimodal,
    task_image_multimodal,
)


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture
def ctx() -> WorkerContext:
    """Worker context for ad-hoc task invocations."""
    return make_ctx()


def _base_payload() -> dict[str, object]:
    """Required-field payload shared across delegation tests."""
    return {
        "tenant_id": 42,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
        "chunk_id": "chunk-7",
        "image_url": "local://tenants/42/images/1.png",
    }


@pytest.fixture
def valid_payload() -> dict[str, object]:
    """A representative JSON payload for the image-multimodal task."""
    return {
        **_base_payload(),
        "image_local_path": "/tmp/1.png",
        "enable_ocr": True,
        "enable_caption": True,
        "language": "en-US",
        "image_source_type": "scanned_pdf",
        "attempt": 3,
        "image_index": 2,
    }


class _StubService:
    """Object standing in for ``ImageMultimodalService`` in seam tests.

    The seam only awaits ``process_image`` and returns the outcome, so a
    stub recording the call is sufficient — the real service is exercised
    by the core-layer tests.
    """

    def __init__(self, outcome: ImageMultimodalOutcome) -> None:
        self._outcome = outcome
        self.calls: list[tuple[Context, CoreImageMultimodalPayload]] = []

    async def process_image(
        self, *, ctx: Context, payload: CoreImageMultimodalPayload
    ) -> ImageMultimodalOutcome:
        self.calls.append((ctx, payload))
        return self._outcome


# ── Registration ────────────────────────────────────────────────────


def test_image_multimodal_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("image_multimodal") is task_image_multimodal


def test_image_multimodal_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("image_multimodl") is None


# ── Payload contract ────────────────────────────────────────────────


def test_payload_parses_full(valid_payload: dict[str, object]) -> None:
    """A complete payload round-trips through the contract model."""
    payload = ImageMultimodalTaskPayload.model_validate(dict(valid_payload))
    assert payload.tenant_id == 42
    assert payload.knowledge_id == "doc-1"
    assert payload.knowledge_base_id == "kb-1"
    assert payload.chunk_id == "chunk-7"
    assert payload.image_url == "local://tenants/42/images/1.png"
    assert payload.image_local_path == "/tmp/1.png"
    assert payload.enable_ocr is True
    assert payload.enable_caption is True
    assert payload.language == "en-US"
    assert payload.image_source_type == "scanned_pdf"
    assert payload.attempt == 3
    assert payload.image_index == 2


def test_payload_defaults_optional_fields() -> None:
    """Optional fields default sensibly when omitted."""
    payload = parse_payload(_base_payload())
    assert payload.image_local_path == ""
    assert payload.enable_ocr is False
    assert payload.enable_caption is False
    assert payload.language == ""
    assert payload.image_source_type == ""
    assert payload.attempt == 0
    assert payload.image_index == 0


def test_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    payload = _base_payload()
    payload.pop("tenant_id")
    with pytest.raises(ValidationError):
        ImageMultimodalTaskPayload.model_validate(payload)


def test_payload_rejects_missing_knowledge_id() -> None:
    """The knowledge id is mandatory."""
    payload = _base_payload()
    payload.pop("knowledge_id")
    with pytest.raises(ValidationError):
        ImageMultimodalTaskPayload.model_validate(payload)


def test_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge base id is mandatory."""
    payload = _base_payload()
    payload.pop("knowledge_base_id")
    with pytest.raises(ValidationError):
        ImageMultimodalTaskPayload.model_validate(payload)


def test_payload_rejects_missing_chunk_id() -> None:
    """The parent text chunk id is mandatory."""
    payload = _base_payload()
    payload.pop("chunk_id")
    with pytest.raises(ValidationError):
        ImageMultimodalTaskPayload.model_validate(payload)


def test_payload_rejects_missing_image_url() -> None:
    """The image reference is mandatory."""
    payload = _base_payload()
    payload.pop("image_url")
    with pytest.raises(ValidationError):
        ImageMultimodalTaskPayload.model_validate(payload)


def test_payload_ignores_unknown_fields() -> None:
    """Tracing fields not modelled here must be tolerated."""
    payload = parse_payload({**_base_payload(), "lf_trace_id": "trace-1"})
    assert payload.tenant_id == 42


def test_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = parse_payload(_base_payload())
    with pytest.raises(ValidationError):
        payload.image_url = "tampered"


# ── Worker dispatch ─────────────────────────────────────────────────


async def test_task_delegates_to_core_seam(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        image_multimodal_module,
        "process_image_multimodal",
        new_callable=AsyncMock,
        return_value=ImageMultimodalOutcome(chunks_created=2),
    ) as mock:
        result = await task_image_multimodal(ctx, **valid_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        chunk_id="chunk-7",
        image_url="local://tenants/42/images/1.png",
        image_local_path="/tmp/1.png",
        enable_ocr=True,
        enable_caption=True,
        language="en-US",
        image_source_type="scanned_pdf",
        service=None,
    )
    assert result == {
        "ocr_text": "",
        "caption": "",
        "image_bytes": 0,
        "chunks_created": 2,
        "indexed": False,
        "skipped": "",
        "read_error": "",
        "ocr_error": "",
        "caption_error": "",
        "vlm_model_id": "",
        "ocr_chars": 0,
        "caption_chars": 0,
        "ocr_skipped": "",
    }


async def test_task_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted optional fields fall back to defaults before dispatch."""
    with patch.object(
        image_multimodal_module,
        "process_image_multimodal",
        new_callable=AsyncMock,
        return_value=ImageMultimodalOutcome(),
    ) as mock:
        await task_image_multimodal(ctx, **_base_payload())  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        chunk_id="chunk-7",
        image_url="local://tenants/42/images/1.png",
        image_local_path="",
        enable_ocr=False,
        enable_caption=False,
        language="",
        image_source_type="",
        service=None,
    )


async def test_task_forwards_injected_service(
    ctx: WorkerContext,
) -> None:
    """A composed service injected by the wiring layer reaches the seam."""
    stub = _StubService(ImageMultimodalOutcome())
    with patch.object(
        image_multimodal_module,
        "process_image_multimodal",
        new_callable=AsyncMock,
        return_value=ImageMultimodalOutcome(),
    ) as mock:
        await task_image_multimodal(
            ctx,
            service=stub,  # type: ignore[arg-type]
            **_base_payload(),
        )

    mock.assert_awaited_once()
    assert mock.await_args.kwargs["service"] is stub


async def test_task_rejects_invalid_payload(
    ctx: WorkerContext,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_image_multimodal(ctx, tenant_id=1)


async def test_task_propagates_core_errors(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            image_multimodal_module,
            "process_image_multimodal",
            new_callable=AsyncMock,
            side_effect=RuntimeError("pipeline exploded"),
        ),
        pytest.raises(RuntimeError, match="pipeline exploded"),
    ):
        await task_image_multimodal(ctx, **valid_payload)  # type: ignore[arg-type]


# ── Core seam ───────────────────────────────────────────────────────


async def test_process_image_multimodal_dispatches_to_service() -> None:
    """The seam forwards the parsed fields onto ``service.process_image``."""
    outcome = ImageMultimodalOutcome(
        ocr_text="extracted text",
        caption="a concise caption",
        chunks_created=2,
    )
    stub = _StubService(outcome)
    result = await process_image_multimodal(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        chunk_id="chunk-7",
        image_url="local://tenants/42/images/1.png",
        image_local_path="/tmp/1.png",
        enable_ocr=True,
        enable_caption=True,
        language="en-US",
        image_source_type="scanned_pdf",
        service=stub,  # type: ignore[arg-type]
    )

    assert result is outcome
    assert len(stub.calls) == 1
    ctx, payload = stub.calls[0]
    assert ctx.is_background_task is True
    assert payload.tenant_id == 42
    assert payload.knowledge_id == "doc-1"
    assert payload.knowledge_base_id == "kb-1"
    assert payload.chunk_id == "chunk-7"
    assert payload.image_url == "local://tenants/42/images/1.png"
    assert payload.image_local_path == "/tmp/1.png"
    assert payload.enable_ocr is True
    assert payload.enable_caption is True
    assert payload.language == "en-US"
    assert payload.image_source_type == "scanned_pdf"


async def test_process_image_multimodal_runs_in_background_context() -> None:
    """The seam marks the run as a background task for the governor."""
    stub = _StubService(ImageMultimodalOutcome())
    await process_image_multimodal(
        tenant_id=1,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        chunk_id="chunk-7",
        image_url="https://example.com/1.png",
        service=stub,  # type: ignore[arg-type]
    )
    assert stub.calls[0][0].is_background_task is True


async def test_process_image_multimodal_raises_not_implemented_without_service() -> None:
    """The seam short-circuits until the wiring layer provides a service."""
    with pytest.raises(NotImplementedError):
        await process_image_multimodal(
            tenant_id=1,
            knowledge_id="doc-1",
            knowledge_base_id="kb-1",
            chunk_id="chunk-7",
            image_url="local://tenants/1/images/1.png",
        )


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
