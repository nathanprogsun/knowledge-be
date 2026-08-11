"""ARQ worker task: ``image_multimodal``.

Maps the upstream image-multimodal task: receives the serialized
``ImageMultimodalPayload`` over ARQ, validates it, and dispatches it to
the core vision pipeline
:class:`src.core.knowledge.documents.image_multimodal.ImageMultimodalService`.

The handler stays thin: payload parsing, logging, and result shaping
live here; the OCR / caption inference, child-chunk creation, and
indexing orchestration live in the core layer. The worker wiring layer
is responsible for constructing a composed :class:`ImageMultimodalService`
(chunk service, KB service, VLM / file / index resolvers, finalizer) and
injecting it into the handler. Until then the core seam raises
``NotImplementedError`` and the task short-circuits before any external
work.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from src.ai.embedding import TaskContext
from src.app_logging import logger
from src.core.knowledge.documents.image_multimodal import (
    ImageMultimodalOutcome,
    ImageMultimodalService,
)
from src.core.knowledge.documents.image_multimodal import (
    ImageMultimodalPayload as CoreImageMultimodalPayload,
)
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class ImageMultimodalTaskPayload(BaseModel):
    """ARQ-side payload for the ``image_multimodal`` task.

    Mirrors the upstream wire contract: tenant / knowledge / KB
    identifiers, the parent text chunk and the image reference, the
    OCR / caption toggles, and the locale / source-type hints.
    ``attempt`` and ``image_index`` locate this image inside the parent
    run for tracing; they are carried for compatibility but not yet
    consumed by the core seam.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    chunk_id: str
    image_url: str
    image_local_path: str = ""
    enable_ocr: bool = False
    enable_caption: bool = False
    language: str = ""
    image_source_type: str = ""
    attempt: int = 0
    image_index: int = 0


async def process_image_multimodal(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    chunk_id: str,
    image_url: str,
    image_local_path: str = "",
    enable_ocr: bool = False,
    enable_caption: bool = False,
    language: str = "",
    image_source_type: str = "",
    service: ImageMultimodalService | None = None,
) -> ImageMultimodalOutcome:
    """Run one image through the core multimodal pipeline.

    A fully composed ``service`` may be injected by the worker wiring
    layer; when omitted the seam short-circuits with
    ``NotImplementedError`` so an unwired task fails loudly instead of
    half-processing an image. The seam runs in the background-task
    context so the provider governor throttles model calls.
    """
    if service is None:
        raise NotImplementedError(
            "Image multimodal processing wiring lands in a later wave.",
        )
    payload = CoreImageMultimodalPayload(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        image_url=image_url,
        image_local_path=image_local_path,
        enable_ocr=enable_ocr,
        enable_caption=enable_caption,
        language=language,
        image_source_type=image_source_type,
    )
    return await service.process_image(
        ctx=TaskContext(is_background_task=True),
        payload=payload,
    )


@register_task("image_multimodal")
async def task_image_multimodal(
    ctx: WorkerContext,
    *,
    service: ImageMultimodalService | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``image_multimodal`` task.

    Parses the ARQ payload into :class:`ImageMultimodalTaskPayload`,
    delegates to :func:`process_image_multimodal`, and returns a
    JSON-serialisable summary of the outcome. A fully composed
    ``service`` may be injected by the worker wiring layer; when omitted
    the core seam raises ``NotImplementedError`` before any external
    work. The ``ctx`` argument is currently unused — the worker context
    carries the ARQ-Redis pool, but the multimodal seam is invoked
    without database access at this stage.
    """
    parsed = ImageMultimodalTaskPayload.model_validate(payload)
    logger.info(
        "image_multimodal: tenant={} knowledge={} kb={} chunk={} image={!r} ocr={} caption={}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.knowledge_base_id,
        parsed.chunk_id,
        parsed.image_url,
        parsed.enable_ocr,
        parsed.enable_caption,
    )

    outcome = await process_image_multimodal(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        chunk_id=parsed.chunk_id,
        image_url=parsed.image_url,
        image_local_path=parsed.image_local_path,
        enable_ocr=parsed.enable_ocr,
        enable_caption=parsed.enable_caption,
        language=parsed.language,
        image_source_type=parsed.image_source_type,
        service=service,
    )
    return _serialise_outcome(outcome)


def _serialise_outcome(outcome: ImageMultimodalOutcome) -> dict[str, JsonValue]:
    """Project an :class:`ImageMultimodalOutcome` onto a JSON dict."""
    return {
        "ocr_text": outcome.ocr_text,
        "caption": outcome.caption,
        "image_bytes": outcome.image_bytes,
        "chunks_created": outcome.chunks_created,
        "indexed": outcome.indexed,
        "skipped": outcome.skipped,
        "read_error": outcome.read_error,
        "ocr_error": outcome.ocr_error,
        "caption_error": outcome.caption_error,
        "vlm_model_id": outcome.vlm_model_id,
        "ocr_chars": outcome.ocr_chars,
        "caption_chars": outcome.caption_chars,
        "ocr_skipped": outcome.ocr_skipped,
    }


def parse_payload(payload: Mapping[str, JsonValue]) -> ImageMultimodalTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return ImageMultimodalTaskPayload.model_validate(dict(payload))


__all__ = [
    "ImageMultimodalTaskPayload",
    "parse_payload",
    "process_image_multimodal",
    "task_image_multimodal",
]
