"""ARQ worker task: ``faq:import``.

Maps the upstream FAQ-import task: receives the serialized import payload
over ARQ, validates it, and dispatches it to the core import runner
:class:`src.core.knowledge.faq.import_runner.FAQImportRunner`, which runs
the parse / validate / persist pipeline and records the completed
progress the progress endpoint polls by ``task_id``.

The handler stays thin — payload parsing, base64 decoding, logging, and
result shaping live here; the file parsing, per-entry validation,
duplicate guard, and persistence live in the core layer. The worker
payload carries the uploaded import file (base64-encoded bytes plus the
file name) rather than the upstream entry list, because this port's FAQ
import pipeline is file-driven: it parses the CSV / Excel template itself.

Wire field names mirror the upstream contract where the shapes overlap
(``tenant_id``, ``task_id``, ``kb_id``, ``knowledge_id``, ``mode``,
``dry_run``); ``extra="ignore"`` drops the upstream tracing and idempotency
fields that no consumer reads yet.

A session-bound :class:`FAQImportRunner` is injected by the worker wiring
layer (the worker context carries no database engine). Until that wiring
lands, an uninjected seam raises rather than silently dropping an import,
so a miswired worker fails loudly.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.knowledge.documents.faq_import import FAQ_BATCH_MODE_APPEND
from src.core.knowledge.faq.import_runner import FAQImportRunner
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

#: Upstream task name kept verbatim so queue consumers see the same
#: string they used to enqueue with.
TASK_NAME = "faq:import"


class FAQImportPayload(BaseModel):
    """ARQ-side payload for the ``faq:import`` task.

    Carries the scope of one FAQ import: the owning tenant, the target
    knowledge base and its FAQ container, the uploaded import file
    (``file_data`` is base64-encoded so the JSON payload stays
    text-safe), and the import mode / dry-run toggle. ``task_id`` is the
    caller-generated correlation id carried for logging; the core runner
    records the task progress under its own generated id.

    Field names use snake_case so ARQ's JSON deserializer maps
    transparently onto this model. The upstream tracing-context and
    idempotency fields are accepted via the JSON payload but not
    modelled here (no consumer reads them yet), so ``extra="ignore"``
    drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    kb_id: str
    knowledge_id: str = ""
    filename: str
    file_data: str
    mode: str = FAQ_BATCH_MODE_APPEND
    dry_run: bool = False
    task_id: str = ""


@register_task(TASK_NAME)
async def task_faq_import(
    ctx: WorkerContext,
    *,
    runner: FAQImportRunner | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``faq:import`` task.

    Parses the ARQ payload into :class:`FAQImportPayload`, decodes the
    base64 file bytes, and delegates to :func:`process_faq_import`. A
    session-bound ``runner`` may be injected by the worker wiring layer;
    when omitted, the seam raises so a miswired import fails loudly.
    ``ctx`` is currently unused — the worker context carries the ARQ-Redis
    pool, which the import seam does not need at this stage.
    """
    parsed = FAQImportPayload.model_validate(payload)
    logger.info(
        "faq_import: tenant={} kb={} knowledge={} file={} mode={} dry_run={} task={}",
        parsed.tenant_id,
        parsed.kb_id,
        parsed.knowledge_id,
        parsed.filename,
        parsed.mode,
        parsed.dry_run,
        parsed.task_id,
    )
    return await process_faq_import(
        file_data=base64.b64decode(parsed.file_data, validate=True),
        filename=parsed.filename,
        tenant_id=parsed.tenant_id,
        knowledge_base_id=parsed.kb_id,
        knowledge_id=parsed.knowledge_id,
        mode=parsed.mode,
        dry_run=parsed.dry_run,
        runner=runner,
    )


async def process_faq_import(
    *,
    file_data: bytes,
    filename: str,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    mode: str = FAQ_BATCH_MODE_APPEND,
    dry_run: bool = False,
    runner: FAQImportRunner | None = None,
) -> dict[str, JsonValue]:
    """Run one FAQ batch import to completion via the core import runner.

    Delegates to :meth:`FAQImportRunner.run` and returns the completed
    task progress as a JSON-serialisable dict. ``runner`` is injected by
    the worker wiring layer — an :class:`FAQImportRunner` bound to a
    per-job ``AsyncSession`` and the process-wide progress store. No
    runner can be constructed here (the worker context carries no
    database engine), so an uninjected call raises ``NotImplementedError``
    instead of silently skipping the import.
    """
    if runner is None:
        raise NotImplementedError(
            "FAQ import requires a session-bound FAQImportRunner "
            "injected by the worker wiring layer",
        )
    progress = await runner.run(
        file_data=file_data,
        filename=filename,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        mode=mode,
        dry_run=dry_run,
    )
    return progress.model_dump(mode="json")


__all__ = [
    "TASK_NAME",
    "FAQImportPayload",
    "process_faq_import",
    "task_faq_import",
]
