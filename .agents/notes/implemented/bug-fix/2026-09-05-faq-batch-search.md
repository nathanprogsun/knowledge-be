# Agent Note: FAQ search, batch upsert, and last-result display

Status: implemented
Date: 2026-09-05
Scope: wire FAQ search, JSON upsert, batch tag/field updates, and last-result display on existing contracts
Related files: src/web/api/knowledge/faq/router.py, src/web/api/knowledge/faq/views.py, src/core/knowledge/faq/service/faq_service.py, src/core/knowledge/faq/import_runner.py, src/core/contracts/knowledge.py, tests/integration/web/api/knowledge/faq/test_controller.py

## Context

The FAQ manager already called search, batch tag, batch fields, JSON `{entries, mode}` upsert, and last-result display. The FAQ router did not declare those routes. `POST /entries` bound a required multipart file, so the SPA JSON POST 422ed.

## Decision

Keep one `POST /entries` and branch on `Content-Type`. `application/json` validates `FAQBatchUpsertPayload` and loops `FAQService.create_entry` (replace clears first). Multipart still runs `FAQImportRunner` on CSV / Excel bytes. JSON upsert records a completed in-memory progress object so the existing poller works. Search is keyword overlap over the knowledge base's entries, with `match_type=keywords` and a numeric `score`; `data` is a list. `FAQEntryTagsBatchUpdate.updates` accepts `int | None` so a null clears a tag. Last-result close updates `display_status` on the newest in-memory task for that knowledge base.

## Alternatives considered

- **Second path that parses JSON as CSV** — rejected: the file runner owns spreadsheet layout. Treating the SPA body as a fake file would invent a second import pipeline.
- **Enqueue a worker for JSON upsert** — rejected: file import is already synchronous and the SPA only needs a completed `task_id` to poll.
- **Vector / embedding FAQ search** — rejected: the manager can render keyword hits. Embedding belongs with retrieval, not this HTTP hole.
- **Database table for display_status** — rejected: progress already lives in `FAQImportTaskStore`. A table would outlive the process-local task the card polls.

## Consequences

The manager's search, JSON import, batch tag/field edits, and close-card action reach live HTTP. CSV / Excel upload on `POST /entries` is unchanged. Similar-question append is still unwired. Search ignores `vector_threshold` and does not call an embedder. Display state is lost when the process restarts.

## Required verification

- `uv run pytest tests/integration/web/api/knowledge/faq/test_controller.py tests/core/knowledge/faq/test_faq_ops.py tests/integration/web/test_routers.py`
- `make openapi` and `make check-endpoint`
- `uv run python scripts/check_layer_violation.py --domains auth,tenant,infra,knowledge,chat`
- `python scripts/verify_agent_notes.py --repo-root .`
