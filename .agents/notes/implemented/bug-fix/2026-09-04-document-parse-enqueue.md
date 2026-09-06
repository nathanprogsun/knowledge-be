# Agent Note: Wire document parse enqueue

Status: implemented
Date: 2026-09-04
Scope: enqueue document_process on URL/file create and compose the worker pipeline
Related files: src/core/knowledge/documents/arq_enqueue.py, src/core/knowledge/documents/create_url.py, src/core/knowledge/documents/process_runtime.py, src/web/deps/knowledge_documents.py, src/workers/main.py

## Context

URL and file create persisted `parse_status=pending` and stopped. The
production orchestrator had no dispatcher, there was no ARQ producer,
and the worker called an uncomposed `DocumentProcessPipeline()`. A
missing docreader container would have marked the row `failed` only
after a job actually ran. Silent pending was the missing enqueue.

## Decision

Open an APP-scope ARQ pool in the FastAPI lifespan and inject
`ArqDocumentEnqueuer` as both the file dispatcher and the reparse
enqueuer. URL create now submits `document_process` with `url=` after
insert. The worker startup builds a core `DocumentProcessRuntime`
(session factory + `DocReaderAdapter` over `DocReaderClient`) so a
consumed job can parse. Worker Redis falls back to `REDIS_URL` when
`WORKER_REDIS_URL` is unset. Password-only Redis URLs normalize an
empty username and unquote the password before AUTH.
`GET /system/parser-engines` probes `DOCREADER_ADDR` instead of
hard-coding disconnected.

## Alternatives considered

- **Parse inline on the API request** — rejected: the pipeline is
  already a worker task with its own session and timeout; blocking
  create on gRPC would hide enqueue failures behind request timeouts.
- **Leave URL create pending until a later worker wave** — rejected:
  the SPA polls pending forever, so the operator loop cannot finish.
- **Put DatabaseEngine construction in `workers/`** — rejected: the
  layer gate forbids workers importing `db`; composition stays in core.

## Consequences

Create and reparse enqueue a real job when Redis is up. If Redis is
down at API startup the pool stays unset and rows still stay pending.
A running worker with no docreader now flips the row to `failed`
instead of leaving it pending. Manual Markdown enqueue still hits an
unimplemented worker seam.

## Required verification

- `uv run pytest tests/core/knowledge/test_arq_enqueue.py tests/core/knowledge/test_docreader_adapter.py tests/core/knowledge/test_create_variants.py tests/workers/test_document_process.py tests/web/test_favorites_views.py -q`
- Live: create or reparse a URL document and confirm `parse_status`
  leaves `pending`.
