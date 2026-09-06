# Agent Note: Persist manual tags without a fake parse

Status: implemented
Date: 2026-09-05
Scope: accept tag_ids and process_config on manual create and PUT; refuse publish until a worker can finish
Related files: src/core/contracts/knowledge.py, src/core/knowledge/documents/create_manual.py, src/core/knowledge/documents/service/knowledge_service.py, src/web/api/knowledge/documents/router.py, src/web/api/knowledge/documents/document_reads.py, frontend/src/api/knowledge-base/index.ts

## Context

The SPA already posts `tag_ids` and `process_config` on manual create
and wants those tags to persist on save. The create contract only
declared `tag_id`, so Pydantic extra=ignore dropped the list. PUT
accepted `process_config` but never forwarded tags. Publish stamped
`parse_status=pending` even though the manual worker cannot finish a
parse, so the editor spinner never resolved.

## Decision

`CreateManualKnowledgeRequest` and `UpdateKnowledgeRequest` accept
`tag_ids` and `process_config`. The create router forwards both. PUT
resolves `tag_ids` (list wins, including empty) over a single `tag_id`
and binds them through `KnowledgeService` plus an optional
`TagRepository`. Empty list clears bindings. File rows may receive a
tags-only write; content, status, and process_config on a non-manual
row still raise `knowledge.manual_fields_unsupported`. Draft creates
store `process_overrides` when the client sent `process_config`.
Create and PUT raise `knowledge.manual_publish_unavailable` when
status normalizes to publish. They do not enqueue `manual_process` and
do not stamp pending.

## Alternatives considered

- **Stamp pending and enqueue `manual_process`** — rejected: the worker
  still raises, so a pending row would spin the editor forever.
- **Keep publish as a draft write with a warning field** — rejected: a
  typed `ValidationError` fails closed and keeps the row out of a
  parse the worker cannot finish.
- **Drop `tag_id` in the same change** — rejected: older clients still
  send the singular field; list wins when both appear.

## Consequences

Draft saves persist tags and process overrides. Publish is an explicit
error until the manual worker lands. File bytes stay untouched on a
tags-only PUT. Generated OpenAPI and the frontend client now name the
same list field the editor already posts.

## Required verification

- `uv run pytest tests/core/knowledge/test_create_variants.py tests/core/knowledge/test_doc_service_crud.py tests/core/knowledge/test_arq_enqueue.py`
- `make openapi` and `make check-endpoint`
- `python scripts/verify_agent_notes.py --repo-root .`
