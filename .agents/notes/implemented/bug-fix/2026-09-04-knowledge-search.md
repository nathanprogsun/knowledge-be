# Agent Note: Chat file picker search route

Status: implemented
Date: 2026-09-04
Scope: register GET /knowledge/search for the chat @ file picker
Related files: src/web/api/knowledge/documents/document_reads.py, src/core/knowledge/documents/service/knowledge_service.py, src/db/dao/knowledge_repository.py

## Context

The chat composer @ picker calls `GET /knowledge/search` with `keyword` or
`recent=true`. That path was unregistered, so mentioning a file 404'd and
the operator loop could not attach documents in a new chat.

## Decision

Register `GET /knowledge/search` on the documents router before `/{id}`.
The handler searches live documents across document-type knowledge bases
in the caller's workspace, matches `file_name` / `title`, optionally
filters `file_types` (url/html aliases included), and answers with
`{success, data, has_more, total}`. Empty keyword is valid only with
`recent=true`. Agent query params are accepted unused. The joined
knowledge-base name is filled on the wire so the picker can label the
file.

## Alternatives considered

- **Leave the 404 until chat retrieval lands** — rejected: the picker
  is a read of existing documents, not the retrieval pipeline, and the
  live composer already calls this URL.
- **Reuse `GET /knowledge-search` (semantic)** — rejected: the picker
  wants a filename/title page, not an embedding query.

## Consequences

The @ picker can list recent or matching files. FAQ knowledge bases stay
out of the join. Shared-agent KB filtering still happens in the SPA.

## Required verification

- `uv run pytest tests/core/knowledge/test_doc_service_crud.py tests/integration/web/api/knowledge/documents/test_controller.py tests/integration/web/test_routers.py`
- `make openapi` and `make check-endpoint`
- Live `GET /knowledge/search?recent=true` and `?keyword=` from the chat
  composer are no longer 404
