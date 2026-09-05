# Agent Note: Document detail routes the SPA already calls

Status: implemented
Date: 2026-09-04
Scope: register batch / spans / regenerate-summary and raise the tag list page-size cap
Related files: src/web/api/knowledge/documents/document_reads.py, src/core/knowledge/documents/span_tree.py, src/web/api/knowledge/tags/router.py

## Context

The knowledge-base list and document drawer already poll
`GET /knowledge/batch`, open `GET /knowledge/{id}/spans`, refresh
summaries through `POST /knowledge/{id}/regenerate-summary`, and load
every tag with `page_size=1000`. Those document routes were absent
(404) and the tag list rejected 1000 (422), so a document that exists
still looked broken in the UI.

## Decision

Register the three document routes on the documents router. Batch read
reuses `KnowledgeService.get_documents` and optionally filters by
`kb_id`. Spans load the document first (cross-workspace stays 404) and
return an empty root tree when the tracker has never recorded an
attempt, so the timeline can render instead of erroring. Regenerate
summary only marks `summary_status=pending`; the worker is still
unwired. Tag list `page_size` accepts up to 1000 so the upload dialog
can fetch the full set in one page.

## Alternatives considered

- **Treat the 404s as out of scope for the config-save fix** — rejected:
  live verification already hit these URLs on the same knowledge-base
  page; leaving them broken keeps the document card polling and Trace
  menu failing.
- **Return 404 from spans when no attempt exists** — rejected: the SPA
  probes Trace availability with this URL and treats a missing tree as
  a hard error.
- **Clamp tag `page_size` to 100** — rejected: the client asks for 1000
  and a silent clamp would drop tags on larger knowledge bases.

## Consequences

Document cards can poll parse/summary status. The Trace drawer can
open on a document that has never been tracked. Regenerating a summary
queues a status the UI already understands; the worker still has to
land later. Tag pickers can load a full page of 1000.

## Required verification

- `uv run pytest tests/core/knowledge/test_span_tree.py tests/core/knowledge/test_doc_service_crud.py tests/integration/web/api/knowledge/documents/test_controller.py tests/integration/web/api/knowledge/tags/test_controller.py`
- `make openapi` and `make check-endpoint`
- Live `GET /knowledge/batch`, `GET /knowledge/{id}/spans`, and tags
  `page_size=1000` are no longer 404 / 422
