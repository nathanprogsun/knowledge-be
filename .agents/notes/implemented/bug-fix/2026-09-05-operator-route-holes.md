# Agent Note: Operator route holes

Status: implemented
Date: 2026-09-05
Scope: chat stop, KB pin, invitation inbox, manual document edit, document download
Related files: src/web/api/chat/sessions/router.py, src/web/api/me/router.py, src/web/api/knowledge_bases/router.py, src/web/api/knowledge/documents/document_reads.py, src/core/knowledge/documents/service/knowledge_service.py, frontend/src/api/knowledge-base/index.ts

## Context

The SPA already called four live surfaces that had no backend route. Stop toasted
failure after every composer abort. The KB list pin button 404ed. The
invitations dialog opened from a working badge and then failed. Manual document
edit posted `/knowledge/manual/{id}` while only `PUT /knowledge/{id}` existed.

## Decision

Add `POST /sessions/{id}/stop` that checks session ownership and calls
`StopStreamService` without a message reader. The composer already aborted the
SSE client, so an empty in-process cancel flag is enough to clear the toast.
Store pins in `user_kb_pins` and hydrate `is_pinned` on the KB list. The pin
read runs before count enrichment so a swallowed count error cannot abort
the list transaction. Wire the existing invitation service to
`/me/invitations` list, accept, and decline.
Extend `PUT /knowledge/{id}` with optional manual `content` / `status` and
repoint the SPA there. `GET /knowledge/{id}/download` and `/preview` stream
stored objects, or the manual markdown body. Download is Contributor+;
preview stays Viewer+. Download answers as an attachment octet-stream;
preview keeps the stored type and both send `nosniff`. URL rows without a
stored object answer `knowledge.file_unavailable`.

## Alternatives considered

- **APP-scope StreamManager plus ChatService cancel** — rejected for this
  slice: it stops the server token loop, but the operator-visible bug was the
  404 toast. Shared cancel lands with continue-stream.
- **New `PUT /knowledge/manual/{id}` alias** — rejected: the documents router
  is already near the 800-line cap, and `PUT /knowledge/{id}` already updates
  the row.
- **Tenant-scoped members twins for the workspace page** — rejected: that
  surface needs org-to-tenant mapping, not a thin alias.

## Consequences

Stop, pin, the invitations inbox, and manual save stop 404ing. Composer attachments and workspace members stay later. Stop does not yet
cancel the in-flight model call. URL documents without stored bytes still
cannot preview.

## Required verification

- `uv run pytest tests/web/test_session_message_views.py tests/web/test_me_invitations.py tests/web/test_document_reads.py tests/core/knowledge/test_kb_service_crud.py tests/core/knowledge/test_doc_service_crud.py tests/integration/web/test_routers.py`
- Live: stop during a stream shows success, KB pin toggles, invitations dialog
  lists, manual editor save returns 200, file download/preview streams bytes
