# Agent Note: Session attachment HTTP for paperclip upload

Status: implemented
Date: 2026-09-05
Scope: persist and expose session attachments on `/sessions/{id}/attachments`
Related files: src/web/api/chat/sessions/router.py, src/web/api/chat/sessions/attachments.py, src/core/knowledge/documents/temporary_document.py, src/core/knowledge/documents/factory.py, src/web/deps/knowledge_documents.py, src/core/contracts/sessions.py, tests/web/test_session_attachments.py

## Context

The paperclip already posts multipart files to `/api/v1/sessions/{id}/attachments` and polls GET until `ready` or `failed`. The sessions router did not declare those routes. Uploads 404ed. Leaving a row in `uploaded` would keep the paperclip polling forever.

## Decision

`TemporaryDocumentService` remains the lifecycle owner. HTTP checks session ownership with `SessionService.get`, writes bytes through the tenant default `FileService.save_bytes(..., temp=True)`, records metadata, then `mark_ready` with empty content. Preview streams `get_file(resource_ref)`. Upload and delete require Contributor. Get, list, and preview stay Viewer. Attachment ids are not bound into the QA runner.

## Alternatives considered

- **Leave status `uploaded` until a parse worker finishes** — rejected: the worker only returns `dispatched` and the paperclip would poll forever.
- **Reuse `BackendFileServiceResolver`** — rejected: that resolver needs a knowledge base id. Session attachments are not KB-scoped.
- **Enqueue parse from the upload handler** — rejected: workers must not import `web` or `db`, and empty ready content is enough for this surface.
- **Bind `attachment_ids` into `KnowledgeQARunner`** — rejected: the send body already accepts the field. Prompt bind is a later change.

## Consequences

The paperclip can upload, poll to `ready`, preview bytes, and delete. Refresh has no frontend list client yet. Chat send still ignores `attachment_ids`. Missing storage is `temporary_document.storage_unavailable`, not a 500.

## Required verification

- `uv run pytest tests/web/test_session_attachments.py tests/web/test_session_message_views.py tests/integration/web/test_routers.py tests/core/knowledge/documents/test_temporary_document_service.py`
- `make openapi` and `make check-endpoint`
- `uv run python scripts/check_layer_violation.py --domains auth,tenant,infra,knowledge,chat`
- `python scripts/verify_agent_notes.py --repo-root .`
