# Agent Note: Store file-url bytes before parse

Status: implemented
Date: 2026-09-05
Scope: persist downloadable file-url bytes onto the knowledge row so preview and download can stream them
Related files: src/core/knowledge/documents/file_url_store.py, src/core/knowledge/documents/backend_file_service.py, src/core/knowledge/documents/process_document.py, src/core/knowledge/documents/factory.py, src/core/knowledge/documents/process_runtime.py, src/web/deps/knowledge_documents.py, src/workers/tasks/document_process.py

## Context

Create already writes type `file_url` for a downloadable file URL and
type `url` for an ordinary web page. Preview and download stream
`file_path` through `document_reads.stream_document`. A `file_url` row
kept an empty path after the worker ran, so those surfaces returned
`knowledge.file_unavailable`. The worker parsed from the remote URL and
never called `FileService.save_bytes`. Ordinary web `url` rows must stay
without bytes. The upload resolver lived in `web/deps`, which the
worker cannot import.

## Decision

The process pipeline downloads a `file_url` row with an empty
`file_path`, calls `FileService.save_bytes`, and writes `file_path` on
the same job session before parse. Type `url` skips that path. HTML
`text/html` and `application/xhtml+xml` bodies are rejected unless the
declared file type is `html`, `htm`, or `mhtml`. Download reuses
`validate_ssrf_safe_url` and `is_safe_url` and follows redirects
hop-by-hop. A failed fetch is `ExternalServiceError` or
`ValidationError` and marks the row failed. `BackendFileServiceResolver`
lives in `core` so upload and the worker share one resolver without
`workers` importing `web`.

## Alternatives considered

- **Parse from the remote URL and leave `file_path` empty** — rejected:
  preview and download already require stored bytes.
- **Store bytes for every URL, including web pages** — rejected: a
  finance article is not a downloadable file, and preview must keep
  returning `knowledge.file_unavailable`.
- **Keep the resolver in `web/deps` and import it from the worker** —
  rejected: `workers` must not import `web`.
- **Download at create time** — rejected: create already classifies the
  type; the worker owns the job session that can persist `file_path`
  next to parse.

## Consequences

A `file_url` document can preview and download after the worker
succeeds. A web `url` row still has no bytes. A fetch failure stops on
a typed failed status instead of a silent pending spinner. The worker
and upload path share one storage resolver.

## Required verification

- `uv run pytest tests/core/knowledge/test_create_variants.py tests/workers/test_document_process.py tests/web/test_document_reads.py`
- `python scripts/verify_agent_notes.py --repo-root .`
- `uv run python scripts/check_layer_violation.py --domains auth,tenant,infra,knowledge,chat`
