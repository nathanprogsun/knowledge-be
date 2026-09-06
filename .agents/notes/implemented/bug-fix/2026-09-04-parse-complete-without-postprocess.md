# Agent Note: Complete parse when enrichment is unwired

Status: implemented
Date: 2026-09-04
Scope: mark a successful parse completed unless a post-process dispatcher exists
Related files: src/core/knowledge/documents/process_document.py

## Context

A successful parse with text chunks stayed `processing` so enrichment
could fan out. Production has no post-process dispatcher. The document
became queryable (`enable_status=enabled`, chunks stored) but the SPA
kept showing 解析中 because it treats `processing` as in-flight.

## Decision

`_finalize` keeps `processing` only when text chunks exist *and* a
post-process dispatcher is bound. Otherwise the row is `completed`.
That matches the upstream "no enrichment → complete immediately" path.

## Alternatives considered

- **Treat `processing` + `enabled` as done in the SPA only** — rejected:
  the card would still say 解析中, and every poll would look unfinished.
- **Ship a no-op post-process job that flips completed** — rejected:
  extra queue hop for a seam that is not composed yet.

## Consequences

Local parses leave the spinner when chunks land. When enrichment is
wired later, the dispatcher path still leaves the row `processing`
until that job finishes.

## Required verification

- `uv run pytest tests/core/knowledge/test_process_document.py`
- Reparse a live document with no post-process worker and confirm
  `parse_status=completed`
