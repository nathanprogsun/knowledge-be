# Agent Note: KB initialization config save

Status: implemented
Date: 2026-09-04
Scope: persist the SPA editor's save-and-close model and chunking payload
Related files: src/web/api/infra/initialization/kb_config.py, src/core/knowledge/knowledge_bases/service/kb_model_config.py

## Context

The knowledge-base editor's save-and-close path writes name and indexing
through `PUT /knowledge-bases/{id}`, then writes models and chunking
through `PUT /initialization/config/{kb_id}`. The second route was
absent, so the first write succeeded and the UI still failed. Chunking
never reached storage.

## Decision

Register `GET` and `PUT /initialization/config/{kb_id}` on the
initialization router. The handlers translate the SPA camelCase body,
require the LLM (and embedding when set) through `ModelService`, count
documents through `KnowledgeService`, and persist via
`KBService.update_model_config`. Changing the embedding model is
refused once documents exist so existing vectors stay readable.
`POST /initialization/initialize/{kb_id}` stays unimplemented.

## Alternatives considered

- **Fold the payload into `PUT /knowledge-bases/{id}`** — rejected: the
  SPA already calls the initialization path; changing the client would
  still leave that URL as a 404 for other callers.
- **Keep a nested `APIRouter` in `kb_config.py`** — rejected: the
  feature-map and endpoint-coverage scanners only see decorators on the
  module that declares `APIRouter(prefix=...)`.

## Consequences

Save-and-close can persist models and `chunking_config`. The editor
keeps its existing two-request sequence. The legacy full-config
initialize route is still missing.

## Required verification

- `uv run pytest tests/core/knowledge/test_kb_model_config.py tests/core/knowledge/test_kb_service_crud.py tests/integration/web/api/initialization/test_kb_config.py`
- `make openapi` and `make check-endpoint`
- Live `PUT /api/v1/initialization/config/{kb_id}` is no longer 404
