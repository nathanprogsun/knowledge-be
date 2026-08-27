---
name: kb-contract-alignment
description: Use when changing an API-bearing file — routers, request/response
  models, pagination, error hierarchy, or anything that alters the OpenAPI
  schema. Verifies the change propagates to the generated frontend types and
  that no consumer breaks.
---

# kb-contract-alignment

The FastAPI OpenAPI schema is the single source of truth for the API
contract (AGENTS.md §6). Field names are part of the contract; renaming
any of them is a breaking change for the frontend.

## Contract-bearing files

- `src/web/api/**/router.py`, `views.py` — response_model and request
  bodies.
- `src/core/contracts/*.py` — shared Pydantic models (frozen by
  convention; no longer gate-enforced).
- `src/common/pagination.py` — request `page`/`page_size` (capped at
  100, default 20); response `total`/`page`/`page_size`/`data`.
- `src/common/exception.py` — error payload `code`/`message`/`details`.

## Procedure

1. Make the backend change.
2. Run `make openapi` — exports `docs/api/openapi.json` and regenerates
   `frontend/src/api/__generated__/schema.ts`. Never hand-edit the
   generated file.
3. Run frontend `npm run type-check` (needs
   `NODE_OPTIONS=--max-old-space-size=6144`) — type errors surface every
   frontend consumer broken by the contract change.
4. Run `make check-schema` (DB model compatibility) when the change
   touches persistence.

## Notes

- The frontend CI lane re-runs the generation and fails on
  `git diff --exit-code`, so a stale or hand-edited `schema.ts` cannot
  merge.
- Frontend view-models (`UserInfo`, `TenantInfo`) intentionally differ
  from wire shapes — they normalize (string tenant ids, fabricated
  `owner_id`). Derive them from generated types via
  `Partial<Omit<...>>`; do not "fix" them to match the wire exactly.

## Drift that has shipped before

- `limit` instead of `page_size` on pagination requests.
- `items` instead of `data` on list responses.
- `details` as a runtime-only attribute instead of a declared field.
- `/auth/login` returning `active_tenant=None` where the SPA expects
  the resolved workspace (broke post-login routing until the membership
  fallback was added).
