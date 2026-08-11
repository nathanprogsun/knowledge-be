# Checkpoint-11 (Organization + Channels Stage) Report

Generated: 2026-08-11T10:50:00Z
Master: ebca8a6

This checkpoint validates the organization + channels delivery:
OrganizationService, EmbedChannelService, and IMChannelService
registration, the embed-channel authentication surface (publish-token
resolution, origin gating, rate limits), and the frozen-contract
invariants — compared against the upstream contracts in the read-only
reference source under the workspace root.

## Merged deliverables (this milestone)

- OrganizationService — request-scoped, assembled per request on the
  shared session via `src.core.organizations.factory`
- EmbedChannelService + anonymous embed session flow — publish/session
  token auth, origin gating, three-tier rate limits, HMAC-signed
  session handles
- IMChannelService + command registry — slash-command dispatch and the
  callback path
- Embed channel views + deps (`web/api/channels/embed/router.py`,
  `web/deps/embed_channels.py`)
- IM channel views + deps (`web/api/channels/im/router.py`,
  `web/deps/im_channels.py`)
- Organization views + deps (`web/api/organizations/router.py`,
  `web/deps/organizations.py`)

## Anti-drift gates (`make check`)

- **check-layer**: FAIL — 3 violations (2 pre-existing baseline in
  rbac.py / auth.py; 1 new in `web/deps/embed_channels.py:60` — a web
  deps file imports a db model for type annotations)
- **check-singleton**: PASS — all 56 Service classes are request-scoped
  (absent from the app-scope lifespan) and no router injects a Service
  class directly
- **check-endpoint**: PASS — 50 FastAPI routes and 50 doc endpoints
  match (full bidirectional coverage)
- **check-schema**: PASS — no `migrations/` root in this checkout;
  nothing to check (exit 0)
- **check-contract**: PASS — 249 models frozen, 1659 body stmts scanned
- **check-imports**: PASS — 734 files, all imports at top level
- **check-sql**: FAIL — 11 pre-existing unsafe-SQL violations
  (knowledge_span / message / session / wiki_page repositories)
- **check-pr-leak**: FAIL — 1 pre-existing PR-id leak in
  `tests/core/agent/test_data_tools.py:1079`
- **check-map-from-db**: FAIL — 1 pre-existing DTO-projection violation
  (`core/knowledge/knowledge_bases/types.py:70`)
- **check-exception-types**: FAIL — 48 pre-existing unsanctioned
  exception violations

## Agent-only anti-drift suite (workspace check)

- **check_layer_violation**: FAIL — 143 violations (superset of the
  `make check-layer` set; includes the AI-layer Any/object annotations
  and the docreader gencode)
- **check_service_singleton**: FAIL — 22 violations (upstream script
  semantics: expects services registered on the app-scope lifespan;
  this repo's scope discipline is the opposite — request-scoped
  services — so the local `check-singleton` is authoritative and
  passes)
- **check_endpoint_coverage**: FAIL — 205 mismatches (upstream doc
  table not aligned for the org / channels domains)
- **check_schema_compatibility**: FAIL — 169 violations (upstream
  script diffs against the upstream SQL migrations; the local
  `check-schema` has no `migrations/` root in this checkout)
- **check_contract_invariants**: PASS — 249 models frozen, 1659 body
  stmts scanned
- **check_imports**: FAIL — 1 violation (generated
  `docreader_pb2_grpc.py` indented import)
- **check_field_drift**: OK

## Service registration verification

- OrganizationService → `OrganizationServiceDep`
  (`web/deps/organizations.py`) — request-scoped
- EmbedChannelService → `EmbedChannelServiceDep`
  (`web/deps/embed_channels.py`) — request-scoped
- IMChannelService → `IMChannelServiceDep`
  (`web/deps/im_channels.py`) — request-scoped
- `check-singleton`: PASS — all 56 Service classes request-scoped,
  absent from the app-scope lifespan

## Embed-channel auth verification

- Publish-token resolution: constant-time compare
  (`hmac.compare_digest`); session tokens are distinguished by prefix
  and resolved via the Redis-backed store
- Origin gating: exact match (case-insensitive), `*.suffix` subdomain
  wildcard, literal `"*"` blanket; an empty allowlist rejects all
- Rate limits: per-IP per-minute, channel-global per-minute
  (`max(per_ip * 20, 120)`), channel-daily; Redis sliding-window Lua
  counter; fail-open when no Redis is wired
- Session handles: HMAC-SHA256 signed, keyed by the publish token;
  rotating the publish token invalidates outstanding handles

## Test counts

```
354 passed in 9.11s
```

(`tests/core/organizations/` + `tests/core/channels/`)

## Field drift

```
[check_field_drift] OK
```

## CI / Makefile status

- `Makefile` already defines the `check` target (all ten anti-drift
  gates); no change needed
- `.github/workflows/ci.yml` already runs `make check` in the
  quality-gate job; no change needed

## New violations attributable to this milestone

- `check-layer`: `web/deps/embed_channels.py:60` imports
  `src.db.models.embed_channel` (web layer must not import `db`). The
  file's docstring states "web never imports db"; the import is used
  only for type annotations. Follow-up: narrow the annotation to a
  protocol or move the type into `core`.

All other failing gates have pre-existing baseline explanations from
earlier milestones.
