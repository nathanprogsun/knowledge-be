# Checkpoint-2 Report — Stage 2 (infrastructure)

PR-20.5 · `chore(checkpoint-2): verify endpoint coverage for infrastructure stage`

## Scope

Closes the stage-2 integration checkpoint: the stage-2 routers were
already mounted in `app_context/lifespan.py` and the per-domain service
dependencies already registered through `web/deps/` (each infra domain
has one deps module re-exported from the package `__init__`), and the
alembic migration chain was already numbered `0009`~`0014`
(`0009_models` … `0014_datasources`). What this checkpoint adds is the
**anti-drift gate coverage** for the stage-2 domains, which previously
scanned only the stage-1 surface:

- `check_layer_violation` now scans all 10 domains (was 92 files, now
  151) — stage-2 files under `core/infra/<domain>/`,
  `web/api/infra/<domain>/`, `db/models/infra/<domain>/` and their
  `db/dao/<domain>*_repository.py` DAOs were previously invisible to
  the layer/typing gate.
- `check_endpoint_coverage` now also verifies the stage-2 domains that
  have an aligned upstream `docs/api/*.md` table (was 24 routes, now
  50).
- `make check` now runs all six gates, including the previously orphaned
  `check-schema` target.

## Results

| Check | Result |
| --- | --- |
| `check_layer_violation` (10 domains) | PASS — 151 files |
| `check_service_singleton` | PASS — 19 Service classes request-scoped |
| `check_endpoint_coverage` (auth,tenants,vector_stores,storage_backends,web_search) | PASS — 50 routes ↔ 50 docs (bidirectional) |
| `check_schema_compatibility` | WARN (exit 0) — Go SQL migrations not present; this project migrates via alembic |
| `check_contract_invariants` | PASS — 249 models frozen, 1657 body stmts |
| `check_imports` | PASS — 196 files, top-level only |

## Fixes applied during this checkpoint

1. **`scripts/check_endpoint_coverage.py`** — the stage-2 infra routes
   live one level deeper than the stage-1 domains
   (`web/api/infra/<domain>/`); the domain filter now matches on the
   second segment for `infra/`. The docs-file alias table was extended
   for the hyphenated/singular upstream file names
   (`model.md`, `mcp-service.md`, `vector-store.md`,
   `storage-backend.md`, `web-search.md`).
2. **`scripts/check_layer_violation.py`** — the domain filter now
   resolves `core/infra/<domain>/`, `web/api/infra/<domain>/`,
   `db/models/infra/<domain>/` and the singular DAO stems
   (`mcp_service_repository` → domain `mcp_services`, …).
3. **39 `Any`/`object` annotations removed** across
   `mcp_services` (36) and `models` (3) — the same sweep checkpoint-1
   did for stage 1, now enforced on stage-2 code. Highlights:
   - the `_ConnectionManagerLike` protocols in `connectivity.py` /
     `discovery.py` now use the real `MCPSession` / `JSONRPCResponse`
     types instead of `object`;
   - `OAuthManager` auth-config helpers take `JsonObject | None`
     instead of `object | None`;
   - `MCPServiceService._build_update_columns` returns
     `dict[str, SqlValue]` (its `updated_at` column carries a
     `datetime`), dropping the `cast("BindParams", …)` at the call site;
   - MCP router endpoints return `dict[str, JsonValue]` (or
     `MCPOAuthAuthorizeURLResponse` / `None` for the 204 and envelope
     cases) instead of `object`;
   - `ModelDebugEnvelope.data` is `dict[str, JsonValue]`.
4. **`Makefile`** — `check` now includes `check-schema`; `check-layer`
   and `check-endpoint` pass the expanded domain set.

## Endpoint-coverage findings (this checkpoint closed)

The three fully aligned stage-2 domains now gate on `make check`:

- **vector_stores** — 8 routes ↔ `vector-store.md` 8 entries
- **storage_backends** — 9 routes ↔ `storage-backend.md` 9 entries
- **web_search** — 9 routes ↔ `web-search.md` 9 entries

(plus the stage-1 `auth` / `tenants` 24 routes, unchanged).

## Known gaps (recorded, not blocking — see checkpoint-1 for the stage-1
precedent)

| Domain | Gap | Direction |
| --- | --- | --- |
| `datasources` | 15 routes fully match the Go `@Router` set, but the upstream `docs/api/*.md` has no datasource table (authoritative docs live in `docs/swagger.json`), so the md-based gate cannot verify it | not checkable |
| `initialization` | 4 upstream docs entries unimplemented: `GET/PUT /initialization/config/{kb_id}`, `POST /initialization/initialize/{kb_id}` (depend on the KB domain, stage 4), `POST /initialization/extract/text-relation` (extraction service); plus 1 local route absent from docs: `POST /initialization/asr/check` (exists in Go `@Router`) | mixed |
| `mcp_services` | 3 local routes absent from docs: `POST /mcp-services/{id}/oauth/authorize-url`, `GET …/oauth/status`, `DELETE …/oauth/token` (PR-17.5b live-OAuth management surface, Go does client-side OAuth only). Upstream docs entry `POST /agent/tool-approvals/{pending_id}` waits for the agent domain | extra routes |
| `models` | 1 local route absent from docs: `POST /models/{id}/debug` (debug probe) | extra route |
| `system` | carried over from checkpoint-1: `/system/admin/*` absent from docs; `/system/info`, `/system/parser-engines*`, `/system/docreader/*`, `/system/storage-engine*` wait for the docreader/parser infra | mixed |

All of the above are documented in the upstream Go repo and are expected
to be closed when the owning domains land or the upstream
`docs/api/*.md` tables are updated; none blocks stage-2 merge. The
`make check` gate covers every domain that is fully aligned today.

## Verification

```
make lint && make typecheck && make test && make check
```

All green: 1759 tests pass (110 skipped without Docker), mypy clean
(273 files), ruff lint + format clean.
