# Checkpoint-1 Report — Stage 1 (auth + tenants + system)

PR-10.5 · `chore(checkpoint-1): verify layer and singleton invariants for auth + tenants`

## Scope

Checks the auth / tenants / system domains for the four anti-drift
invariants: layered architecture, singleton registration, frozen-contract
integrity, and endpoint coverage. The domain-scoped runner
(`--domains auth,tenants,system`) is used so future-stage WIP
(unimplemented agent/infra/knowledge stubs) does not pollute the result.

## Results

| Check | Result |
| --- | --- |
| `check_layer_violation` (auth,tenants,system) | PASS — 68 files |
| `check_service_singleton` | PASS — all Service classes registered |
| `check_endpoint_coverage` (auth,tenants) | PASS — 24 routes ↔ 24 docs (bidirectional) |
| `check_contract_invariants` | PASS — 228 models frozen, 1497 body stmts |
| `check_imports` | PASS — 89 files, top-level only |

`make check` runs all five gates to green.

## Fixes applied during this checkpoint

1. **`JsonObject` / `JsonValue` / `BindParams` aliases** (`src/common/json.py`)
   — replaced every `dict[str, object]` / `Any` JSON annotation across
   `src/`. Pydantic's recursive `JsonValue` is used so schema generation
   terminates; `BindParams` adds `datetime` for SQL bindparams.
2. **`db/models` de-cluttering** — moved `insert_sql_column_list`
   overrides into a base `db_generated_columns` ClassVar on `TableModel`;
   moved `TenantInvitation.is_share_link` / `is_expired` business methods
   into module-level functions in the tenants service layer. `db/models`
   are now declarative only.
3. **`LifeSpanService` relocated to `src/app_context/registry.py`** —
   broke the `lifespan -> routers -> web.deps -> lifespan` import cycle so
   router imports could move to module top (removing function-level
   imports). All 2 `*Service` classes are registered.
4. **Function-level imports removed** — `lifespan.py`, `auth/router.py`,
   `system_setting_repository.py`.

## Endpoint-coverage findings (this checkpoint closed)

Auth and tenants endpoint coverage is now **bidirectional and complete**
(24 routes ↔ 24 doc entries). The following were implemented as part of
this checkpoint to close documented stage-1 gaps that prior PRs had
deferred:

- **auth**: `register`, `me`, `change-password`, `validate`, and the three
  OIDC endpoints (`config`, `url`, `callback`).
- **tenants**: api-keys CRUD, api-principal-config GET/PUT, tenant KV
  GET/PUT, and `GET /tenants` (current user's visible workspaces).

### Known gaps (recorded, not blocking — see below)

System-domain endpoint coverage has two residual classes that are
intentionally **not** treated as failures:

1. **`/system/admin/settings*` + `/system/admin/audit-log`** — implemented
   in PR-11 but absent from `docs/api/system.md`. The frozen doc file does
   not enumerate the admin surface (it lives in `docs/swagger.json`).
   These are documented in `docs/api/system.md` in a later stage.
2. **`/system/info`, `/system/parser-engines*`, `/system/docreader*`,
   `/system/storage-engine*`** — listed in `docs/api/system.md` but depend
   on Stage 2+ infra services (ModelService, AI docreader). Deferred to
   their owning PRs.

Both are tracked in this report; neither blocks stage-1 merge.

## Contract changes (per AGENTS.md §10.7)

Two frozen contracts were corrected to match the Go wire shape (both were
real inconsistencies found during endpoint implementation):

- `tenants.TenantAPIKey` — corrected to the Go `tenantAPIKeyResponse`
  shape (`id: int`, `scope_type`, `full_access`, `capabilities`, ...),
  removing the spurious `role` / `token` / `key_prefix` fields.
- `tenants.CreateAPIKeyRequest` — `role` → `full_access` + `capabilities`,
  matching the Go create request.
- `auth.RegisterResponse.active_tenant` — `Tenant` → `Tenant | None`,
  matching Go's nullable `*TenantResponse`.

## New migration

`alembic/versions/0008_tenant_kv.py` — creates the `tenant_kv` table
(generic per-tenant JSON key-value store backing `/tenants/kv/{key}`).

## Verification

```
make lint && make typecheck && make test && make check
```
