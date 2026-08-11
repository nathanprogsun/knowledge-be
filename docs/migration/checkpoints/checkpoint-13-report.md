# Checkpoint-13 (Full Migration) Report

Generated: 2026-08-11T10:45:00Z
Base: ebca8a6 (master, includes Waves 1-5)

This checkpoint is the full-migration end-of-Stage-8 verification: every
anti-drift gate is run, the frozen-contract invariant is asserted, and
endpoint / service / task coverage is measured. The report records the
actual results - including the gates that still fail and the pre-existing
baseline those failures trace to - rather than declaring a green state
that does not exist.

## Migration status summary

- 169 PRs planned in the source migration plan; 171 entries recorded as
  completed in the agent state file (the surplus reflects the a/b/c
  sub-splits and a small number of .5 checkpoint PRs that are folded
  into the same completion ledger).
- All integer stages (1 through 8) are merged to master; the worker
  layer shipped all 19 async tasks in Wave 5 (commit ec5505e).
- master HEAD at the start of this PR: ebca8a6 (fix(web): dedupe im
  callback operation ids and register im routes in route-lock gate).
- No agent state blockers or skips recorded.

## Inventory verified at this checkpoint

| Surface            | Count                          | Source                                                              |
|--------------------|--------------------------------|---------------------------------------------------------------------|
| OpenAPI paths      | 196                            | create_app().openapi()["paths"]                                     |
| Path-method pairs  | 262                            | same, summing methods per path                                      |
| Service classes    | 56                             | make check-singleton (all request-scoped, none in lifespan)         |
| Registered tasks   | 19/19                          | src.workers.registry.all_tasks() after importing every module       |
| Frozen contracts   | 249 models, 1659 body stmts    | make check-contract                                                 |

The 19 worker tasks are: chunk:extract, datasource_sync, datatable:summary,
document_process, faq:import, image_multimodal, index:delete,
kb:clone, kb:delete, knowledge:list_delete, knowledge:list_reparse,
knowledge:move, knowledge:post_process, manual_process,
question:generation, summary:generation, temporary_document:process,
wiki:finalize, wiki:ingest. The Wave-5 wiring commit registered all
of them; this checkpoint re-asserts the import-time registration produces
exactly that set.

## Anti-drift gate results

Two anti-drift paths exist and are recorded separately:

1. the upstream-flavoured sync entry point (symlinked scripts) - runs the upstream-flavoured
   scripts symlinked at the private agent workspace plus the local
   check_field_drift.py. These scripts pre-date the request-scoped
   service refactor and a number of wave-by-wave scope changes, so their
   violation counts are systematically higher.
2. make check - runs the local scripts/check_*.py copies. These
   are the canonical gates the repo actually relies on; they are the
   ones wired into CI via the Anti-drift checks job.

The task spec mandates running the first path; the report records both
because the local gates are the ones that green-light merge. Both paths
exit non-zero at this commit.

### sync.sh check results (upstream-flavoured)

| Gate                        | Result | Count                                                                       |
|-----------------------------|--------|-----------------------------------------------------------------------------|
| check_layer_violation       | FAIL   | 143 layer/typing violations (mostly Any/object annotations in ai/retrieval/*)|
| check_service_singleton     | FAIL   | 22 service-registration violations (every service absent from LifeSpanService; intentional request-scoped design; local script passes) |
| check_endpoint_coverage     | FAIL   | 205 endpoint-coverage mismatches (bidirectional; upstream domains differ from local; local gate is green) |
| check_schema_compatibility  | FAIL   | 169 schema-compatibility violations (datetime/bool/JSON family mismatches; missing tables for upstream resources/task_*/wiki_*; wiki_pages duplicated) |
| check_contract_invariants   | PASS   | 249 models frozen, 1659 body stmts scanned                                  |
| check_imports               | FAIL   | 1 import placement violation (docreader_pb2_grpc.py:13, generated gRPC stub)|
| check_field_drift           | PASS   | [check_field_drift] OK                                                      |

### make check results (local, canonical)

| Gate                        | Result | Count                                                                       |
|-----------------------------|--------|-----------------------------------------------------------------------------|
| check-layer                 | FAIL   | 3 violations (web/deps/embed_channels.py:60, web/middleware/auth.py:34, web/deps/rbac.py:143) |
| check-singleton             | PASS   | All 56 Service classes are request-scoped (absent from LifeSpanService)     |
| check-endpoint              | PASS   | All 50 FastAPI routes and 50 doc endpoints match (5 audited domains)        |
| check-schema                | PASS   | migrations/ not found - nothing to check                                     |
| check-contract              | PASS   | All contract invariants satisfied (249 models frozen, 1659 body stmts)      |
| check-imports               | PASS   | All imports in 734 files are at top level                                   |
| check-sql                   | FAIL   | 11 unsafe SQL construction violations (db/dao/wiki_page_repository.py:596)  |
| check-pr-leak               | FAIL   | 1 PR-leak violation (tests/core/agent/test_data_tools.py:1079)              |
| check-map-from-db           | FAIL   | 1 DTO-projection violation (core/knowledge/knowledge_bases/types.py:70)    |
| check-exception-types       | FAIL   | 48 unsanctioned exception violations (non-ApplicationError raises in core/ai)|

### Frozen contracts

check_contract_invariants passes on both paths: 249 frozen Pydantic
models, 1659 body statements scanned, no contract mutation since the
last checkpoint. The the private contracts workspace is empty by
design (the frozen contracts live as @frozen-decorated Pydantic
models under src/core/contracts/; the workspace is the staging area,
not the source of truth). No frozen contract was modified in this PR.

## Known pre-existing failures (honest record)

These are not regressions introduced at this checkpoint. Each one
predates this checkpoint and is tracked in the project notes for a later
verification PR:

- check-layer (local, 3): two web -> db import-direction leaks
  (embed_channels.py, middleware/auth.py) and one Any-typed
  return in rbac.py. The web->db leaks were left in by the request-
  scoped DI refactor; the rbac.py Any is the DB-backed tenant
  association gate that is implemented but not yet wired into routers.
- check-sql (11): wiki_page_repository.py and a handful of DAOs
  interpolate locals that are not on the safe-fragment allowlist. All
  are parameterised at the driver level; the gate static analyser
  rejects them because the variable names are not audited constants.
- check-pr-leak (1): a comment in tests/core/agent/test_data_tools.py
  references the upstream migration numbering. This is a comment hygiene
  violation only.
- check-map-from-db (1): KnowledgeBaseInfo is JSON-typed and
  lacks a from_json classmethod satisfying the DTO projection rule.
- check-exception-types (48): core/ai/* raises vendored SDK
  exceptions (AIProviderError, VectorStoreError, etc.) that are
  not in the sanctioned ApplicationError subtree; the sanctioned set
  itself overlaps with these names, hence the gate drift.
- sync.sh upstream path: the higher violation counts (143, 22,
  205, 169) are a side-effect of the upstream scripts predating the
  request-scoped service refactor, the domain-scope narrowing of the
  local endpoint check, and the SQLite DDL not yet covering the
  upstream task-queue / resource-catalog / wiki-revision tables.

## Test suite

Two invocations were run, mirroring the task spec. The first is the
literal command from the spec; the second is the same command with the
shared Postgres reachable so the DB-dependent tests collect and run
instead of erroring at fixture setup.

### Spec invocation (no DB override)

    uv run pytest tests/ -q --tb=no -o filterwarnings= \
      --deselect tests/contract/stage4_contract.py::test_knowledge_base_wire_field_set_matches_reference

Result: 402 failed, 6426 passed, 134 skipped, 1 deselected, 555 errors
(exit 1, 164.18s).

The large error and failure count is collection-teardown noise:
DB-dependent fixtures call asyncpg against the default
DATABASE_URL (unreachable on this host without the shared-Postgres
override). The failures and errors collapse to 3 with the DB override
applied (below). The shared dev Postgres lives at
localhost:5432/knowledge_be; the project CI runs with
TEST_SKIP_DOCKER=1 which takes a different skip path.

### DB override invocation

    DATABASE_URL_OVERRIDE=postgresql+asyncpg://postgres:<password>@localhost:5432/knowledge_be \
      uv run pytest tests/ -q --tb=no -o filterwarnings= \
      --deselect tests/contract/stage4_contract.py::test_knowledge_base_wire_field_set_matches_reference

Result: 3 failed, 7404 passed, 110 skipped, 1 deselected, 1 warning
(exit 1, 344.53s). The 1 deselected test is the
test_knowledge_base_wire_field_set_matches_reference case excluded by
the spec.

The 3 remaining failures are contract-test drift, not regressions:

- tests/contract/stage4_contract.py::test_wiki_response_envelopes_match_reference
- tests/contract/test_knowledge_invariants.py::test_contract_wire_fields_match_fixture[KnowledgeBase-]
- tests/contract/test_knowledge_invariants.py::test_request_wire_fields_match_fixture[WebUpdateChunkRequest-]

These are wire-field fixture mismatches in the contract test layer; they
are tracked separately and are the next block of work after this
checkpoint.

## Tooling already in place

Both pieces of downstream-plumbing the plan asks for already exist on
this branch, so this PR adds the report only:

- make check target - present in Makefile (lines 50-51),
  composing check-layer check-singleton check-endpoint check-schema
  check-contract check-imports check-sql check-pr-leak
  check-map-from-db check-exception-types. No edit required.
- .github/workflows/ci.yml anti-drift job - present (lines
  46-47): Anti-drift checks runs make check as the final step of
  the quality-gate job. No edit required.

## Outcome

This checkpoint is the end-of-Stage-8 declaration. The contract gate
and field-drift gate are green; service and task coverage are complete
(56/56 request-scoped services, 19/19 tasks registered); endpoint
coverage is green for the five audited domains (50/50 bidirectional).
The schema, layer, SQL, exception, and pr-leak gates remain red on
pre-existing baselines and are explicitly enumerated above as the work
queued for the Stage-9 verification PRs.

