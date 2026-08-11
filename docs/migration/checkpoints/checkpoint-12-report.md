# Checkpoint-12 (ARQ Worker Layer) Report

Generated: 2026-08-11T02:57:14Z
Master: ebca8a6

This checkpoint verifies the ARQ worker-layer stage: task registration
coverage, task-name alignment with the upstream task-type constants, the
workers-to-core dependency invariant, and frozen-contract stability. It
also records the current anti-drift gate results so later stages can
attribute new regressions against this baseline.

## Merged deliverables (this milestone)

- ARQ worker infrastructure: `src/workers/` with registry, base worker
  context, settings, and the `main` entrypoint that serves the full task
  set.
- 19 task handlers across 19 task modules, each registering itself at
  import time through the `register_task` decorator:
  chunk extract, document process, datasource sync, datatable summary,
  FAQ import, image multimodal, index delete, KB clone, KB delete,
  knowledge list delete, knowledge list reparse, knowledge move,
  knowledge post process, manual process, question generation, summary
  generation, temporary document process, wiki finalize, wiki ingest.
- Per-task payload models, outcome serialization, and injection seams
  that delegate to core services; handlers remain thin (parse, validate,
  delegate, shape the result).

## Worker invariants

1. **Task registration count: 19 / 19.** `registry.all_functions()`
   returns exactly 19 ARQ `Function` entries, one per upstream task-type
   constant.
2. **Task-name alignment with upstream: 15 / 19 exact; 4 differ.**
   Four registered names use an underscore separator where the upstream
   constant uses a colon:

   | Registered name | Upstream constant |
   |---|---|
   | `document_process` | `document:process` |
   | `image_multimodal` | `image:multimodal` |
   | `manual_process` | `manual:process` |
   | `datasource_sync` | `datasource:sync` |

   The 15 colon-separated names match the upstream constants verbatim.
   The four deviations are covered by the unit suite, so changing them
   is a deliberate rename decision for a later stage, not a silent drift.
3. **Workers-to-core dependency: no direct db/ai service invocation.**
   All task modules route external work through core services. Four
   type-only seam imports reference db/ai symbols for annotations and
   injected protocols:

   - `src/workers/tasks/chunk_extract.py` — `src.ai.embedding.TaskContext`,
     `src.ai.llm.Chat` (annotation / protocol surface only)
   - `src/workers/tasks/image_multimodal.py` — `src.ai.embedding.TaskContext`
     (background-task context holder passed into the core service)
   - `src/workers/tasks/kb_delete.py` — `src.db.models.knowledge.Document`
     (repository-seam protocol annotation only)

   None of these invoke db/ai services directly. Note the automated layer
   gate does not flag them: its workers rule forbids only `web` / `db`
   prefixes and matches against the dotted path as-is, so `src.`-prefixed
   modules and `ai` imports fall outside the enforced set.
4. **Frozen contracts unchanged.** Contract invariants pass (249 frozen
   models, 1659 body statements scanned) and the field-drift check is
   clean.

## Anti-drift gates (full-coverage agent check)

| Gate | Result | Count |
|---|---|---|
| check_layer_violation | FAIL | 143 violations (3 in `src/workers/`) |
| check_service_singleton | FAIL | 22 violations |
| check_endpoint_coverage | FAIL | 205 mismatches |
| check_schema_compatibility | FAIL | 169 violations |
| check_contract_invariants | PASS | 249 frozen models, 1659 stmts |
| check_imports | FAIL | 1 violation |
| check_field_drift | PASS | no drift |

The three `src/workers/` layer violations are `Any`/`object` return
annotations in the task registry protocol and the document-process task;
they are new in this milestone. Every other failing gate is a pre-existing
baseline item (details below).

## Anti-drift gates (CI: `make check`)

| Gate | Result | Count |
|---|---|---|
| check-layer | FAIL | 3 violations (web layer only) |
| check-singleton | PASS | 56 request-scoped service classes |
| check-endpoint | PASS | 50 routes / 50 doc endpoints |
| check-schema | PASS | nothing to check (no `migrations/` dir) |
| check-contract | PASS | 249 frozen models, 1659 stmts |
| check-imports | PASS | 734 files, all top-level imports |
| check-sql | FAIL | 11 unsafe SQL constructions (db layer) |
| check-pr-leak | FAIL | 1 comment-hygiene violation |
| check-map-from-db | FAIL | 1 DTO-projection violation |
| check-exception-types | FAIL | 48 unsanctioned raises (core / ai layers) |

None of the `make check` failures touch `src/workers/`.

## Known pre-existing failures

- Layer gate (web): `src/web/deps/embed_channels.py` imports a db model;
  `src/web/deps/rbac.py` and `src/web/middleware/auth.py` contain
  `Any`/`object` annotations.
- Endpoint-coverage gate: 205 doc/route mismatches across the web API
  surface; a subset of doc tables is not fully aligned.
- Schema gate: 169 model/column family mismatches and missing table models
  for several migration tables.
- Service-singleton gate: 22 services that the request-scope rule does not
  yet cover.
- SQL gate: 11 interpolated-fragment constructions in four db DAOs.
- Exception-type gate: 48 raises in core/ai layers that are not sanctioned
  exception subclasses.
- Import gate: 1 indented import in generated gRPC stubs
  (`src/ai/docreader/proto/docreader_pb2_grpc.py`).
- Comment-hygiene gate: 1 marker leak in `tests/core/agent/test_data_tools.py`.
- DTO-projection gate: 1 JSON-shaped model without a `from_json`
  classmethod in the knowledge-base types.

All of the above predate this milestone; none are attributable to the
workers stage.

## Test counts

```
298 passed, 2 warnings in 3.58s   (tests/workers/)
```
