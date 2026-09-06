# Agent Note: In-process builtin document reader

Status: implemented
Date: 2026-09-07
Scope: worker default document parse is in-process; no external reader service required
Related files: src/core/knowledge/documents/builtin_reader.py, src/core/knowledge/documents/process_runtime.py, src/core/system/parser_engine.py, src/core/knowledge/documents/process_document.py, src/core/knowledge/documents/parse_pipeline.py, tests/core/knowledge/test_builtin_reader.py, tests/core/system/test_parser_engine.py

## Context

The worker default opened a gRPC channel to a docreader process. A local checkout without that process could not parse ordinary text. The engine registry also marked `builtin` unavailable when docreader was disconnected, so the UI offered no local default even though `simple` already existed as a no-service engine.

## Decision

`BuiltinDocumentReader` is the worker default. It implements `DocumentReader` and dispatches through an extension-to-handler table (`HANDLERS`). Phase-1 handlers are stdlib-only (text, markdown, csv, json, html, plus a filename-only image stub). Empty or `builtin` `parser_engine` uses every handler; `simple` omits html; any other engine name raises `document_parse.engine_unavailable`.

The `builtin` registry spec has `requires_docreader=False` and an empty `unconfigured_reason`. Advertised `file_types` match handlers that exist. `DocReaderAdapter` stays in tree for explicit inject; `aclose` still closes it when that is the wired reader.

A missing pipeline reader stamps `document reader is not configured`. Knowledge-base `parser_engine` is not copied onto `ReadRequest` in this change.

## Alternatives considered

- **Keep docreader as the default and add builtin as an opt-in** — rejected: the goal is a checkout that parses without a third-party reader process.
- **If/elif per format inside `read`** — rejected: each new type would grow a branch; the handler table is the extension point.
- **Delete `DocReaderAdapter` in the same change** — rejected: remote engines still need that seam; unwiring the default is enough.
- **Add pypdf / python-docx now** — rejected: those are later handler rows; claiming pdf/docx in the registry before handlers exist would lie to the UI.
- **Pull WeKnora's docreader image into compose** — rejected: this product must not depend on that third-party reader.

## Consequences

Workers parse md/txt/csv/json/html (and image stubs) with no gRPC client. pdf/docx/epub stay unsupported until handlers exist. SPA copy that still describes builtin as a DocReader-backed complex-format engine will drift from the registry description until it is updated. URL-only reads still need bytes loaded by the parse pipeline; the builtin reader does not fetch.

## Required verification

- `uv run pytest tests/core/knowledge/test_builtin_reader.py tests/core/system/test_parser_engine.py -q`
- `python scripts/verify_agent_notes.py --repo-root .`
