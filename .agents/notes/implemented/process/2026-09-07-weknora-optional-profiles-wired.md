# Agent Note: WeKnora optional profiles wired for real use

Status: implemented
Date: 2026-09-07
Scope: sandbox, MCP server, Dex, Langfuse, and ODL-hybrid are compose profiles with a callable path in this product
Related files: compose.yaml, deploy/env/compose.env, deploy/docker/Dockerfile, deploy/docker/Dockerfile.sandbox, deploy/docker/Dockerfile.odl-hybrid, deploy/dex/config.yaml, mcp-server/, src/core/agents/skills/factory.py, src/core/agents/engine/sandbox/types.py, src/core/knowledge/documents/builtin_reader.py

## Context

Compose previously listed MinIO/Qdrant-style sidecars but left sandbox, MCP, Dex, Langfuse, and ODL out. Skills defaulted to Docker image `wechatopenai/kb-sandbox:latest`, a port-time rename of WeKnora's sandbox that this repo never published or built. Skills factory ignored `KB_SANDBOX_MODE`. PDF had no OpenDataLoader path without the WeKnora docreader image.

## Decision

- **Sandbox.** Default image is `knowledge-be-sandbox:local`, built from `deploy/docker/Dockerfile.sandbox` (same contents as WeKnora's skill sandbox: Python 3.11 + Node 20). Compose profile `sandbox`/`full` builds that image. `KB_SANDBOX_MODE` (default `local`) and `KB_SANDBOX_DOCKER_IMAGE` drive `build_skills_manager`. api/worker mount docker.sock so mode `docker` can `docker run` the local image.
- **MCP.** Vendored `mcp-server/` (WeKnora package) with `KNOWLEDGE_BE_BASE_URL` / `KNOWLEDGE_BE_API_KEY` env (legacy `WEKNORA_*` still accepted). Profile `mcp`/`full` serves HTTP on `:8082`.
- **Dex.** Profile `dex`/`full` runs Dex with `deploy/dex/config.yaml` for local OIDC; API OIDC flags stay off until the operator enables them.
- **Langfuse.** Profile `langfuse` runs the self-hosted stack. Export uses existing OTEL (`OTEL_ENABLED` + `OTEL_EXPORTER=otlp` to Langfuse's `/api/public/otel`), not a separate Langfuse SDK.
- **ODL-hybrid.** Profile `odl-hybrid` builds the Docling hybrid service. Builtin reader handles `pdf` via `opendataloader_pdf.convert(..., hybrid_url=...)` when `ODL_HYBRID_URL` is set and the worker image was built with `WITH_ODL=1`. No WeKnora docreader image.

## Alternatives considered

- **Keep `wechatopenai/kb-sandbox:latest`** — rejected: that tag is not published for this product; pulls fail.
- **Pull `wechatopenai/weknora-sandbox` as the default** — rejected: skills runtime should be built from this repo so we do not depend on WeKnora Hub tags for core compose.
- **Ship Langfuse Python SDK** — rejected: OTEL is already the observability seam; Langfuse accepts OTLP.
- **Reintroduce WeKnora docreader for PDF** — rejected: product constraint; ODL client-in-worker is the PDF path.

## Consequences

`docker compose --profile full up --build` brings sandbox image, MCP, and Dex alongside the earlier sidecars. PDF needs `WITH_ODL=1` rebuild plus `--profile odl-hybrid`. Langfuse needs a UI key + OTEL env after the profile is up. MCP needs a tenant API key in `KNOWLEDGE_BE_API_KEY`.

## Required verification

- `docker compose config --services` (default) and `docker compose --profile full config --services`
- `uv run pytest tests/core/system/test_parser_engine.py tests/core/knowledge/test_builtin_reader.py -q`
- `python scripts/verify_agent_notes.py --repo-root .`
