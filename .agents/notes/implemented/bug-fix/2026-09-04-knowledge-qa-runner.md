# Agent Note: Knowledge chat now answers from stored chunks

Status: implemented
Date: 2026-09-04
Scope: wire the knowledge-QA runner so `/knowledge-chat` streams a real answer
Related files: src/core/chat/sessions/knowledge_qa_runner.py, src/core/infra/models/chat_service.py, src/core/chat/factory.py

## Context

`POST /knowledge-chat/{session_id}` already returned 200 and opened SSE
with `agent_query`. `build_chat_service` still injected `_NotWiredRunner`,
so the turn raised `chat.knowledge_qa_not_wired` and the SPA stayed on
正在思考. Hybrid search is also unwired. The live turn already names a
parsed file via `knowledge_ids`.

## Decision

Knowledge QA loads enabled text chunks for the turn's files (or every
document in named KBs), caps them, and runs
`INTO_CHAT_MESSAGE` then `CHAT_COMPLETION_STREAM`. A request
`summary_model_id` that is type `KnowledgeQA` wins. The agent config,
the first KB `summary_model_id`, and the first tenant KnowledgeQA model
follow. `AGENT_COMPLETE` is emitted after the pipeline. Agent-chat and
HTTP hybrid search stay stubbed.

`ChatModelService` lives in infra like `EmbeddingService`. It keeps
stored API keys. The runner talks to it through `ChatModelCatalog` so
infra does not import chat `Context`.

## Alternatives considered

- **Send knowledge-chat through `run_agent_qa`** — rejected: that is
  the ReAct loop. Knowledge-chat is RAG / pure chat.
- **Wait for `SearchParallelPlugin` and the retrieve registry** — rejected:
  a missing plugin is a silent no-op, so RAG would stream an empty
  user message. Stored chunks already exist for the operator-loop file.
- **Reuse `ModelService.get_model` for the live client** — rejected: that
  path redacts `api_key`.

## Consequences

A knowledge-chat turn with a completed document and a KnowledgeQA model
streams `references`, `answer`, and `complete`. Semantic
`POST /knowledge-search` and agent-chat remain unwired.

The typewriter snaps to the full answer when the turn is already
complete. Background tabs barely receive `requestAnimationFrame`, so a
late `done` frame would otherwise leave the answer stuck on the first
few characters.

## Required verification

- `uv run pytest tests/core/infra/models/test_chat_service.py tests/core/chat/test_knowledge_qa_runner.py tests/core/chat/test_factory.py`
- Live creatChat send with the 基金研报 file attached streams an answer
