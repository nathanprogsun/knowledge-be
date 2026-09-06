# Agent Note: Wire leftover operator-loop seams

Status: implemented
Date: 2026-09-04
Scope: agent-chat, knowledge-search, builtin config PUT, regenerate-summary
Related files: src/core/chat/factory.py, src/core/chat/sessions/keyword_searcher.py, src/core/agents/service/custom_agent_service.py, src/core/knowledge/documents/summary_refresh.py

## Context

The login-to-answer loop already streamed a real knowledge-chat reply.
Four operator-visible holes remained. Agent-chat raised
`chat.agent_qa_not_wired`. Command palette `POST /knowledge-search`
raised `chat.search_not_wired`. Saving a builtin agent always returned
409 even for `config.model_id`. Regenerate-summary flipped
`summary_status` to pending with no worker, so the drawer spun.

## Decision

Agent-chat reuses `KnowledgeQARunner` until the ReAct engine factory is
composed. Knowledge search ranks stored text chunks by keyword overlap.
Builtin updates may change `config` and must leave name, description,
and avatar alone. A preset that only exists in the registry is
inserted on the first config save so PUT is not a 404. The SPA editor
used to PUT the full agent row. The write client now keeps only name,
description, avatar, and config so `extra="forbid"` does not 422 first. Regenerate-summary runs `process_summary` with the KB
summary model, or the first tenant KnowledgeQA model. A missing
refresher is an error, not a queued job.

## Alternatives considered

- **Compose `run_agent_qa` and `AgentEngine` now** — rejected: that
  path still needs a tool registry, rerank, and history loader. The
  operator needs a streamed answer first.
- **Leave search unwired until hybrid retrieval lands** — rejected: the
  command palette already calls the endpoint and swallows the 501 as
  an empty list.
- **Keep regenerate-summary as pending** — rejected: the UI polls
  forever.

## Consequences

智能体 send streams the same grounded answer as 快速问答. Palette
search shows keyword hits from parsed files. The builtin editor can
store `model_id`. The summary button writes `description` or returns a
clear 422. Vector hybrid search and the ReAct loop stay later work.

## Required verification

- `uv run pytest tests/core/chat/test_factory.py tests/core/chat/test_keyword_searcher.py tests/core/agent/test_custom_service.py tests/core/knowledge/test_summary_refresh.py tests/core/knowledge/test_doc_service_crud.py tests/core/knowledge/test_summary_question.py tests/integration/web/api/agents/test_controller.py`
- Live palette search, builtin save, regenerate-summary, and 智能体 send
