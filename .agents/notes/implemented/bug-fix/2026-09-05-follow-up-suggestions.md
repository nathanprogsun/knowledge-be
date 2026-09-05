# Agent Note: Follow-up suggestions and message persistence

Status: implemented
Date: 2026-09-05
Scope: after-answer chips and chat history rows
Related files: src/core/chat/messages/suggestion_service.py, src/core/chat/messages/gateway.py, src/core/chat/factory.py, src/core/chat/service.py, frontend/src/api/message-suggestion.ts

## Context

Every completed chat turn called `POST /sessions/{id}/messages/{id}/suggestions`
and received `feature.not_implemented`. The toolbar flashed 加载推荐问题 and
vanished. `GET /messages/{session}/load` returned an empty list because the
chat gateway minted ids and never wrote rows.

## Decision

`PersistentMessageGateway` writes the user and assistant rows. The stream
bridge concatenates `AGENT_FINAL_ANSWER` chunks and completes the assistant
row. `MessageSuggestionService` generates three follow-up questions from the
turn text through the first KnowledgeQA model, or a template fallback. The
SPA sends `query` and `answer` on ensure because that POST is a later
request than the stream commit. Analytics `record_event` and
`validate_attribution` stay no-ops so a click cannot 501 the next send.

## Alternatives considered

- **Keep suggestions stubbed and hide the toolbar** — rejected: the operator
  already sees the loading row after every answer.
- **Wait for full message-history lease pipeline before chips** —
  rejected: the repository already has `acquire_generation`. The missing
  piece was turn text and a model call.
- **Generate only from stored messages** — rejected: the suggestions POST
  can race the stream commit.

## Consequences

A completed turn can show follow-up chips. Refreshing the session reloads
the persisted messages. Regenerating a ready set no longer binds an unused
`:generating` parameter. Vector hybrid search, ReAct, and suggestion
analytics stay later work.

## Required verification

- `uv run pytest tests/chat/test_suggestion_service.py tests/core/chat/test_follow_up_generator.py tests/core/chat/test_message_gateway.py tests/core/chat/test_factory.py tests/core/chat/test_service.py tests/web/test_session_message_views.py`
- Live: after a knowledge-chat answer, suggestions return 200 with questions
