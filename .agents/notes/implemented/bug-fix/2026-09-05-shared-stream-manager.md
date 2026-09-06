# Agent Note: Share one in-process stream manager for stop

Status: implemented
Date: 2026-09-05
Scope: POST /sessions/{id}/stop must cancel the in-flight QA loop

## Context

`StopStreamService` and `ChatService` each received a fresh
`MemoryStreamManager`. `POST /sessions/{id}/stop` returned 200 and set
a cancel flag on a manager the running loop never saw, so tokens kept
arriving.

## Decision

`LifeSpanService` holds one APP-scope `MemoryStreamManager`. The lifespan
constructs it once. `get_stream_manager_from_lifespan` raises when it is
missing. `build_stop_stream_service` and `build_chat_service` take that
instance. `ChatService._stream_qa` races the queue drain against
`wait_cancelled(session_id, assistant.id)` and still completes the
assistant row with the buffered text. Stop does not take a
`message_reader`: the in-flight assistant shell is often uncommitted, so
a row lookup would 404 a valid stop.

## Alternatives considered

- **A Redis stream manager** — rejected: cancel is process-local today
  and a Redis backend would add an unused dependency.
- **Inject `message_reader` into stop** — rejected: the assistant row
  may not be visible yet, so ownership checks would 404 a live turn.
- **Wire `continue_stream` onto the same manager** — rejected: that
  path is a separate leftover and is not required for stop to cancel
  tokens.

## Consequences

Stop and chat in one process share cancel flags. Multi-worker stop still
cannot cross processes. Unit constructors may omit `stream_manager` and
then the drain does not race cancel.

## Required verification

- `uv run pytest tests/core/chat/test_factory.py tests/web/test_session_message_views.py tests/core/chat/test_service.py`
- `python scripts/verify_agent_notes.py --repo-root .`
