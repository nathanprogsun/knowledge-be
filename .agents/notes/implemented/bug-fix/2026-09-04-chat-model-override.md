# Agent Note: Composer model can stand in for agent model_id

Status: implemented
Date: 2026-09-04
Scope: let a request-level chat model send when the builtin agent has no model_id
Related files: src/core/chat/sessions/agent_qa.py, frontend/src/components/Input-field.vue

## Context

Auto-setup seeds `builtin-quick-answer` with an empty `model_id`. The
composer still shows a tenant KnowledgeQA model (mimo-v2.5) and passes
it as `summary_model_id`. Send was blocked in the SPA, and the backend
rejected the same turn before considering the override.

## Decision

If the agent has no `model_id`, a valid request `summary_model_id` is
the chat model for that turn. The composer readiness check skips the
missing-agent-model reason when the dropdown already has a model.

## Alternatives considered

- **Require saving `model_id` on the builtin agent first** — rejected:
  PUT currently refuses every builtin edit, and the composer already
  picked a model.
- **Treat any tenant KnowledgeQA model as implicit** — rejected: the
  request must name the model the user selected.

## Consequences

Quick-answer send works after auto-setup without an agent-editor save.
A turn with no agent model and no override still fails.

## Required verification

- `uv run pytest tests/core/chat/test_agent_qa.py`
- Live creatChat send with mimo-v2.5 selected creates a session (2xx)
