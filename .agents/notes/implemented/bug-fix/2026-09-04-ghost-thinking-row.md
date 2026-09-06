# Agent Note: Drop the leftover 正在思考 assistant row

Status: implemented
Date: 2026-09-04
Scope: one knowledge-chat turn must render one assistant row
Related files: frontend/src/composables/useChatStreamHandler.ts

## Context

A completed knowledge-chat answer still left an empty incomplete
assistant above it. That row has no event stream, so
`RagPipelineProgress` stays on 正在思考. Stream frames all carry the
same request id, but the first shell often has only an assistant
message id. Later chunks then miss it and push a second row.

## Decision

Resolve the in-flight assistant by request id, then by the trailing
incomplete row. On complete, stop, or error, drop any other empty incomplete assistant.
Hide an empty incomplete agent row when a completed sibling already
exists.

## Alternatives considered

- **Wait for message persistence so ids always match** — rejected: the
  drawer is already showing two rows.
- **Hide every incomplete agent row** — rejected: the first wait state
  is the one we want while the model is still running.

## Consequences

A finished turn shows the answer without a stuck thinking card. An
in-flight turn still shows 正在思考 on the single shell.

## Required verification

- `cd frontend && npx tsx --test src/composables/useChatStreamHandler.test.ts`
- Live creatChat send leaves one assistant row after `agent_complete`
