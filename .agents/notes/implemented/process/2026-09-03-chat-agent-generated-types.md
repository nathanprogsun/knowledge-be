# Agent Note: Chat and agent clients use generated types

Status: implemented
Tags: frontend, openapi, generated-types
Date: 2026-09-03
Scope: how chat and agent frontend clients bind to OpenAPI schemas
Related files: frontend/src/api/chat/index.ts, frontend/src/api/chat/streame.ts, frontend/src/api/agent/index.ts, frontend/src/api/chat-history.ts, frontend/src/api/message-suggestion.ts, frontend/src/api/tenant/index.ts, frontend/src/api/system/index.ts

## Context

`frontend/src/api/auth` already aliases `components["schemas"]`. Chat and
agent clients still declared wire shapes by hand, so a schema change
could pass `make openapi` and still leave the SPA on stale fields.

## Decision

Chat session/message envelopes, knowledge-QA stream bodies, chat-history
stats/search, suggestion envelopes, and agent/IM wire types now alias
the generated schema. Local view-models stay only where the wire type
is missing or the UI needs extra fields (`Partial<Omit<...>>` plus
explicit unions). Temporary attachments and WeChat QR payloads have no
named schema yet, so they remain client-side types.

`vue-tsc` also required two compatibility aliases outside the chat/agent
clients: `SystemInfo` maps to `SystemInfoWireData`, and `TenantAPIKey`
keeps the list-row fields (`api_key`, capability unions) the settings
pages already use.

## Alternatives considered

- **Replace every consumer with raw `Schema['Agent']`** — rejected: the
  editor and lists require required nested suggestion objects and
  `creator_name`, which the wire type does not guarantee.
- **Migrate every `frontend/src/api/*` module in one change** — rejected:
  `vue-tsc` cascades; the plan stop-loss is chat/agent first.

## Consequences

New OpenAPI fields on these schemas flow into the clients after
`make openapi`. Hand-written `export interface` under
`frontend/src/api/chat` and `frontend/src/api/agent` is gone.

## Required verification

- `make frontend-typecheck`
- `make frontend-test`
- `make openapi` then `git diff --exit-code -- frontend/src/api/__generated__/schema.ts`
- `rg "^export interface " frontend/src/api/chat frontend/src/api/agent`
