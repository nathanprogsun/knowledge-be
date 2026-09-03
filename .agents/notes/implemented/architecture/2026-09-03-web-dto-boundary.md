# Agent Note: Web layer consumes service DTOs

Status: implemented
Tags: layering, dto, web, favorites, chat, embed, sharing
Date: 2026-09-03
Scope: web views no longer import storage table models
Related files: src/web/api/favorites/views.py, src/web/api/chat/messages/views.py, src/web/api/channels/embed/views.py, src/web/api/organizations/shared_views.py

## Context

Four web view modules imported `src.db.models` and projected table
rows onto the HTTP contract. That bypassed the service DTO boundary
and made `check-layer` lie about product domains: the layer rule
says `web` never imports `db`, but those four files did.

## Decision

Services return core DTOs; web views consume only those DTOs.

- Favorites: `FavoriteInfo.map_from_db`.
- Messages: `MessageServiceImpl` returns `MessageInfo`.
- Embed admin get: `EmbedChannelOwnedInfo` carries `publish_token`
  and `has_webhook_secret` without exposing `webhook_secret`.
- Agent shares: `AgentShareInfo.map_from_db`.
- Session create/update take display fields as keywords and return
  `SessionInfo`, so routers no longer construct `Session` rows.

## Alternatives considered

- **Keep table models in web and only expand `check-layer` later** —
  rejected: the gate would stay false-green and every new view would
  copy the leak.
- **Return raw rows from services and wrap them only in routers** —
  rejected: the leak would move to routers and still import `db`.
- **One mega DTO per domain that includes every secret column** —
  rejected: embed publish tokens stay admin-only; a second owned
  projection is narrower than leaking secrets on every read.

## Consequences

`rg "from src.db.models" src/web` is empty. Product-domain
`check-layer` can tell the truth. Message `rendered_content` stays
storage-only and is no longer on `MessageInfo`.

## Required verification

- `rg "from src.db.models" src/web` has no matches.
- `uv run pytest tests/web/test_favorites_views.py tests/core/chat/test_message_service.py tests/web/test_session_message_views.py tests/web/test_embed_views.py`
- `make check-map-from-db`
- `python scripts/check_layer_violation.py --src-root src/ --domains auth,tenants,system,datasources,initialization,mcp_services,models,storage_backends,vector_stores,web_search,favorites,chat,organizations,channels,knowledge,knowledge_bases,agents,evaluation,sharing,me,files,cloud`
