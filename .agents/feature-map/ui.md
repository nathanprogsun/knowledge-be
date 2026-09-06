# UI feature map

Manual overlay on `.agents/feature-map/generated.json`. Each row is
`route or query` → Vue file → frontend API module → generated.json key
(`METHOD path`).

`generated.json` only indexes `@router.*` in `**/router.py`. Screens
that call routes registered on another `APIRouter`, or in `*_views.py`,
list the nearest key that is present.

## Top-level routes

| Surface | Route | Vue | API module | generated.json |
| --- | --- | --- | --- | --- |
| Login / invite register | `/login`, `/register` | `frontend/src/views/auth/Login.vue` | `frontend/src/api/auth/index.ts` | `POST /api/v1/auth/login` |
| Workspace onboarding | `/onboarding/workspace` | `frontend/src/views/auth/WorkspaceOnboarding.vue` | `frontend/src/api/tenant/index.ts` | `POST /api/v1/tenants` |
| Knowledge base list | `/platform/knowledge-bases` | `frontend/src/views/knowledge/KnowledgeBaseList.vue` | `frontend/src/api/knowledge-base/index.ts` | `GET /api/v1/knowledge-bases` |
| Knowledge base detail | `/platform/knowledge-bases/:kbId` | `frontend/src/views/knowledge/KnowledgeBase.vue` | `frontend/src/api/knowledge-base/index.ts` | `GET /api/v1/knowledge-bases/{id}` |
| Agent list | `/platform/agents` | `frontend/src/views/agent/AgentList.vue` | `frontend/src/api/agent/index.ts` | `GET /api/v1/agents` |
| New chat | `/platform/creatChat` | `frontend/src/views/creatChat/creatChat.vue` | `frontend/src/api/chat/index.ts` | `POST /api/v1/sessions` |
| Chat thread | `/platform/chat/:chatid` | `frontend/src/views/chat/index.vue` | `frontend/src/api/chat/index.ts` | `GET /api/v1/sessions/{session_id}` |
| Organizations | `/platform/organizations` | `frontend/src/views/organization/OrganizationList.vue` | `frontend/src/api/organization/index.ts` | `GET /api/v1/organizations` |
| Settings shell | `/platform/settings?section=` | `frontend/src/views/settings/Settings.vue` | see sections below | `GET /api/v1/system/info` |

`/platform/integrations` and `/platform/system/*` redirect into
Settings sections. `/knowledgeBase` is a legacy alias of the KB editor.
Public embed chat is not a route in this SPA.

## Settings sections

Query: `/platform/settings?section=<key>`. Keys come from
`frontend/src/views/settings/Settings.vue` and
`frontend/src/config/settingsAccess.ts`. Integration tabs use
`integration-<tab>` (`im`, `embed`, `api`, `chrome`, `claw`).

| Section | Vue | API module | generated.json |
| --- | --- | --- | --- |
| `general` | `frontend/src/views/settings/GeneralSettings.vue` | `frontend/src/stores/settings.ts` | `GET /api/v1/auth/me` |
| `userprofile` | `frontend/src/views/settings/UserProfile.vue` | `frontend/src/api/auth/index.ts` | `GET /api/v1/auth/me` |
| `tenant` | `frontend/src/views/settings/TenantInfo.vue` | `frontend/src/api/tenant/index.ts` | `GET /api/v1/tenants/{tenant_id}` |
| `members` | `frontend/src/views/settings/TenantMembers.vue` | `frontend/src/api/tenant/members.ts` | `GET /api/v1/tenants/{tenant_id}` |
| `models` | `frontend/src/views/settings/ModelSettings.vue` | `frontend/src/api/model/index.ts` | `GET /api/v1/models` |
| `ollama` | `frontend/src/views/settings/OllamaSettings.vue` | `frontend/src/api/initialization/index.ts` | `GET /api/v1/initialization/ollama/status` |
| `cloud` | `frontend/src/views/settings/CloudSettings.vue` | `frontend/src/api/model/index.ts` | `GET /api/v1/models/cloud/status` |
| `websearch` | `frontend/src/views/settings/WebSearchSettings.vue` | `frontend/src/api/web-search-provider.ts` | `GET /api/v1/web-search-providers` |
| `chathistory` | `frontend/src/views/settings/ChatHistorySettings.vue` | `frontend/src/api/chat-history.ts` | `GET /api/v1/messages/chat-history-stats` |
| `vectorstore` | `frontend/src/views/settings/VectorStoreSettings.vue` | `frontend/src/api/vector-store.ts` | `GET /api/v1/vector-stores` |
| `parser` | `frontend/src/views/settings/ParserEngineSettings.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/tenants/kv/{key}` |
| `storage` | `frontend/src/views/settings/StorageEngineSettings.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/storage-backends` |
| `mcp` | `frontend/src/views/settings/McpSettings.vue` | `frontend/src/api/mcp-service.ts` | `GET /api/v1/mcp-services` |
| `integration-im` | `frontend/src/components/IMChannelPanel.vue` | `frontend/src/api/agent/index.ts` | `GET /api/v1/agents` |
| `integration-embed` | `frontend/src/components/AgentEmbedChannelPanel.vue` | `frontend/src/api/embed/index.ts` | `GET /api/v1/embed-channels` |
| `integration-api` | `frontend/src/views/integrations/ApiIntegrationSettings.vue` | `frontend/src/api/tenant/index.ts` | `GET /api/v1/tenants/{tenant_id}/api-principal-config` |
| `system` | `frontend/src/views/settings/SystemInfo.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/system/info` |
| `system-global` | `frontend/src/views/system/SystemSettings.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/system/admin/settings` |
| `runtime-queues` | `frontend/src/views/system/RuntimeQueues.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/system/admin/settings` |
| `platform-api-keys` | `frontend/src/views/system/PlatformAPIKeys.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/tenants/{tenant_id}/api-keys` |
| `system-audit-log` | `frontend/src/views/system/SystemAuditLog.vue` | `frontend/src/api/system/index.ts` | `GET /api/v1/system/admin/audit-log` |

## Knowledge-base tabs

Route: `/platform/knowledge-bases/:kbId?tab=`. Tabs are declared in
`KnowledgeBase.vue` as `documents` / `wiki` / `graph`.

| Tab | Vue | API module | generated.json |
| --- | --- | --- | --- |
| `documents` (default) | `frontend/src/views/knowledge/KnowledgeBase.vue` | `frontend/src/api/knowledge-base/index.ts` | `GET /api/v1/knowledge-bases/{id}/knowledge` |
| `wiki` | `frontend/src/views/knowledge/wiki/WikiBrowser.vue` | `frontend/src/api/wiki/index.ts` | `GET /api/v1/knowledgebase/{kb_id}/wiki/index` |
| `graph` | `WikiBrowser.vue` (`?tab=graph`) | `frontend/src/api/wiki/index.ts` | `GET /api/v1/knowledgebase/{kb_id}/wiki/graph` |

FAQ editing is a drawer on the documents tab
(`frontend/src/views/knowledge/components/FAQEntryManager.vue`), not
its own route. Map: `GET /api/v1/knowledge-bases/{id}/faq/entries`.

## Do not split in this round

`KnowledgeBase.vue`, `AgentEditorModal.vue`, and `FAQEntryManager.vue`
stay as shells. Navigate through this table instead of reading the
whole file.
