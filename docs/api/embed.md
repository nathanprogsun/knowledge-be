# `embed` endpoints

Routes registered under `/api/v1/embed`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| POST | `/embed/{channel_id}/agent-chat/{session_id}` |
| GET | `/embed/{channel_id}/chunks/{chunk_id}` |
| GET | `/embed/{channel_id}/config` |
| POST | `/embed/{channel_id}/exchange` |
| GET | `/embed/{channel_id}/files` |
| POST | `/embed/{channel_id}/knowledge-chat/{session_id}` |
| GET | `/embed/{channel_id}/messages/{session_id}/load` |
| POST | `/embed/{channel_id}/sessions` |
| POST | `/embed/{channel_id}/sessions/{session_id}/events` |
| POST | `/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}` |
| POST | `/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel` |
| POST | `/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url` |
| GET | `/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status` |
| GET | `/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions` |
| POST | `/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions` |
| POST | `/embed/{channel_id}/sessions/{session_id}/stop` |
| POST | `/embed/{channel_id}/sessions/{session_id}/suggestion-events` |
| POST | `/embed/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}` |
| GET | `/embed/{channel_id}/suggested-questions` |
