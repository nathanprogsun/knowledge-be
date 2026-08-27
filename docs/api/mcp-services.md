# `mcp-services` endpoints

Routes registered under `/api/v1/mcp-services`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/mcp-services` |
| POST | `/mcp-services` |
| DELETE | `/mcp-services/{service_id}` |
| GET | `/mcp-services/{service_id}` |
| PUT | `/mcp-services/{service_id}` |
| POST | `/mcp-services/{service_id}/oauth/authorize-url` |
| GET | `/mcp-services/{service_id}/oauth/status` |
| DELETE | `/mcp-services/{service_id}/oauth/token` |
| GET | `/mcp-services/{service_id}/resources` |
| POST | `/mcp-services/{service_id}/test` |
| GET | `/mcp-services/{service_id}/tool-approvals` |
| PUT | `/mcp-services/{service_id}/tool-approvals/{tool_name}` |
| GET | `/mcp-services/{service_id}/tools` |
