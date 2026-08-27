# `agents` endpoints

Routes registered under `/api/v1/agents`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/agents` |
| POST | `/agents` |
| GET | `/agents/placeholders` |
| GET | `/agents/type-presets` |
| GET | `/agents/{agent_id}/embed-channels` |
| POST | `/agents/{agent_id}/embed-channels` |
| GET | `/agents/{agent_id}/im-channels` |
| POST | `/agents/{agent_id}/im-channels` |
| GET | `/agents/{agent_id}/shares` |
| POST | `/agents/{agent_id}/shares` |
| DELETE | `/agents/{agent_id}/shares/{share_id}` |
| PUT | `/agents/{agent_id}/shares/{share_id}` |
| DELETE | `/agents/{id}` |
| GET | `/agents/{id}` |
| PUT | `/agents/{id}` |
| POST | `/agents/{id}/copy` |
| GET | `/agents/{id}/suggested-questions` |
