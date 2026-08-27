# `knowledgebase` endpoints

Routes registered under `/api/v1/knowledgebase`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| POST | `/knowledgebase/{kb_id}/wiki/auto-fix` |
| GET | `/knowledgebase/{kb_id}/wiki/folders` |
| POST | `/knowledgebase/{kb_id}/wiki/folders` |
| DELETE | `/knowledgebase/{kb_id}/wiki/folders/{folder_id}` |
| PUT | `/knowledgebase/{kb_id}/wiki/folders/{folder_id}` |
| GET | `/knowledgebase/{kb_id}/wiki/graph` |
| GET | `/knowledgebase/{kb_id}/wiki/index` |
| GET | `/knowledgebase/{kb_id}/wiki/lint` |
| PUT | `/knowledgebase/{kb_id}/wiki/move-page` |
| GET | `/knowledgebase/{kb_id}/wiki/pages` |
| POST | `/knowledgebase/{kb_id}/wiki/pages` |
| DELETE | `/knowledgebase/{kb_id}/wiki/pages/{slug:path}` |
| GET | `/knowledgebase/{kb_id}/wiki/pages/{slug:path}` |
| PUT | `/knowledgebase/{kb_id}/wiki/pages/{slug:path}` |
| POST | `/knowledgebase/{kb_id}/wiki/rebuild-links` |
| GET | `/knowledgebase/{kb_id}/wiki/search` |
| GET | `/knowledgebase/{kb_id}/wiki/stats` |
