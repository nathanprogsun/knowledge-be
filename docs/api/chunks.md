# `chunks` endpoints

Routes registered under `/api/v1/chunks`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/chunks/by-id/{id}` |
| DELETE | `/chunks/by-id/{id}/questions` |
| PUT | `/chunks/by-id/{id}/questions` |
| POST | `/chunks/by-id/{id}/questions/regenerate` |
| DELETE | `/chunks/{knowledge_id}` |
| GET | `/chunks/{knowledge_id}` |
| DELETE | `/chunks/{knowledge_id}/{id}` |
| PUT | `/chunks/{knowledge_id}/{id}` |
| POST | `/chunks/{knowledge_id}/{id}/revert` |
| GET | `/chunks/{knowledge_id}/{id}/revisions` |
