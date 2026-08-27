# `datasource` endpoints

Routes registered under `/api/v1/datasource`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/datasource` |
| POST | `/datasource` |
| GET | `/datasource/logs/{log_id}` |
| GET | `/datasource/types` |
| POST | `/datasource/validate-credentials` |
| DELETE | `/datasource/{id}` |
| GET | `/datasource/{id}` |
| PUT | `/datasource/{id}` |
| GET | `/datasource/{id}/logs` |
| POST | `/datasource/{id}/pause` |
| POST | `/datasource/{id}/resource-ancestors` |
| GET | `/datasource/{id}/resources` |
| POST | `/datasource/{id}/resume` |
| POST | `/datasource/{id}/sync` |
| POST | `/datasource/{id}/validate` |
