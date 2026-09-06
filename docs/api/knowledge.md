# `knowledge` endpoints

Routes registered under `/api/v1/knowledge`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/knowledge/batch` |
| GET | `/knowledge/search` |
| POST | `/knowledge/batch-delete` |
| POST | `/knowledge/batch-reparse` |
| POST | `/knowledge/move` |
| GET | `/knowledge/move/progress/{task_id}` |
| PUT | `/knowledge/tags` |
| DELETE | `/knowledge/{id}` |
| GET | `/knowledge/{id}` |
| PUT | `/knowledge/{id}` |
| GET | `/knowledge/{id}/download` |
| GET | `/knowledge/{id}/preview` |
| POST | `/knowledge/{id}/cancel-parse` |
| POST | `/knowledge/{id}/clone` |
| POST | `/knowledge/{id}/regenerate-summary` |
| POST | `/knowledge/{id}/reparse` |
| GET | `/knowledge/{id}/spans` |
