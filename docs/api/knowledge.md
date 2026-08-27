# `knowledge` endpoints

Routes registered under `/api/v1/knowledge`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| POST | `/knowledge/batch-delete` |
| POST | `/knowledge/batch-reparse` |
| POST | `/knowledge/move` |
| GET | `/knowledge/move/progress/{task_id}` |
| PUT | `/knowledge/tags` |
| DELETE | `/knowledge/{id}` |
| GET | `/knowledge/{id}` |
| PUT | `/knowledge/{id}` |
| POST | `/knowledge/{id}/cancel-parse` |
| POST | `/knowledge/{id}/clone` |
| POST | `/knowledge/{id}/reparse` |
