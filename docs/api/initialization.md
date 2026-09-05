# `initialization` endpoints

Routes registered under `/api/v1/initialization`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| POST | `/initialization/asr/check` |
| GET | `/initialization/config/{kb_id}` |
| PUT | `/initialization/config/{kb_id}` |
| POST | `/initialization/embedding/test` |
| POST | `/initialization/extract/fabri-tag` |
| POST | `/initialization/extract/fabri-text` |
| POST | `/initialization/extract/text-relation` |
| POST | `/initialization/multimodal/test` |
| GET | `/initialization/ollama/download/progress/{task_id}` |
| GET | `/initialization/ollama/download/tasks` |
| GET | `/initialization/ollama/models` |
| POST | `/initialization/ollama/models/check` |
| POST | `/initialization/ollama/models/download` |
| GET | `/initialization/ollama/status` |
| POST | `/initialization/remote/check` |
| POST | `/initialization/rerank/check` |
