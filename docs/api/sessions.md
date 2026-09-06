# `sessions` endpoints

Routes registered under `/api/v1/sessions`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/sessions` |
| POST | `/sessions` |
| DELETE | `/sessions/batch` |
| DELETE | `/sessions/{session_id}` |
| GET | `/sessions/{session_id}` |
| PUT | `/sessions/{session_id}` |
| DELETE | `/sessions/{session_id}/messages` |
| GET | `/sessions/{session_id}/messages/{message_id}/suggestions` |
| POST | `/sessions/{session_id}/messages/{message_id}/suggestions` |
| DELETE | `/sessions/{session_id}/pin` |
| POST | `/sessions/{session_id}/pin` |
| POST | `/sessions/{session_id}/stop` |
| POST | `/sessions/{session_id}/suggestion-events` |
