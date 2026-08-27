# `messages` endpoints

Routes registered under `/api/v1/messages`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/messages/chat-history-stats` |
| POST | `/messages/search` |
| GET | `/messages/{session_id}/load` |
| DELETE | `/messages/{session_id}/{message_id}` |
