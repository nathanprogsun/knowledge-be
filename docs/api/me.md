# `me` endpoints

Routes registered under `/api/v1/me`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/me/invitations` |
| POST | `/me/invitations/{invitation_id}/accept` |
| POST | `/me/invitations/{invitation_id}/decline` |
| GET | `/me/invitations/pending-count` |
