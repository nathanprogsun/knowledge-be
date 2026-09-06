# `organizations` endpoints

Routes registered under `/api/v1/organizations`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/organizations` |
| POST | `/organizations` |
| POST | `/organizations/join` |
| POST | `/organizations/join-by-id` |
| POST | `/organizations/join-request` |
| GET | `/organizations/preview/{code}` |
| GET | `/organizations/search` |
| DELETE | `/organizations/{id}` |
| GET | `/organizations/{id}` |
| PUT | `/organizations/{id}` |
| POST | `/organizations/{id}/invite` |
| POST | `/organizations/{id}/invite-code` |
| GET | `/organizations/{id}/join-requests` |
| PUT | `/organizations/{id}/join-requests/{request_id}/review` |
| POST | `/organizations/{id}/leave` |
| GET | `/organizations/{id}/members` |
| DELETE | `/organizations/{id}/members/{tenant_id}` |
| PUT | `/organizations/{id}/members/{tenant_id}` |
| POST | `/organizations/{id}/request-upgrade` |
| GET | `/organizations/{id}/search-tenants` |
| GET | `/organizations/{id}/search-users` |
| GET | `/organizations/{id}/shares` |
| GET | `/organizations/{id}/shared-knowledge-bases` |
