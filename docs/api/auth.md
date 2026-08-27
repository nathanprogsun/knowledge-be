# `auth` endpoints

Routes registered under `/api/v1/auth`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| POST | `/auth/auto-setup` |
| POST | `/auth/change-password` |
| GET | `/auth/config` |
| POST | `/auth/invitations/lookup` |
| POST | `/auth/login` |
| POST | `/auth/logout` |
| GET | `/auth/me` |
| PUT | `/auth/me/preferences` |
| GET | `/auth/oidc/callback` |
| GET | `/auth/oidc/config` |
| GET | `/auth/oidc/url` |
| POST | `/auth/refresh` |
| POST | `/auth/register` |
| POST | `/auth/register-by-invite` |
| GET | `/auth/tenant` |
| GET | `/auth/validate` |
