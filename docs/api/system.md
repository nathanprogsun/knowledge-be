# `system` endpoints

Routes registered under `/api/v1/system`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/system/admin/api-keys` |
| POST | `/system/admin/api-keys` |
| DELETE | `/system/admin/api-keys/{key_id}` |
| GET | `/system/admin/audit-log` |
| GET | `/system/admin/list` |
| POST | `/system/admin/promote` |
| POST | `/system/admin/revoke` |
| GET | `/system/admin/runtime/queues` |
| GET | `/system/admin/runtime/queues/{queue}/tasks` |
| GET | `/system/admin/settings` |
| DELETE | `/system/admin/settings/{key}` |
| GET | `/system/admin/settings/{key}` |
| PUT | `/system/admin/settings/{key}` |
| POST | `/system/admin/tenants/apply-default-storage-quota` |
| POST | `/system/admin/users/reset-password` |
| POST | `/system/docreader/reconnect` |
| GET | `/system/info` |
| GET | `/system/parser-engines` |
| POST | `/system/parser-engines/check` |
| POST | `/system/storage-engine-check` |
| GET | `/system/storage-engine-status` |
